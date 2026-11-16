// Runs the peerinfer-core engine off the main thread.
//
// init: loads an MLC q4f16_1 model. Files come from the cache that
//   cacheModelWeights fills, or from the network.
// generate: runs one chat request and posts text chunks, then the counts.
// stop: ends the running request after its current batch of tokens.
import { Tokenizer } from "@huggingface/tokenizers";
import init, { Engine } from "../engine-wasm/peerinfer_engine.js";
import { prepareGeneration } from "./generation";
import { MODEL_CACHE } from "../model-cache";
import type { CompletionRequest } from "../types";

let engine: Engine | undefined;
let tokenizer: Tokenizer | undefined;
let stopRequested = false;
let current = -1;

/** Token IDs that end a reply: <|im_end|> and <|endoftext|>. */
const STOP = new Set([2, 0]);

async function file(base: string, name: string): Promise<Response> {
  const url = new URL(name, base).href;
  const cached = await (await caches.open(MODEL_CACHE)).match(url);
  if (cached) return cached;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response;
}

async function load(base: string) {
  await init();
  const [chatConfig, tensorCache, tokenizerJson, tokenizerConfig] =
    await Promise.all(
      [
        "mlc-chat-config.json",
        "tensor-cache.json",
        "tokenizer.json",
        "tokenizer_config.json",
      ].map(async (n) => (await file(base, n)).text()),
    );
  const records = (
    JSON.parse(tensorCache) as { records: { dataPath: string }[] }
  ).records;
  const shards = await Promise.all(
    records.map(
      async (r) => new Uint8Array(await (await file(base, r.dataPath)).arrayBuffer()),
    ),
  );
  engine = await Engine.load(chatConfig, tensorCache, shards);
  tokenizer = new Tokenizer(JSON.parse(tokenizerJson), JSON.parse(tokenizerConfig));
  return { adapter: engine.adapter() };
}

async function generate(id: number, model: string, request: CompletionRequest) {
  if (!engine || !tokenizer) throw new Error("The model is not loaded.");
  const tk = tokenizer;
  const { tokens, max, temperature } = prepareGeneration(tk, request);
  stopRequested = false;
  current = id;
  engine.reset();
  const seed = crypto.getRandomValues(new Uint32Array(1))[0];
  engine.setSampling(temperature, 1, seed);

  const output: number[] = [];
  let text = "";
  let done = false;
  // Decode the growing output so split UTF-8 sequences are not sent as
  // replacement characters.
  const accept = (token: number) => {
    if (done) return;
    if (STOP.has(token) || output.length >= max) {
      done = true;
      return;
    }
    output.push(token);
    const decoded = tk
      .decode(output, { skip_special_tokens: true, clean_up_tokenization_spaces: false })
      .replace(/�+$/, "");
    if (!decoded.startsWith(text))
      throw new Error("Tokenizer produced an unstable text prefix.");
    const delta = decoded.slice(text.length);
    if (delta) self.postMessage({ id, chunk: delta });
    text = decoded;
  };

  accept(await engine.forward(Uint32Array.from(tokens)));
  if (!done && !stopRequested && max > 1) {
    await engine.generateStream(max - 1, (batch: Uint32Array) => {
      for (const t of batch) accept(t);
      return !done && !stopRequested;
    });
  }
  return { model, prompt_tokens: tokens.length, completion_tokens: output.length };
}

// Requests run one after another: a stopped request may still be finishing
// its last batch when the next one arrives.
let queue: Promise<void> = Promise.resolve();

self.onmessage = ({ data }) => {
  if (data.method === "stop") {
    // Ignore a stop for a request that already finished.
    if (data.id === current) stopRequested = true;
    return;
  }
  queue = queue.then(() => handle(data));
};

async function handle(data: { id: number; method: string; args: any }) {
  const { id, method, args } = data;
  try {
    const result =
      method === "init"
        ? await load(args.base)
        : method === "generate"
          ? await generate(id, args.model, args.request)
          : (() => {
              throw new Error("Unknown worker method");
            })();
    self.postMessage({ id, result });
  } catch (e) {
    self.postMessage({ id, error: e instanceof Error ? e.message : String(e) });
  }
}
