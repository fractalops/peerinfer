// Runs the peerinfer-core engine off the main thread.
//
// Whole model:
//   init {base}: loads an MLC q4f16_1 model from the weight cache or the network.
//   generate {model, request}: runs one chat request, posts text chunks, then counts.
//   stop: ends the running request after its current batch of tokens.
// Stages:
//   shardsFor {chatConfig, cache, start, end}: shard files that layers start..end need.
//   loadStage {chatConfig, cache, shards, start, end}: loads layers start..end.
//   runTokens {tokens} / runHidden {hidden}: one step; returns a token or hidden states.
//   reset, sampling {temperature, seed}.
import { Tokenizer } from "@huggingface/tokenizers";
import init, { Engine } from "../engine-wasm/peerinfer_engine.js";
import { prepareGeneration } from "./generation";
import { readModelFile } from "../model-cache";
import type { CompletionRequest } from "../types";

let engine: Engine | undefined;
let tokenizer: Tokenizer | undefined;
let stopRequested = false;
let current = -1;
const ready = init();

/** Token IDs that end a reply: <|im_end|> and <|endoftext|>. */
const STOP = new Set([2, 0]);
const text = (b: Uint8Array) => new TextDecoder().decode(b);

async function load(base: string) {
  await ready;
  const [chatConfig, tensorCache, tokenizerJson, tokenizerConfig] =
    await Promise.all(
      [
        "mlc-chat-config.json",
        "tensor-cache.json",
        "tokenizer.json",
        "tokenizer_config.json",
      ].map(async (n) => text(await readModelFile(base, n))),
    );
  const records = (
    JSON.parse(tensorCache) as { records: { dataPath: string }[] }
  ).records;
  const shards = await Promise.all(
    records.map((r) => readModelFile(base, r.dataPath)),
  );
  engine?.free();
  engine = await Engine.load(chatConfig, tensorCache, shards);
  tokenizer = new Tokenizer(
    JSON.parse(tokenizerJson),
    JSON.parse(tokenizerConfig),
  );
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
  let out = "";
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
      .decode(output, {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
      })
      .replace(/�+$/, "");
    if (!decoded.startsWith(out))
      throw new Error("Tokenizer produced an unstable text prefix.");
    const delta = decoded.slice(out.length);
    if (delta) self.postMessage({ id, chunk: delta });
    out = decoded;
  };

  accept(await engine.forward(Uint32Array.from(tokens)));
  if (!done && !stopRequested && max > 1) {
    await engine.generateStream(max - 1, (batch: Uint32Array) => {
      for (const t of batch) accept(t);
      return !done && !stopRequested;
    });
  }
  return {
    model,
    prompt_tokens: tokens.length,
    completion_tokens: output.length,
  };
}

async function loadStage(args: {
  chatConfig: string;
  cache: string;
  shards: (ArrayBuffer | null)[];
  start: number;
  end: number;
}) {
  await ready;
  engine?.free();
  const loaded = await Engine.loadStage(
    args.chatConfig,
    args.cache,
    args.shards.map((b) => (b ? new Uint8Array(b) : new Uint8Array())),
    args.start,
    args.end,
  );
  engine = loaded;
  return { adapter: loaded.adapter(), hiddenSize: loaded.hiddenSize };
}

function stage(): Engine {
  if (!engine) throw new Error("The model is not loaded.");
  return engine;
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
    let result: unknown;
    switch (method) {
      case "init":
        result = await load(args.base);
        break;
      case "generate":
        result = await generate(id, args.model, args.request);
        break;
      case "shardsFor":
        await ready;
        result = Array.from(
          Engine.shardsFor(args.chatConfig, args.cache, args.start, args.end),
        );
        break;
      case "loadStage":
        result = await loadStage(args);
        break;
      case "runTokens":
        result = await stage().runTokens(Uint32Array.from(args.tokens));
        break;
      case "runHidden":
        result = await stage().runHidden(args.hidden);
        break;
      case "reset":
        stage().reset();
        result = true;
        break;
      case "sampling":
        stage().setSampling(args.temperature, 1, args.seed);
        result = true;
        break;
      default:
        throw new Error("Unknown worker method");
    }
    const transfer =
      result instanceof Float32Array ? [result.buffer as ArrayBuffer] : [];
    self.postMessage({ id, result }, { transfer });
  } catch (e) {
    self.postMessage({
      id,
      error: e instanceof Error ? e.message : String(e),
    });
  }
}
