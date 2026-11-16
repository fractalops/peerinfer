import { Tokenizer } from "@huggingface/tokenizers";
import { withAbort } from "./abort";
import { cacheModelWeights, readModelFile } from "../model-cache";
import type { Completion, Runtime, RuntimeOptions } from "./types";

type Reply =
  | { id: number; chunk: string }
  | { id: number; result: unknown }
  | { id: number; error: string };

/** The peerinfer-core engine in a Web Worker, with request/response calls. */
export class EngineWorker {
  private worker = new Worker(new URL("./core-worker.ts", import.meta.url), {
    type: "module",
  });
  private pending = new Map<
    number,
    {
      resolve: (v: unknown) => void;
      reject: (e: Error) => void;
      chunk?: (text: string) => Promise<void>;
      // Chunks are passed on in order; the result waits for them.
      queue: Promise<void>;
    }
  >();
  private next = 0;

  constructor() {
    this.worker.onmessage = ({ data }: MessageEvent<Reply>) => {
      const p = this.pending.get(data.id);
      if (!p) return;
      if ("chunk" in data) {
        const send = p.chunk;
        if (send) p.queue = p.queue.then(() => send(data.chunk));
        return;
      }
      this.pending.delete(data.id);
      if ("error" in data) p.reject(new Error(data.error));
      else void p.queue.then(() => p.resolve(data.result), p.reject);
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values())
        p.reject(new Error(e.message || "The model worker failed."));
      this.pending.clear();
    };
  }

  /** A request ID for `call` and `stop`. */
  id(): number {
    return this.next++;
  }

  call<T>(
    method: string,
    args: unknown = {},
    options: {
      id?: number;
      chunk?: (text: string) => Promise<void>;
      transfer?: Transferable[];
    } = {},
  ): Promise<T> {
    const id = options.id ?? this.id();
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        chunk: options.chunk,
        queue: Promise.resolve(),
      });
      this.worker.postMessage(
        { id, method, args },
        { transfer: options.transfer ?? [] },
      );
    });
  }

  /** Ends request `id` after its current batch of tokens. */
  stop(id: number) {
    this.worker.postMessage({ id, method: "stop" });
  }

  terminate() {
    this.worker.terminate();
    for (const p of this.pending.values())
      p.reject(new Error("The model worker stopped."));
    this.pending.clear();
  }
}

/** The tokenizer of an MLC model, from the weight cache or the network. */
export async function loadTokenizer(
  base: string,
  signal?: AbortSignal,
): Promise<Tokenizer> {
  const json = async (name: string) =>
    JSON.parse(
      new TextDecoder().decode(await readModelFile(base, name, signal)),
    );
  return new Tokenizer(
    await json("tokenizer.json"),
    await json("tokenizer_config.json"),
  );
}

/** Downloads an MLC model into the weight cache with progress. */
export async function downloadModel(base: string, options: RuntimeOptions) {
  await cacheModelWeights(
    base,
    ({ loaded, total }) =>
      options.progress({
        phase: "downloading",
        message: "Downloading model weights",
        loaded,
        total,
      }),
    options.signal,
  );
  options.signal.throwIfAborted();
}

/**
 * Runs an MLC q4f16_1 model with the peerinfer-core engine in a Web Worker.
 * Weights share the cache that the WebLLM runtime uses.
 */
export async function createCore(
  modelId: string,
  base: string,
  options: RuntimeOptions,
): Promise<Runtime> {
  await downloadModel(base, options);
  const engine = new EngineWorker();
  const closed = new AbortController();
  const abort = () => {
    closed.abort();
    engine.terminate();
  };
  options.signal.addEventListener("abort", abort, { once: true });
  try {
    options.progress({ phase: "preparing", message: "Preparing your GPU" });
    await withAbort(engine.call("init", { base }), options.signal);
    options.signal.throwIfAborted();
    let active = false;
    return {
      async generate(request, signal, chunk) {
        if (active) throw new Error("This GPU is already generating.");
        active = true;
        signal = AbortSignal.any([signal, closed.signal]);
        const id = engine.id();
        const stop = () => engine.stop(id);
        signal.addEventListener("abort", stop, { once: true });
        try {
          signal.throwIfAborted();
          return await withAbort(
            engine.call<Completion>(
              "generate",
              { model: modelId, request },
              { id, chunk },
            ),
            signal,
          );
        } finally {
          signal.removeEventListener("abort", stop);
          active = false;
        }
      },
      async dispose() {
        closed.abort();
        engine.terminate();
      },
    };
  } catch (error) {
    engine.terminate();
    throw error;
  } finally {
    options.signal.removeEventListener("abort", abort);
  }
}
