import { withAbort } from "./abort";
import { cacheModelWeights } from "../model-cache";
import type { Completion, Runtime, RuntimeOptions } from "./types";

type Reply =
  | { id: number; chunk: string }
  | { id: number; result: unknown }
  | { id: number; error: string };

/**
 * Runs an MLC q4f16_1 model with the peerinfer-core engine in a Web Worker.
 * Weights share the cache that the WebLLM runtime uses.
 */
export async function createCore(
  modelId: string,
  source: string,
  options: RuntimeOptions,
): Promise<Runtime> {
  const base = source.endsWith("/") ? source : `${source}/`;
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

  const worker = new Worker(new URL("./core-worker.ts", import.meta.url), {
    type: "module",
  });
  const closed = new AbortController();
  const pending = new Map<
    number,
    {
      resolve: (v: unknown) => void;
      reject: (e: Error) => void;
      chunk?: (text: string) => Promise<void>;
      // Chunks are passed on in order; the result waits for them.
      queue: Promise<void>;
    }
  >();
  let next = 0;
  worker.onmessage = ({ data }: MessageEvent<Reply>) => {
    const p = pending.get(data.id);
    if (!p) return;
    if ("chunk" in data) {
      const send = p.chunk;
      if (send) p.queue = p.queue.then(() => send(data.chunk));
      return;
    }
    pending.delete(data.id);
    if ("error" in data) p.reject(new Error(data.error));
    else void p.queue.then(() => p.resolve(data.result), p.reject);
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message || "The model worker failed."));
    pending.clear();
  };
  const call = <T>(
    method: string,
    args: unknown,
    chunk?: (text: string) => Promise<void>,
    id = next++,
  ): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        chunk,
        queue: Promise.resolve(),
      });
      worker.postMessage({ id, method, args });
    });

  const abort = () => {
    closed.abort();
    worker.terminate();
  };
  options.signal.addEventListener("abort", abort, { once: true });
  try {
    options.progress({ phase: "preparing", message: "Preparing your GPU" });
    await withAbort(call("init", { base }), options.signal);
    options.signal.throwIfAborted();
    let active = false;
    return {
      async generate(request, signal, chunk) {
        if (active) throw new Error("This GPU is already generating.");
        active = true;
        signal = AbortSignal.any([signal, closed.signal]);
        const id = next++;
        const stop = () => worker.postMessage({ id, method: "stop" });
        signal.addEventListener("abort", stop, { once: true });
        try {
          signal.throwIfAborted();
          return await withAbort(
            call<Completion>("generate", { model: modelId, request }, chunk, id),
            signal,
          );
        } finally {
          signal.removeEventListener("abort", stop);
          active = false;
        }
      },
      async dispose() {
        closed.abort();
        worker.terminate();
      },
    };
  } catch (error) {
    worker.terminate();
    throw error;
  } finally {
    options.signal.removeEventListener("abort", abort);
  }
}
