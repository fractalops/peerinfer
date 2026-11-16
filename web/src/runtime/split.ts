import type { Tokenizer } from "@huggingface/tokenizers";
import { StagePeer } from "../wasm/peerinfer_web.js";
import { MODELS, testRelay, type ModelSpec } from "../config";
import { readModelFile } from "../model-cache";
import { ModelStore } from "./store";
import { generateText, prepareGeneration } from "./generation";
import { pack, unpack } from "./packet.js";
import { StageState } from "./stage-state";
import { EngineWorker, downloadModel, loadTokenizer } from "./core";
import type { Chunk, PeerAddress, Runtime, RuntimeOptions } from "./types";

/** Model files travel as hash-verified chunks of at most this size. */
const CHUNK_BYTES = 1 << 20;
/** The configuration files a stage needs besides weight shards. */
const CONFIG_FILES = ["mlc-chat-config.json", "tensor-cache.json"];

export interface Invitation {
  version: 2;
  /** The model the team runs; the manifest must name the same model. */
  model: string;
  address: PeerAddress;
  manifest: Chunk;
  secret: string;
}
export interface SharedRuntime extends Runtime {
  invite?: Invitation;
}

/** What the host offers: the model, the split point and its files. */
interface Manifest {
  version: 2;
  model: string;
  layers: number;
  split: number;
  files: Array<{ name: string; bytes: number; chunks: Chunk[] }>;
}

const encode = (v: unknown) => new TextEncoder().encode(JSON.stringify(v));
const decode = (b: Uint8Array) => new TextDecoder().decode(b);

export function invitationUrl(invite: Invitation): string {
  const url = new URL(location.href);
  url.hash = `join=${btoa(JSON.stringify(invite))}`;
  return url.href;
}
export function readInvitation(hash: string): Invitation | undefined {
  if (!hash.startsWith("#join=")) return;
  if (hash.length > 6000) throw new Error("This invitation is too large.");
  const invite = JSON.parse(atob(hash.slice(6)));
  if (
    invite.version !== 2 ||
    !MODELS.some(
      (m) => m.id === invite.model && m.shared && m.runtime === "core",
    ) ||
    !/^[0-9a-f]{64}$/.test(invite.address?.id) ||
    !Array.isArray(invite.address?.addrs) ||
    invite.address.addrs.length > 8 ||
    !/^[0-9a-f]{64}$/.test(invite.manifest?.hash) ||
    !Number.isInteger(invite.manifest.bytes) ||
    invite.manifest.bytes < 1 ||
    invite.manifest.bytes > 1_048_576 ||
    !/^[0-9a-f-]{36}$/.test(invite.secret)
  )
    throw new Error("This invitation is not supported. Ask for a new link.");
  return invite;
}

/** Rejects manifests for unknown models, odd file names or odd sizes. */
export function validateManifest(m: Manifest): ModelSpec {
  const model = MODELS.find((x) => x.id === m?.model);
  if (
    m?.version !== 2 ||
    !model ||
    !model.shared ||
    model.runtime !== "core" ||
    !Number.isInteger(m.layers) ||
    !Number.isInteger(m.split) ||
    m.split < 1 ||
    m.split >= m.layers ||
    m.layers > 256 ||
    !Array.isArray(m.files) ||
    m.files.length > 64 ||
    !CONFIG_FILES.every((n) => m.files.some((f) => f.name === n))
  )
    throw new Error("This invitation offers an unsupported model.");
  let total = 0;
  for (const f of m.files) {
    const sum = Array.isArray(f.chunks)
      ? f.chunks.reduce((n, c) => n + c.bytes, 0)
      : -1;
    if (
      typeof f.name !== "string" ||
      !/^[\w.-]{1,64}$/.test(f.name) ||
      !Number.isSafeInteger(f.bytes) ||
      sum !== f.bytes ||
      f.chunks.some(
        (c) =>
          !/^[0-9a-f]{64}$/.test(c.hash) ||
          !Number.isInteger(c.bytes) ||
          c.bytes < 1 ||
          c.bytes > CHUNK_BYTES,
      )
    )
      throw new Error("This invitation has an invalid model manifest.");
    total += f.bytes;
  }
  if (total > 8e9) throw new Error("This model is too large to share.");
  return model;
}

async function rpc(
  peer: StagePeer,
  address: PeerAddress,
  message: unknown,
): Promise<any> {
  // Enrollment and liveness messages are idempotent. Never replay a GPU step.
  const safeToRetry = ["join", "heartbeat", "ready"].includes(
    (message as any).type,
  );
  for (let attempt = 0; ; attempt++) {
    try {
      const result = unpack(await peer.request(address, pack(message)));
      if (result.error) throw new Error(result.error);
      return result;
    } catch (error) {
      if (!safeToRetry || attempt === 1) throw error;
    }
  }
}

async function listen(
  peer: StagePeer,
  signal: AbortSignal,
  handler: (message: any, remote: string) => Promise<unknown>,
) {
  while (!signal.aborted) {
    const job = await peer.nextJob();
    if (!job) return;
    try {
      const remote = await job.peer();
      const message = unpack(await job.payload());
      const result = await handler(message, remote);
      await job.respond(pack(result));
    } catch (error) {
      await job
        .respond(
          pack({
            error: error instanceof Error ? error.message : String(error),
          }),
        )
        .catch(() => {});
    }
  }
}
async function peerFor(options: RuntimeOptions) {
  const peer = await StagePeer.spawn(testRelay());
  try {
    options.signal.throwIfAborted();
    await peer.online();
    return peer;
  } catch (error) {
    await peer.shutdown();
    throw error;
  }
}

/** Shard files (by index in tensor-cache.json) that layers start..end need. */
async function stageShards(
  engine: EngineWorker,
  chatConfig: string,
  cache: string,
  start: number,
  end: number,
): Promise<Set<number>> {
  return new Set(
    await engine.call<number[]>("shardsFor", { chatConfig, cache, start, end }),
  );
}

/** Loads layers start..end in the worker from file bytes. */
async function loadStage(
  engine: EngineWorker,
  chatConfig: string,
  cache: string,
  shards: (Uint8Array | null)[],
  start: number,
  end: number,
) {
  // Copy into owned buffers and hand them to the worker without another copy.
  const buffers = shards.map((s) => (s ? s.slice().buffer : null));
  await engine.call(
    "loadStage",
    { chatConfig, cache, shards: buffers, start, end },
    { transfer: buffers.filter((b): b is ArrayBuffer => b !== null) },
  );
}

/**
 * Hosts a two-device team: shares the model's files over iroh-blobs, runs
 * the first half of the layers and asks the invited device for each token.
 */
export async function createSplit(
  model: ModelSpec,
  base: string,
  options: RuntimeOptions,
  onInvite: (invite: Invitation) => void,
): Promise<SharedRuntime> {
  const engine = new EngineWorker();
  let peer: StagePeer | undefined;
  let helper: PeerAddress | undefined;
  let alive = Date.now();
  let disposed = false;
  let active = false;
  const closed = new AbortController();
  const abort = () => {
    closed.abort();
    engine.terminate();
    void peer?.shutdown();
  };
  options.signal.addEventListener("abort", abort, { once: true });
  let heartbeat: ReturnType<typeof setInterval> | undefined;
  try {
    peer = await peerFor(options);
    const p = peer;
    await downloadModel(base, options);
    const tokenizer: Tokenizer = await loadTokenizer(base, options.signal);
    const read = (name: string) => readModelFile(base, name, options.signal);
    const chatConfig = decode(await read("mlc-chat-config.json"));
    const cache = decode(await read("tensor-cache.json"));
    const records = (JSON.parse(cache) as { records: { dataPath: string }[] })
      .records;
    const layers = JSON.parse(chatConfig).model_config
      .num_hidden_layers as number;
    const split = Math.floor(layers / 2);

    options.progress({
      phase: "preparing",
      message: "Preparing verified weights to share",
    });
    const files: Manifest["files"] = [];
    const shardBytes: Uint8Array[] = [];
    for (const name of [...CONFIG_FILES, ...records.map((r) => r.dataPath)]) {
      options.signal.throwIfAborted();
      const bytes = await read(name);
      if (!CONFIG_FILES.includes(name)) shardBytes.push(bytes);
      const chunks: Chunk[] = [];
      for (let at = 0; at < bytes.length; at += CHUNK_BYTES) {
        const part = bytes.subarray(at, at + CHUNK_BYTES);
        chunks.push({ hash: await p.addBlob(part), bytes: part.length });
      }
      files.push({ name, bytes: bytes.length, chunks });
    }
    const manifest: Manifest = {
      version: 2,
      model: model.id,
      layers,
      split,
      files,
    };
    const manifestBytes = encode(manifest);
    const hash = await p.addBlob(manifestBytes);

    options.progress({ phase: "preparing", message: "Preparing your GPU" });
    const need = await stageShards(engine, chatConfig, cache, 0, split);
    await loadStage(
      engine,
      chatConfig,
      cache,
      shardBytes.map((b, i) => (need.has(i) ? b : null)),
      0,
      split,
    );
    options.signal.throwIfAborted();

    const invite: Invitation = {
      version: 2,
      model: model.id,
      address: p.address(),
      manifest: { hash, bytes: manifestBytes.length },
      secret: crypto.randomUUID(),
    };
    let ready!: () => void;
    const waiting = new Promise<void>((resolve) => (ready = resolve));
    void listen(p, closed.signal, async (message, remote) => {
      if (message.secret !== invite.secret)
        throw new Error("Invalid invitation.");
      if (message.type === "join") {
        if (message.address?.id !== remote || (helper && helper.id !== remote))
          throw new Error("This team already has a second device.");
        helper = message.address;
        alive = Date.now();
        return { ok: true };
      }
      if (!helper || helper.id !== remote)
        throw new Error("Join this team first.");
      if (message.type === "ready") {
        alive = Date.now();
        ready();
        return { ok: true };
      }
      if (message.type === "heartbeat") {
        alive = Date.now();
        return { ok: true };
      }
      if (message.type === "leave") {
        options.unavailable(
          "The other device left. Stop this team and invite it again.",
        );
        helper = undefined;
        return { ok: true };
      }
      throw new Error("Unsupported team message.");
    }).catch((error) => {
      if (!disposed) options.unavailable(String(error));
    });
    onInvite(invite);
    options.progress({
      phase: "pairing",
      message: "Waiting for the other device to load its half",
    });
    await Promise.race([
      waiting,
      new Promise<never>((_, reject) =>
        options.signal.addEventListener(
          "abort",
          () => reject(options.signal.reason),
          { once: true },
        ),
      ),
    ]);
    options.signal.throwIfAborted();
    heartbeat = setInterval(() => {
      if (!disposed && Date.now() - alive > 120_000)
        options.unavailable(
          "The other device is offline. Keep both tabs visible, then start a new team.",
        );
    }, 5000);
    options.signal.removeEventListener("abort", abort);
    return {
      invite,
      async generate(request, signal, chunk) {
        if (active) throw new Error("This team is already generating.");
        if (!helper || disposed || Date.now() - alive > 120_000)
          throw new Error("The other device is unavailable.");
        const { temperature } = prepareGeneration(tokenizer, request);
        active = true;
        const address = helper,
          id = crypto.randomUUID(),
          seed = crypto.getRandomValues(new Uint32Array(1))[0];
        let step = 0;
        try {
          await engine.call("reset");
          await rpc(p, address, { type: "begin", id, temperature, seed });
          return await generateText(
            tokenizer,
            model.id,
            request,
            signal,
            async (tokens) => {
              const hidden = await engine.call<Float32Array>("runTokens", {
                tokens,
              });
              signal.throwIfAborted();
              const second = await rpc(p, address, {
                type: "step",
                id,
                step: step++,
                hidden,
              });
              return second.token;
            },
            chunk,
          );
        } catch (error) {
          if (!signal.aborted)
            options.unavailable(
              `Team interrupted: ${error instanceof Error ? error.message : String(error)}`,
            );
          throw error;
        } finally {
          await engine.call("reset").catch(() => {});
          await rpc(p, address, { type: "end", id }).catch(() => {});
          active = false;
        }
      },
      async dispose() {
        disposed = true;
        closed.abort();
        clearInterval(heartbeat);
        engine.terminate();
        await p.shutdown();
      },
    };
  } catch (error) {
    disposed = true;
    abort();
    throw error;
  } finally {
    options.signal.removeEventListener("abort", abort);
  }
}

/** The invited device runs only the second half of the layers; it is not advertised as a complete model. */
export async function joinSplit(
  invite: Invitation,
  options: RuntimeOptions,
): Promise<SharedRuntime> {
  const store = await ModelStore.open(),
    engine = new EngineWorker();
  let peer: StagePeer | undefined;
  let heartbeat: ReturnType<typeof setInterval> | undefined;
  let disposed = false;
  let lastCommand = Date.now();
  const closed = new AbortController();
  const state = new StageState(invite.address.id);
  const abort = () => {
    closed.abort();
    engine.terminate();
    void peer?.shutdown();
  };
  options.signal.addEventListener("abort", abort, { once: true });
  try {
    peer = await peerFor(options);
    const p = peer;
    await rpc(p, invite.address, {
      type: "join",
      secret: invite.secret,
      address: p.address(),
    });
    let beating = false,
      lastHeartbeat = Date.now();
    heartbeat = setInterval(async () => {
      if (beating || disposed) return;
      beating = true;
      try {
        await rpc(p, invite.address, {
          type: "heartbeat",
          secret: invite.secret,
        });
        lastHeartbeat = Date.now();
      } catch {
        if (!disposed && Date.now() - lastHeartbeat > 120_000)
          options.unavailable(
            "The host is unreachable. Keep both tabs open, then rejoin the team.",
          );
      } finally {
        beating = false;
      }
    }, 5000);
    // A received chunk also shows that the host is reachable: heartbeats
    // can time out while weight transfers fill a slow relay connection.
    const receiving: RuntimeOptions = {
      ...options,
      progress: (value) => {
        lastHeartbeat = Date.now();
        options.progress(value);
      },
    };
    options.progress({
      phase: "downloading",
      message: "Receiving the model manifest",
    });
    const manifest = JSON.parse(
      decode(
        await store.blob(invite.manifest, p, invite.address, options.signal),
      ),
    ) as Manifest;
    validateManifest(manifest);
    if (manifest.model !== invite.model)
      throw new Error("This invitation offers a different model.");
    // Receives a file's chunks (verified and saved by the store), then joins them.
    const fetchFile = async (name: string, receive = true) => {
      const file = manifest.files.find((f) => f.name === name)!;
      if (receive)
        await store.prefetch(file.chunks, p, invite.address, receiving);
      const bytes = new Uint8Array(file.bytes);
      let at = 0;
      for (const c of file.chunks) {
        const part = await store.read(c.hash);
        if (part?.length !== c.bytes)
          throw new Error("Received weights are missing. Rejoin the team.");
        bytes.set(part, at);
        at += c.bytes;
      }
      return bytes;
    };
    const chatConfig = decode(await fetchFile("mlc-chat-config.json"));
    const cache = decode(await fetchFile("tensor-cache.json"));
    const records = (JSON.parse(cache) as { records: { dataPath: string }[] })
      .records;
    const { split, layers } = manifest;
    const need = await stageShards(engine, chatConfig, cache, split, layers);
    // Receive all needed chunks first so progress covers the whole stage.
    const needed = records
      .filter((_, i) => need.has(i))
      .map((r) => manifest.files.find((f) => f.name === r.dataPath));
    if (needed.some((f) => !f))
      throw new Error("This invitation has an invalid model manifest.");
    await store.prefetch(
      needed.flatMap((f) => f!.chunks),
      p,
      invite.address,
      receiving,
    );
    const shards: (Uint8Array | null)[] = [];
    for (const [i, r] of records.entries())
      shards.push(need.has(i) ? await fetchFile(r.dataPath, false) : null);
    options.progress({ phase: "preparing", message: "Preparing your GPU" });
    await loadStage(engine, chatConfig, cache, shards, split, layers);
    options.signal.throwIfAborted();
    const range = `layers ${split}–${layers - 1}`;
    void listen(p, closed.signal, async (message, remote) => {
      state.authorize(remote);
      lastCommand = Date.now();
      if (message.type === "begin") {
        if (
          !Number.isFinite(message.temperature) ||
          message.temperature < 0 ||
          message.temperature > 2 ||
          !Number.isInteger(message.seed) ||
          message.seed < 0 ||
          message.seed > 0xffffffff
        )
          throw new Error("Invalid stage input.");
        state.begin(remote, message.id);
        await engine.call("reset");
        await engine.call("sampling", {
          temperature: message.temperature,
          seed: message.seed,
        });
        options.progress({
          phase: "busy",
          message: "Running the second half of a reply",
        });
        return { ok: true };
      }
      if (message.type === "end") {
        state.end(remote, message.id);
        await engine.call("reset");
        options.progress({ phase: "ready", message: `Paired · ${range} ready` });
        return { ok: true };
      }
      if (message.type === "step") {
        state.next(remote, message.id, message.step);
        if (!(message.hidden instanceof Float32Array))
          throw new Error("Invalid stage input.");
        const token = await engine.call<number>(
          "runHidden",
          { hidden: message.hidden },
          { transfer: [message.hidden.buffer] },
        );
        return { token };
      }
      throw new Error("Unsupported stage command.");
    }).catch((error) => {
      if (!disposed) options.unavailable(String(error));
    });
    await rpc(p, invite.address, { type: "ready", secret: invite.secret });
    options.signal.removeEventListener("abort", abort);
    options.progress({
      phase: "ready",
      message: `Paired · ${range} · ${(store.downloaded / 1e6).toFixed(1)} MB received, ${(store.reused / 1e6).toFixed(1)} MB reused`,
    });
    // Bound abandoned KV state. Every generation still explicitly resets before use.
    const idle = setInterval(() => {
      if (Date.now() - lastCommand > 90_000) {
        state.clear();
        void engine.call("reset").catch(() => {});
        lastCommand = Date.now();
      }
    }, 15_000);
    return {
      async generate() {
        throw new Error("This tab is a model stage. Chat through the host.");
      },
      async dispose() {
        disposed = true;
        closed.abort();
        clearInterval(heartbeat);
        clearInterval(idle);
        engine.terminate();
        await rpc(p, invite.address, {
          type: "leave",
          secret: invite.secret,
        }).catch(() => {});
        await p.shutdown();
        store.close();
      },
    };
  } catch (error) {
    disposed = true;
    abort();
    store.close();
    throw error;
  } finally {
    options.signal.removeEventListener("abort", abort);
  }
}
