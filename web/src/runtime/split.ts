import { TeamHostPeer, TeamMemberPeer } from "../wasm/peerinfer_web.js";
import { MODELS, testRelay, type ModelSpec } from "../config";
import { readModelFile } from "../model-cache";
import { ModelStore } from "./store";
import { generateText, prepareGeneration } from "./generation";
import { EngineWorker, downloadModel, loadTokenizer } from "./core";
import type { Runtime, RuntimeOptions } from "./types";

/**
 * A team invitation: the text `peerinfer join` also accepts, and the model,
 * so the page can show it before joining.
 */
export interface Invitation {
  text: string;
  model: string;
}
/** Layer ranges of a team as [start, end) pairs. */
export interface TeamLayout {
  ranges: Array<[number, number]>;
  /** Index of this device's range in `ranges`. */
  mine: number;
}
export interface SharedRuntime extends Runtime {
  invite?: Invitation;
  layout?: TeamLayout;
}

/** The configuration files a stage needs besides weight shards. */
const CONFIG_FILES = ["mlc-chat-config.json", "tensor-cache.json"];
const decode = (b: Uint8Array) => new TextDecoder().decode(b);

export function invitationUrl(invite: Invitation): string {
  const url = new URL(location.href);
  url.hash = `join=${invite.text}&model=${encodeURIComponent(invite.model)}`;
  return url.href;
}
export function readInvitation(hash: string): Invitation | undefined {
  if (!hash.startsWith("#join=")) return;
  if (hash.length > 9000) throw new Error("This invitation is too large.");
  const params = new URLSearchParams(hash.slice(1));
  const text = params.get("join") ?? "";
  const model = params.get("model") ?? "";
  if (
    !/^([0-9a-f]{2}){20,4000}$/.test(text) ||
    !MODELS.some((m) => m.id === model && m.shared && m.runtime === "core")
  )
    throw new Error("This invitation is not supported. Ask for a new link.");
  return { text, model };
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

const range = (start: number, end: number) => `layers ${start}–${end - 1}`;

/**
 * Hosts a team of `devices`: offers the model's files, runs the first
 * layers and passes each token's hidden states through the members.
 */
export async function createSplit(
  model: ModelSpec,
  base: string,
  devices: number,
  options: RuntimeOptions,
  onInvite: (invite: Invitation) => void,
): Promise<SharedRuntime> {
  const engine = new EngineWorker();
  let peer: TeamHostPeer | undefined;
  let disposed = false;
  let active = false;
  const abort = () => {
    engine.terminate();
    void peer?.shutdown();
  };
  options.signal.addEventListener("abort", abort, { once: true });
  try {
    await downloadModel(base, options);
    const tokenizer = await loadTokenizer(base, options.signal);
    const read = (name: string) => readModelFile(base, name, options.signal);
    const chatConfig = decode(await read("mlc-chat-config.json"));
    const cache = decode(await read("tensor-cache.json"));
    const records = (JSON.parse(cache) as { records: { dataPath: string }[] })
      .records;
    const layers = JSON.parse(chatConfig).model_config
      .num_hidden_layers as number;

    options.progress({
      phase: "preparing",
      message: "Preparing the model files to share",
    });
    const names = [...CONFIG_FILES, ...records.map((r) => r.dataPath)];
    const files = await Promise.all(names.map(read));
    const p = await TeamHostPeer.start(
      model.id,
      layers,
      devices,
      names,
      files,
      testRelay(),
    );
    peer = p;
    await p.online();
    options.signal.throwIfAborted();
    const [start, end] = p.hostLayers();
    const slots = () =>
      p.slots() as Array<{
        start: number;
        end: number;
        member: string | null;
        ready: boolean;
      }>;
    const layout: TeamLayout = {
      ranges: [
        [start, end],
        ...slots().map((s) => [s.start, s.end] as [number, number]),
      ],
      mine: 0,
    };

    options.progress({ phase: "preparing", message: "Preparing your GPU" });
    const need = await stageShards(engine, chatConfig, cache, start, end);
    await loadStage(
      engine,
      chatConfig,
      cache,
      files.slice(CONFIG_FILES.length).map((b, i) => (need.has(i) ? b : null)),
      start,
      end,
    );
    options.signal.throwIfAborted();

    const invite: Invitation = { text: p.invitation(), model: model.id };
    onInvite(invite);
    const waiting = () => {
      const s = slots();
      const ready = s.filter((x) => x.ready).length;
      options.progress({
        phase: "pairing",
        message: `Waiting for the other devices: ${ready} of ${s.length} ready`,
      });
    };
    waiting();
    while (!p.isComplete()) {
      await Promise.race([
        p.changed(),
        new Promise<never>((_, reject) =>
          options.signal.addEventListener(
            "abort",
            () => reject(options.signal.reason),
            { once: true },
          ),
        ),
      ]);
      options.signal.throwIfAborted();
      waiting();
    }
    // A member that leaves ends the team: its layers are missing.
    void (async () => {
      while (!disposed) {
        await p.changed();
        if (!disposed && !p.isComplete()) {
          options.unavailable(
            "A device left the team. Stop this team and invite the devices again.",
          );
          return;
        }
      }
    })();
    options.signal.removeEventListener("abort", abort);
    return {
      invite,
      layout,
      async generate(request, signal, chunk) {
        if (active) throw new Error("This team is already generating.");
        if (disposed || !p.isComplete())
          throw new Error("A device in this team is unavailable.");
        const { temperature } = prepareGeneration(tokenizer, request);
        active = true;
        const seed = crypto.getRandomValues(new Uint32Array(1))[0];
        let run: Awaited<ReturnType<TeamHostPeer["begin"]>> | undefined;
        try {
          await engine.call("reset");
          run = await p.begin(temperature, seed);
          const r = run;
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
              return await r.step(hidden);
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
          await run?.end();
          await engine.call("reset").catch(() => {});
          active = false;
        }
      },
      async dispose() {
        disposed = true;
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

/** Joins a team and runs the layer range the host assigns. Not advertised as a worker. */
export async function joinSplit(
  invite: Invitation,
  options: RuntimeOptions,
): Promise<SharedRuntime> {
  const store = await ModelStore.open(),
    engine = new EngineWorker();
  let peer: TeamMemberPeer | undefined;
  let disposed = false;
  const abort = () => {
    engine.terminate();
    void peer?.leave();
  };
  options.signal.addEventListener("abort", abort, { once: true });
  try {
    options.progress({ phase: "downloading", message: "Joining the team" });
    const p = await TeamMemberPeer.join(invite.text, testRelay());
    peer = p;
    const a = p.assignment() as {
      start: number;
      end: number;
      model: string;
      layers: number;
      files: Array<{ name: string; hash: string; bytes: number }>;
    };
    if (a.model !== invite.model)
      throw new Error("This invitation offers a different model.");
    const file = (name: string) => {
      const f = a.files.find((x) => x.name === name);
      if (!f) throw new Error(`The host offers no ${name}.`);
      return f;
    };
    // Files are saved by hash and checked again when reused.
    let received = 0,
      reused = 0;
    const fetchFile = async (name: string, report: () => void) => {
      const f = file(name);
      const saved = await store.read(f.hash);
      if (saved?.length === f.bytes && TeamMemberPeer.verify(saved, f.hash)) {
        reused += f.bytes;
        report();
        return saved;
      }
      const bytes = await p.fetch(name);
      options.signal.throwIfAborted();
      await store.write(f.hash, bytes);
      received += f.bytes;
      report();
      return bytes;
    };
    const quiet = () => {};
    const chatConfig = decode(await fetchFile("mlc-chat-config.json", quiet));
    const cache = decode(await fetchFile("tensor-cache.json", quiet));
    const records = (JSON.parse(cache) as { records: { dataPath: string }[] })
      .records;
    const need = await stageShards(engine, chatConfig, cache, a.start, a.end);
    const total = records
      .filter((_, i) => need.has(i))
      .reduce((n, r) => n + file(r.dataPath).bytes, 0);
    received = reused = 0;
    const report = () =>
      options.progress({
        phase: "downloading",
        message: "Receiving model files from the host",
        loaded: received + reused,
        total,
      });
    report();
    const shards: (Uint8Array | null)[] = [];
    for (const [i, r] of records.entries())
      shards.push(need.has(i) ? await fetchFile(r.dataPath, report) : null);
    options.progress({ phase: "preparing", message: "Preparing your GPU" });
    await loadStage(engine, chatConfig, cache, shards, a.start, a.end);
    options.signal.throwIfAborted();
    await p.ready();
    const mine = range(a.start, a.end);
    void (async () => {
      for (;;) {
        const command = await p.next();
        if (!command) {
          if (!disposed)
            options.unavailable(
              "The host closed the team. Open a new invitation to join again.",
            );
          return;
        }
        try {
          switch (command.kind()) {
            case "begin":
              await engine.call("reset");
              await engine.call("sampling", {
                temperature: command.temperature(),
                seed: command.seed(),
              });
              options.progress({
                phase: "busy",
                message: `Running ${mine} of a reply`,
              });
              command.replyOk();
              break;
            case "step": {
              const hidden = command.hidden();
              const out = await engine.call<number | Float32Array>(
                "runHidden",
                { hidden },
                { transfer: [hidden.buffer as ArrayBuffer] },
              );
              if (typeof out === "number") command.replyToken(out);
              else command.replyHidden(out);
              break;
            }
            case "end":
              await engine.call("reset");
              options.progress({
                phase: "ready",
                message: `In the team · ${mine}`,
              });
              command.replyOk();
              break;
          }
        } catch (error) {
          command.replyError(
            error instanceof Error ? error.message : String(error),
          );
        }
      }
    })();
    options.signal.removeEventListener("abort", abort);
    options.progress({
      phase: "ready",
      message: `In the team · ${mine} · ${(received / 1e6).toFixed(1)} MB received, ${(reused / 1e6).toFixed(1)} MB reused`,
    });
    return {
      layout: { ranges: [[a.start, a.end]], mine: 0 },
      async generate() {
        throw new Error("This tab runs part of a team. Chat through the host.");
      },
      async dispose() {
        disposed = true;
        engine.terminate();
        await p.leave();
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
