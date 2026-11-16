export const POOL_NAME = "peerinfer/pool/v1";
export const SMOL_ID = "SmolLM2-135M-Instruct-q4f16_1-MLC";
export interface ModelSpec {
  id: string;
  name: string;
  description: string;
  downloadMB: number;
  runtime: "webllm" | "core";
  /** MLC q4f16_1 weights for the core runtime. */
  source?: string;
  /** Can run as a two-device team. */
  shared: boolean;
  /** Transformer layers, for models that can run as a team. */
  layers?: number;
}
/** Layer ranges of a two-device team: the host runs 0..split, the invited device split..layers. */
export function teamLayers(model: ModelSpec): { split: number; layers: number } {
  const layers = model.layers ?? 0;
  return { split: Math.floor(layers / 2), layers };
}
/** "Layers a–b" for a range start..end. */
export const layerLabel = (start: number, end: number) =>
  `Layers ${start}–${end - 1}`;
export const MODELS: ModelSpec[] = [
  {
    id: SMOL_ID,
    name: "SmolLM2 · 135M",
    description: "For short prompts and shared-compute experiments.",
    downloadMB: 76,
    runtime: "core",
    source: "https://huggingface.co/mlc-ai/SmolLM2-135M-Instruct-q4f16_1-MLC/resolve/main/",
    shared: true,
    layers: 30,
  },
  {
    id: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    name: "SmolLM2 · 360M",
    description: "A small chat model. Runs on one browser GPU or on two.",
    downloadMB: 210,
    runtime: "core",
    source: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f16_1-MLC/resolve/main/",
    shared: true,
    layers: 32,
  },
  {
    id: "Qwen3-4B-q4f16_1-MLC",
    name: "Qwen3 · 4B",
    description:
      "A larger model for longer replies. Sharing needs about 3.4 GB of GPU memory.",
    downloadMB: 2280,
    runtime: "webllm",
    shared: false,
  },
];
export const initialModel = () =>
  MODELS.find(
    (m) => m.id === new URLSearchParams(location.search).get("model"),
  ) ?? MODELS[0];
export function bootstrapIds(): string[] {
  return (
    new URLSearchParams(location.search).get("bootstrap") ??
    import.meta.env.VITE_BOOTSTRAP ??
    ""
  )
    .split(",")
    .map((s: string) => s.trim())
    .filter(Boolean);
}
/** Local weights for the core runtime, for example /mlc-fixture/<model>/resolve/main/. Development only. */
export function testWeights(): string | undefined {
  return import.meta.env.DEV
    ? (new URLSearchParams(location.search).get("weights") ?? undefined)
    : undefined;
}
/** Local test overrides are not accepted by the deployed app. */
export function testRelay(): string | undefined {
  return import.meta.env.DEV
    ? (new URLSearchParams(location.search).get("relay") ?? undefined)
    : undefined;
}
