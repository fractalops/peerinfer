import type { CompletionRequest } from "../types";
export type Phase =
  | "downloading"
  | "preparing"
  | "pairing"
  | "ready"
  | "busy"
  | "paused"
  | "error"
  | "stopped";
export interface RuntimeProgress {
  phase: Phase;
  message: string;
  loaded?: number;
  total?: number;
}
export interface Completion {
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
}
export interface Runtime {
  generate(
    request: CompletionRequest,
    signal: AbortSignal,
    chunk: (text: string) => Promise<void>,
  ): Promise<Completion>;
  dispose(): Promise<void>;
}
export interface RuntimeOptions {
  signal: AbortSignal;
  progress: (progress: RuntimeProgress) => void;
  unavailable: (reason: string) => void;
}
