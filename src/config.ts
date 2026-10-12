import { z } from "zod";

export const Provider = z.enum(["ollama", "anthropic", "openai"]);
export type Provider = z.infer<typeof Provider>;

/** Defaults chosen for structured DevOps reasoning; override with LLM_MODEL. */
export const DEFAULT_MODELS: Readonly<Record<Provider, string>> = {
  ollama: "qwen3.5:9b",
  anthropic: "claude-sonnet-5-5",
  openai: "gpt-5-mini",
};

export const DEFAULT_OLLAMA_HOST = "http://localhost:11434";

export interface LlmConfig {
  provider: Provider;
  model: string;
  /** Ollama server URL, no trailing slash. Ignored by hosted providers. */
  ollamaHost: string;
  /** Ollama context window in tokens. Ollama's default (2-4k) is far too small for real inputs. */
  numCtx: number;
  /** Cap on generated tokens; stops runaway generations. */
  maxOutputTokens: number;
  /** Per-request timeout. */
  timeoutMs: number;
}

/** Values from CLI flags; they win over environment variables. */
export type LlmConfigInput = Partial<{ [K in keyof LlmConfig]: string | number }>;

type Env = Record<string, string | undefined>;

/**
 * Environment variables shared by every llm-kit tool. Tool-specific settings should use their own prefix.
 * `OLLAMA_HOST` is Ollama's own variable, so a machine already set up for Ollama needs no extra config.
 */
export const LLM_ENV = {
  provider: "LLM_PROVIDER",
  model: "LLM_MODEL",
  ollamaHost: "OLLAMA_HOST",
  numCtx: "LLM_NUM_CTX",
  maxOutputTokens: "LLM_MAX_OUTPUT_TOKENS",
  timeoutMs: "LLM_TIMEOUT_MS",
} as const satisfies Record<keyof LlmConfig, string>;

const positiveInt = (min = 1) => z.coerce.number().int().min(min);

const Resolved = z.object({
  provider: Provider,
  model: z.string().min(1),
  ollamaHost: z.url(),
  numCtx: positiveInt(2048),
  maxOutputTokens: positiveInt(64),
  timeoutMs: positiveInt(1000),
});

/**
 * Resolve the LLM settings. Precedence: `input` (CLI flags) > `env` > defaults.
 * Empty strings count as unset, so `LLM_MODEL=` in a .env file falls back to the default.
 * Without an explicit provider, a present ANTHROPIC_API_KEY or OPENAI_API_KEY picks that provider, else Ollama.
 */
export function resolveLlmConfig(input: LlmConfigInput = {}, env: Env = process.env): LlmConfig {
  const pick = (key: keyof LlmConfig) => nonEmpty(input[key]) ?? nonEmpty(env[LLM_ENV[key]]);
  const provider = (pick("provider") ?? detectProvider(env)).toString().toLowerCase();

  const raw = {
    provider,
    model: pick("model") ?? DEFAULT_MODELS[provider as Provider],
    ollamaHost: normalizeHost(pick("ollamaHost")?.toString() ?? DEFAULT_OLLAMA_HOST),
    numCtx: pick("numCtx") ?? 16_384,
    maxOutputTokens: pick("maxOutputTokens") ?? 4096,
    timeoutMs: pick("timeoutMs") ?? 180_000,
  };
  const parsed = Resolved.safeParse(raw);
  if (!parsed.success) throw new Error(`Invalid LLM configuration:\n${z.prettifyError(parsed.error)}`);
  return parsed.data;
}

/** Pick a hosted provider when its API key is present, otherwise fall back to local Ollama. */
export function detectProvider(env: Env = process.env): Provider {
  if (nonEmpty(env.ANTHROPIC_API_KEY)) return "anthropic";
  if (nonEmpty(env.OPENAI_API_KEY)) return "openai";
  return "ollama";
}

/** Ollama's OLLAMA_HOST is often "host:port" or "0.0.0.0:11434"; HTTP clients need a full URL. */
export function normalizeHost(host: string): string {
  let url = /^https?:\/\//.test(host) ? host : `http://${host}`;
  // 0.0.0.0 is a listen address, not one you can connect to.
  url = url.replace("//0.0.0.0", "//127.0.0.1");
  return url.replace(/\/+$/, "");
}

function nonEmpty<T>(v: T | undefined): T | undefined {
  return typeof v === "string" && v.trim() === "" ? undefined : v;
}
