import { z } from "zod";
import { type CallOptions, withOneRetry } from "./client.js";
import { DEFAULT_OLLAMA_HOST, normalizeHost } from "./config.js";
import { importOptional } from "./provider.js";

export const EmbedProvider = z.enum(["ollama", "openai"]);
export type EmbedProvider = z.infer<typeof EmbedProvider>;

/** Anthropic has no embedding API, so embeddings have their own provider setting. */
export const DEFAULT_EMBED_MODELS: Readonly<Record<EmbedProvider, string>> = {
  ollama: "qwen3-embedding:0.6b",
  openai: "text-embedding-3-small",
};

export interface EmbedConfig {
  provider: EmbedProvider;
  model: string;
  /**
   * Vector length to request and enforce. Models with Matryoshka embeddings (qwen3-embedding,
   * text-embedding-3) can shorten their vectors; a fixed size lets a vector index survive a provider change.
   * Unset: the model's native size, still checked to be the same for every vector.
   */
  dimensions: number | undefined;
  /** Ollama server URL, no trailing slash. Ignored by OpenAI. */
  ollamaHost: string;
  /** Texts per request. */
  batchSize: number;
  /** Per-request timeout. */
  timeoutMs: number;
}

export type EmbedConfigInput = Partial<{ [K in keyof EmbedConfig]: string | number }>;

type Env = Record<string, string | undefined>;

export const EMBED_ENV = {
  provider: "EMBED_PROVIDER",
  model: "EMBED_MODEL",
  dimensions: "EMBED_DIMENSIONS",
  ollamaHost: "OLLAMA_HOST",
  batchSize: "EMBED_BATCH_SIZE",
  timeoutMs: "EMBED_TIMEOUT_MS",
} as const satisfies Record<keyof EmbedConfig, string>;

const positiveInt = (min = 1) => z.coerce.number().int().min(min);

const Resolved = z.object({
  provider: EmbedProvider,
  model: z.string().min(1),
  dimensions: positiveInt().optional(),
  ollamaHost: z.url(),
  batchSize: positiveInt().max(2048),
  timeoutMs: positiveInt(1000),
});

/**
 * Resolve the embedding settings, like resolveLlmConfig: `input` > `env` > defaults, empty values unset.
 * Without EMBED_PROVIDER, a present OPENAI_API_KEY picks OpenAI, otherwise Ollama.
 */
export function resolveEmbedConfig(input: EmbedConfigInput = {}, env: Env = process.env): EmbedConfig {
  const pick = (key: keyof EmbedConfig) => nonEmpty(input[key]) ?? nonEmpty(env[EMBED_ENV[key]]);
  const provider = (pick("provider") ?? (nonEmpty(env.OPENAI_API_KEY) ? "openai" : "ollama")).toString().toLowerCase();

  const raw = {
    provider,
    model: pick("model") ?? DEFAULT_EMBED_MODELS[provider as EmbedProvider],
    dimensions: pick("dimensions"),
    ollamaHost: normalizeHost(pick("ollamaHost")?.toString() ?? DEFAULT_OLLAMA_HOST),
    batchSize: pick("batchSize") ?? 32,
    timeoutMs: pick("timeoutMs") ?? 120_000,
  };
  const parsed = Resolved.safeParse(raw);
  if (!parsed.success) throw new Error(`Invalid embedding configuration:\n${z.prettifyError(parsed.error)}`);
  return { ...parsed.data, dimensions: parsed.data.dimensions };
}

/** Turns texts into vectors. Depend on this interface; tests can use `fakeEmbedder` from `/testing`. */
export interface Embedder {
  readonly provider: EmbedProvider | "custom";
  readonly model: string;
  readonly dimensions: number | undefined;
  /** One vector per text, in input order. Batches internally; every vector has the same length. */
  embed(texts: string[], opts?: CallOptions): Promise<number[][]>;
}

export interface EmbedderOptions {
  /** How long Ollama keeps the model loaded between calls. */
  keepAlive?: string;
  /** Appended to context-overflow errors, e.g. "lower the chunk size". */
  overflowHint?: string;
  /** Receives one-line notes such as retries. */
  log?: (msg: string) => void;
  /** For tests of the Ollama client. */
  fetchImpl?: typeof fetch;
}

type EmbedBatch = (texts: string[], signal: AbortSignal) => Promise<number[][]>;

/** Build an Embedder for `cfg`, with batching, timeouts, vector checks and (Ollama) one retry. */
export async function createEmbedder(cfg: EmbedConfig, opts: EmbedderOptions = {}): Promise<Embedder> {
  const batch = cfg.provider === "ollama" ? ollamaBatch(cfg, opts) : await openAiBatch(cfg);
  const isOllama = cfg.provider === "ollama";
  const retryOpts = { overflowHint: opts.overflowHint ?? "send shorter texts", log: opts.log };

  return {
    provider: cfg.provider,
    model: cfg.model,
    dimensions: cfg.dimensions,

    async embed(texts, o) {
      const vectors: number[][] = [];
      for (let i = 0; i < texts.length; i += cfg.batchSize) {
        const slice = texts.slice(i, i + cfg.batchSize);
        const signal = () => {
          const timeout = AbortSignal.timeout(cfg.timeoutMs);
          return o?.signal ? AbortSignal.any([o.signal, timeout]) : timeout;
        };
        // Like chat: hosted SDKs retry on their own; a local GPU occasionally fails a single request.
        const result = await withOneRetry(() => batch(slice, signal()), isOllama, "Embed call", retryOpts);
        if (result.length !== slice.length) {
          throw new Error(`${cfg.model} returned ${result.length} vectors for ${slice.length} texts`);
        }
        vectors.push(...result);
      }
      checkDimensions(vectors, cfg);
      return vectors;
    },
  };
}

function checkDimensions(vectors: number[][], cfg: EmbedConfig): void {
  const expected = cfg.dimensions ?? vectors[0]?.length;
  const bad = vectors.find((v) => v.length !== expected);
  if (bad) throw new Error(`${cfg.model} returned a ${bad.length}-dimensional vector, expected ${expected}`);
}

/**
 * Ollama's /api/embed, called directly: it truncates over-long inputs by default, and
 * `truncate: false` turns that into an error instead of an embedding of part of the text.
 */
function ollamaBatch(cfg: EmbedConfig, opts: EmbedderOptions): EmbedBatch {
  const fetchImpl = opts.fetchImpl ?? fetch;
  return async (texts, signal) => {
    const res = await fetchImpl(`${cfg.ollamaHost}/api/embed`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: cfg.model,
        input: texts,
        truncate: false,
        dimensions: cfg.dimensions,
        keep_alive: opts.keepAlive ?? "10m",
      }),
      signal,
    });
    const body = (await res.json().catch(() => ({}))) as { embeddings?: number[][]; error?: string };
    if (!res.ok || !body.embeddings) {
      throw new Error(`Ollama embed failed (${res.status}): ${body.error ?? "no embeddings in response"}`);
    }
    return body.embeddings;
  };
}

async function openAiBatch(cfg: EmbedConfig): Promise<EmbedBatch> {
  const { OpenAIEmbeddings } = await importOptional<typeof import("@langchain/openai")>("@langchain/openai");
  const model = new OpenAIEmbeddings({
    model: cfg.model,
    dimensions: cfg.dimensions,
    batchSize: cfg.batchSize,
    timeout: cfg.timeoutMs,
    maxRetries: 2,
  });
  // OpenAIEmbeddings takes no per-call signal; its own timeout applies instead.
  return (texts) => model.embedDocuments(texts);
}

function nonEmpty<T>(v: T | undefined): T | undefined {
  return typeof v === "string" && v.trim() === "" ? undefined : v;
}
