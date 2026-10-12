import type { CallbackHandlerMethods } from "@langchain/core/callbacks/base";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { type AIMessage, type BaseMessage, isAIMessage } from "@langchain/core/messages";
import type { ChatGeneration, LLMResult } from "@langchain/core/outputs";
import type { StructuredToolInterface } from "@langchain/core/tools";
import type { z } from "zod";
import type { LlmConfig, Provider } from "./config.js";
import { errorMessage, explainOverflow, isAbortError, isConnectionError, isContextOverflow } from "./errors.js";
import { createChatModel, type ChatModelOptions } from "./provider.js";
import { boundedJsonSchema, type SchemaLimits } from "./schema.js";
import { type LlmUsage, UsageMeter } from "./usage.js";

export interface CallOptions {
  /** Cancels the call; combined with the configured per-request timeout. */
  signal?: AbortSignal;
}

export interface StructuredOptions extends CallOptions {
  /** Function / tool name the provider sees for the structured answer, e.g. "iac_review". */
  name: string;
  /**
   * Length limits for free-text fields (see boundedJsonSchema). Applied only where the provider enforces them
   * by constrained decoding (Ollama); the answer is still validated against `schema` everywhere.
   */
  limits?: SchemaLimits;
}

/**
 * What a tool needs from an LLM. Code that depends on this interface, rather than on a LangChain model,
 * can be tested with a scripted fake (see `@meirakrish/llm-kit/testing`).
 */
export interface LlmClient {
  readonly provider: Provider | "custom";
  readonly model: string;
  /** One chat turn where the model may answer with tool calls. */
  chatWithTools(messages: BaseMessage[], tools: StructuredToolInterface[], opts?: CallOptions): Promise<AIMessage>;
  /** One chat turn whose answer is validated against `schema`. Retries once if the answer is unusable. */
  structured<S extends z.ZodType>(schema: S, messages: BaseMessage[], opts: StructuredOptions): Promise<z.infer<S>>;
  /** Token usage of the calls made since the last `takeUsage()`; resets the count. */
  takeUsage(): LlmUsage;
}

export interface LlmClientOptions extends ChatModelOptions {
  /** Use this model instead of building one from the config (tests, evals, other LangChain providers). */
  model?: BaseChatModel;
  /** Appended to context-overflow errors, e.g. "raise LLM_NUM_CTX or lower TOOL_OUTPUT_MAX_CHARS". */
  overflowHint?: string;
  /** Receives one-line notes such as retries. */
  log?: (msg: string) => void;
}

/** Errors that will fail the same way again, or mean the caller gave up. */
function isFatal(err: unknown): boolean {
  return isContextOverflow(err) || isAbortError(err) || isConnectionError(err);
}

/**
 * Run `call`, retrying once when `shouldRetry` allows it. Fatal errors (overflow, timeout, server unreachable)
 * never retry. Overflow errors are rewritten into a ContextOverflowError with an actionable message.
 */
async function withOneRetry<T>(
  call: () => Promise<T>,
  shouldRetry: boolean,
  what: string,
  opts: { overflowHint?: string; log?: (msg: string) => void },
): Promise<T> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await call();
    } catch (err) {
      if (attempt >= 2 || !shouldRetry || isFatal(err)) explainOverflow(err, opts.overflowHint);
      opts.log?.(`${what} failed, retrying once: ${errorMessage(err).split("\n")[0]?.slice(0, 120)}`);
    }
  }
}

/** Build an LlmClient for `cfg`: provider model, timeouts, retries, usage metering and Ollama safeguards. */
export async function createLlmClient(cfg: LlmConfig, opts: LlmClientOptions = {}): Promise<LlmClient> {
  const model = opts.model ?? (await createChatModel(cfg, opts));
  const isOllama = !opts.model && cfg.provider === "ollama";

  // A callback sees every model call, including the one inside withStructuredOutput,
  // whose parsed result no longer carries the message with its usage_metadata.
  const meter = new UsageMeter();
  const callbacks: CallbackHandlerMethods[] = [
    {
      handleLLMEnd: (output: LLMResult) => {
        const message = (output.generations[0]?.[0] as ChatGeneration | undefined)?.message;
        meter.record(message && isAIMessage(message) ? message.usage_metadata : undefined);
      },
    },
  ];
  const invokeConfig = (o?: CallOptions) => {
    const timeout = AbortSignal.timeout(cfg.timeoutMs);
    return { callbacks, signal: o?.signal ? AbortSignal.any([o.signal, timeout]) : timeout };
  };
  const retryOpts = { overflowHint: opts.overflowHint, log: opts.log };

  return {
    provider: opts.model ? "custom" : cfg.provider,
    model: opts.model ? ((opts.model as { model?: string }).model ?? "custom") : cfg.model,

    chatWithTools(messages, tools, o) {
      if (!model.bindTools) throw new Error(`${cfg.provider}/${cfg.model} does not support tool calling`);
      const bound = model.bindTools(tools);
      // Hosted SDKs already retry transport errors. Local GPUs occasionally fail a single request
      // (e.g. a transient "CUDA error" under WSL) and the next one succeeds.
      return withOneRetry(() => bound.invoke(messages, invokeConfig(o)), isOllama, "LLM call", retryOpts);
    },

    structured(schema, messages, o) {
      const target = isOllama && o.limits ? boundedJsonSchema(schema, o.limits) : schema;
      const runnable = model.withStructuredOutput(target as z.ZodType, { name: o.name });
      // Retry once on unusable output (truncated or invalid JSON, schema mismatch) for every provider.
      return withOneRetry(
        async () => {
          const raw = await runnable.invoke(messages, invokeConfig(o));
          // Re-validate: JSON-schema mode returns untyped data, and some providers return loosely typed JSON.
          return schema.parse(raw) as z.infer<typeof schema>;
        },
        true,
        `Structured answer "${o.name}"`,
        retryOpts,
      );
    },

    takeUsage: () => meter.take(),
  };
}
