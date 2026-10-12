import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { LlmConfig } from "./config.js";
import { ollamaCapabilities, StrictChatOllama } from "./ollama.js";

export interface ChatModelOptions {
  /**
   * Let thinking models (Qwen3, DeepSeek-R1, ...) reason before answering. Off by default: for structured
   * DevOps answers it is slower and eats into the output cap. Only sent to Ollama models that support it.
   */
  think?: boolean;
  /** How long Ollama keeps the model loaded between calls. */
  keepAlive?: string;
  /** For tests of the Ollama capability probe. */
  fetchImpl?: typeof fetch;
}

/**
 * Build a LangChain chat model for the configured provider, with deterministic settings (temperature 0 where
 * the model accepts it) and an output cap. Hosted SDKs are imported lazily, so a tool that only uses Ollama
 * does not need them installed.
 */
export async function createChatModel(cfg: LlmConfig, opts: ChatModelOptions = {}): Promise<BaseChatModel> {
  switch (cfg.provider) {
    case "ollama": {
      const caps = await ollamaCapabilities(cfg.ollamaHost, cfg.model, opts.fetchImpl);
      return new StrictChatOllama({
        model: cfg.model,
        baseUrl: cfg.ollamaHost,
        temperature: 0,
        // Ollama's default context silently truncates; much larger contexts reserve KV-cache VRAM and can OOM.
        numCtx: cfg.numCtx,
        // Backstop for runaway generations (small models sometimes write one string forever).
        numPredict: cfg.maxOutputTokens,
        keepAlive: opts.keepAlive ?? "10m",
        // Models without thinking support reject the option, so only send it to those that have it.
        think: caps.includes("thinking") ? (opts.think ?? false) : undefined,
      });
    }
    case "anthropic": {
      const { ChatAnthropic } = await importOptional<typeof import("@langchain/anthropic")>("@langchain/anthropic");
      return new ChatAnthropic({ model: cfg.model, temperature: 0, maxTokens: cfg.maxOutputTokens, maxRetries: 2 });
    }
    case "openai": {
      const { ChatOpenAI } = await importOptional<typeof import("@langchain/openai")>("@langchain/openai");
      // No temperature: GPT-5 / o-series reasoning models reject anything but the default.
      return new ChatOpenAI({ model: cfg.model, maxRetries: 2 });
    }
  }
}

async function importOptional<T>(pkg: string): Promise<T> {
  try {
    return (await import(pkg)) as T;
  } catch (err) {
    if ((err as { code?: string }).code === "ERR_MODULE_NOT_FOUND") {
      throw new Error(`The ${pkg} package is needed for this provider. Install it: npm install ${pkg}`);
    }
    throw err;
  }
}
