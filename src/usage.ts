import { z } from "zod";
import type { UsageMetadata } from "@langchain/core/messages";

/** Token counts for a group of LLM calls (one triage step, one investigation, one review). */
export const LlmUsage = z.object({
  calls: z.number().int().nonnegative(),
  promptTokens: z.number().int().nonnegative(),
  outputTokens: z.number().int().nonnegative(),
  /** Largest single prompt. For Ollama, compare with numCtx: the whole conversation must fit in it. */
  peakPromptTokens: z.number().int().nonnegative(),
});
export type LlmUsage = z.infer<typeof LlmUsage>;

export function emptyUsage(): LlmUsage {
  return { calls: 0, promptTokens: 0, outputTokens: 0, peakPromptTokens: 0 };
}

export function addUsage(a: LlmUsage, b: LlmUsage): LlmUsage {
  return {
    calls: a.calls + b.calls,
    promptTokens: a.promptTokens + b.promptTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    peakPromptTokens: Math.max(a.peakPromptTokens, b.peakPromptTokens),
  };
}

/** Adds up token counts per call until `take()` returns them and starts over. */
export class UsageMeter {
  private usage = emptyUsage();

  record(tokens: UsageMetadata | undefined): void {
    const prompt = tokens?.input_tokens ?? 0;
    this.usage.calls++;
    this.usage.promptTokens += prompt;
    this.usage.outputTokens += tokens?.output_tokens ?? 0;
    this.usage.peakPromptTokens = Math.max(this.usage.peakPromptTokens, prompt);
  }

  /** Usage since the last call; resets the count. */
  take(): LlmUsage {
    const usage = this.usage;
    this.usage = emptyUsage();
    return usage;
  }

  /** Usage since the last `take()`, without resetting. */
  peek(): LlmUsage {
    return { ...this.usage };
  }
}
