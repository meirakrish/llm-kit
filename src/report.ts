import { z } from "zod";
import { LlmUsage } from "./usage.js";

/** What every llm-kit tool records about its LLM step, in JSON reports and the markdown footer. */
export const LlmRunInfo = z.object({
  /** "none" when the LLM step was disabled (--no-llm). */
  provider: z.string(),
  model: z.string(),
  ok: z.boolean(),
  /** One-line reason the LLM step failed; the tool fell back to its rule-based result. */
  error: z.string().nullable(),
  usage: LlmUsage.optional(),
  durationMs: z.number().nonnegative().optional(),
});
export type LlmRunInfo = z.infer<typeof LlmRunInfo>;

/** Header of every JSON report, so CI and dashboards can read any tool's output the same way. */
export const ReportMeta = z.object({
  tool: z.string(),
  toolVersion: z.string(),
  /** Bump on breaking changes to the tool's report shape. */
  schemaVersion: z.number().int().positive(),
  generatedAt: z.iso.datetime(),
  llm: LlmRunInfo,
});
export type ReportMeta = z.infer<typeof ReportMeta>;

export function llmDisabled(): LlmRunInfo {
  return { provider: "none", model: "none", ok: false, error: null };
}

/** Footer line for markdown reports, e.g. "LLM: ollama/qwen3.5:9b · 3 calls · 12,345 prompt tokens". */
export function describeLlmRun(info: LlmRunInfo): string {
  if (info.provider === "none") return "LLM: disabled (rule-based checks only)";
  const parts = [`LLM: ${info.provider}/${info.model}`];
  if (!info.ok) parts.push(`failed (${info.error ?? "unknown error"}), rule-based results only`);
  if (info.usage?.calls) {
    const n = (x: number) => x.toLocaleString("en-US");
    parts.push(`${n(info.usage.calls)} call${info.usage.calls === 1 ? "" : "s"}`);
    parts.push(`${n(info.usage.promptTokens)} prompt / ${n(info.usage.outputTokens)} output tokens`);
  }
  if (info.durationMs !== undefined) parts.push(`${(info.durationMs / 1000).toFixed(1)}s`);
  return parts.join(" · ");
}
