import { HumanMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { describeLlmError } from "../src/errors.js";
import { checkOllama, ollamaCapabilities } from "../src/ollama.js";
import { describeLlmRun, llmDisabled } from "../src/report.js";
import { boundedJsonSchema } from "../src/schema.js";
import { compareSeverity, maxSeverity, meetsThreshold, parseFailOn, type Severity } from "../src/severity.js";
import { scriptedLlm } from "../src/testing.js";
import { trimToSentence, truncateMiddle } from "../src/text.js";
import { respond } from "./helpers.js";

describe("severity", () => {
  it("sorts most severe first and finds the max", () => {
    const levels: Severity[] = ["low", "critical", "medium", "high"];
    expect([...levels].sort(compareSeverity)).toEqual(["critical", "high", "medium", "low"]);
    expect(maxSeverity(levels)).toBe("critical");
    expect(maxSeverity([])).toBe("low");
  });

  it("applies fail-on thresholds", () => {
    expect(meetsThreshold("high", "high")).toBe(true);
    expect(meetsThreshold("medium", "high")).toBe(false);
    expect(meetsThreshold("critical", "none")).toBe(false);
    expect(parseFailOn(" HIGH ")).toBe("high");
    expect(() => parseFailOn("warning")).toThrow(/none, low, medium, high, critical/);
  });
});

describe("boundedJsonSchema", () => {
  const Review = z.object({
    summary: z.string(),
    findings: z.array(z.object({ issue: z.string(), fix: z.string().nullable() })),
  });

  it("adds limits by path without touching the shared schema", () => {
    const js = boundedJsonSchema(Review, {
      summary: 700,
      findings: 20,
      "findings[].issue": 400,
      "findings[].fix": 300,
    });
    expect(js.properties.summary.maxLength).toBe(700);
    expect(js.properties.findings.maxItems).toBe(20);
    expect(js.properties.findings.items.properties.issue.maxLength).toBe(400);
    expect(JSON.stringify(js.properties.findings.items.properties.fix)).toContain('"maxLength":300');
    expect(JSON.stringify(boundedJsonSchema(Review, {}))).not.toContain("maxLength");
  });

  it("fails loudly on a wrong path", () => {
    expect(() => boundedJsonSchema(Review, { sumary: 10 })).toThrow(/no property "sumary"/);
    expect(() => boundedJsonSchema(Review, { "summary[].x": 10 })).toThrow(/not an array/);
  });
});

describe("text", () => {
  it("trims a cut-off answer back to the last sentence", () => {
    const cut = "First point. Second point. Third poi";
    expect(trimToSentence(cut, cut.length)).toBe("First point. Second point.");
    expect(trimToSentence("Short.", 400)).toBe("Short.");
  });

  it("truncates the middle and says how much was cut", () => {
    const text = Array.from({ length: 200 }, (_, i) => `line ${i}`).join("\n");
    const out = truncateMiddle(text, 300);
    expect(out.length).toBeLessThanOrEqual(300);
    expect(out).toMatch(/^line 0\n/);
    expect(out).toMatch(/line 199$/);
    expect(out).toMatch(/\[\d+ characters truncated\]/);
    expect(truncateMiddle("short", 300)).toBe("short");
  });
});

describe("ollama", () => {
  it("checks the server and the model", async () => {
    const ok = await checkOllama("http://x", "qwen", {
      fetchImpl: respond(200, { models: [{ name: "qwen:latest" }] }),
    });
    expect(ok).toMatchObject({ reachable: true, modelAvailable: true, error: undefined });

    const missing = await checkOllama("http://x", "llama3", { fetchImpl: respond(200, { models: [{ name: "q" }] }) });
    expect(missing.error).toBe("model llama3 is not pulled (run: ollama pull llama3)");

    const down = (async () => Promise.reject(new TypeError("fetch failed"))) as typeof fetch;
    expect((await checkOllama("http://x", "m", { fetchImpl: down })).reachable).toBe(false);
  });

  it("reads model capabilities, empty when unknown", async () => {
    expect(
      await ollamaCapabilities("http://x/", "m", respond(200, { capabilities: ["tools", 3, "thinking"] })),
    ).toEqual(["tools", "thinking"]);
    expect(await ollamaCapabilities("http://x", "m", respond(404, {}))).toEqual([]);
  });
});

describe("errors and reports", () => {
  it("describes LLM failures in one line", () => {
    expect(describeLlmError(Object.assign(new Error("x"), { name: "TimeoutError" }))).toBe("request timed out");
    expect(describeLlmError(new TypeError("fetch failed"))).toMatch(/could not connect/);
    expect(describeLlmError(new Error("bad thing\nstack"))).toBe("bad thing");
  });

  it("renders the LLM footer line", () => {
    expect(describeLlmRun(llmDisabled())).toBe("LLM: disabled (rule-based checks only)");
    const usage = { calls: 3, promptTokens: 12345, outputTokens: 678, peakPromptTokens: 5000 };
    expect(describeLlmRun({ provider: "ollama", model: "q", ok: true, error: null, usage, durationMs: 4200 })).toBe(
      "LLM: ollama/q · 3 calls · 12,345 prompt / 678 output tokens · 4.2s",
    );
  });
});

describe("scriptedLlm", () => {
  it("plays back a script, validates answers and records calls", async () => {
    const llm = scriptedLlm({ structured: [{ n: 1 }, { n: "x" }] });
    const S = z.object({ n: z.number() });
    expect(await llm.structured(S, [new HumanMessage("q")], { name: "count" })).toEqual({ n: 1 });
    await expect(llm.structured(S, [], { name: "count" })).rejects.toThrow();
    expect(llm.calls.map((c) => c.name)).toEqual(["count", "count"]);
    expect(llm.takeUsage().calls).toBe(2);
  });
});
