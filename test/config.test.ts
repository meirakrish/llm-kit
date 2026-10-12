import { describe, expect, it } from "vitest";
import { normalizeHost, resolveLlmConfig } from "../src/config.js";

describe("resolveLlmConfig", () => {
  it("defaults to local Ollama", () => {
    expect(resolveLlmConfig({}, {})).toEqual({
      provider: "ollama",
      model: "qwen3.5:9b",
      ollamaHost: "http://localhost:11434",
      numCtx: 16384,
      maxOutputTokens: 4096,
      timeoutMs: 180000,
    });
  });

  it("picks a hosted provider from its API key", () => {
    expect(resolveLlmConfig({}, { ANTHROPIC_API_KEY: "k" }).provider).toBe("anthropic");
    expect(resolveLlmConfig({}, { OPENAI_API_KEY: "k" }).model).toBe("gpt-5-mini");
    expect(resolveLlmConfig({}, { ANTHROPIC_API_KEY: "" }).provider).toBe("ollama");
  });

  it("lets flags win over env, and env over defaults", () => {
    const env = { LLM_PROVIDER: "openai", LLM_MODEL: "env-model", LLM_NUM_CTX: "8192" };
    const cfg = resolveLlmConfig({ model: "flag-model" }, env);
    expect(cfg).toMatchObject({ provider: "openai", model: "flag-model", numCtx: 8192 });
  });

  it("treats empty env values as unset", () => {
    expect(resolveLlmConfig({}, { LLM_MODEL: " ", OLLAMA_HOST: "" }).model).toBe("qwen3.5:9b");
  });

  it("rejects bad values with a readable error", () => {
    expect(() => resolveLlmConfig({}, { LLM_PROVIDER: "gemini" })).toThrow(/Invalid LLM configuration/);
    expect(() => resolveLlmConfig({ numCtx: "lots" })).toThrow(/numCtx/);
    expect(() => resolveLlmConfig({ numCtx: 512 })).toThrow(/numCtx/);
  });
});

describe("normalizeHost", () => {
  it.each([
    ["localhost:11434", "http://localhost:11434"],
    ["0.0.0.0:11434", "http://127.0.0.1:11434"],
    ["https://gpu.example.com/", "https://gpu.example.com"],
  ])("%s -> %s", (input, expected) => expect(normalizeHost(input)).toBe(expected));
});
