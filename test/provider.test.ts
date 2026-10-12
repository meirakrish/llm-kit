import { describe, expect, it } from "vitest";
import { StrictChatOllama } from "../src/ollama.js";
import { createChatModel } from "../src/provider.js";
import { fakeConfig, respond } from "./helpers.js";

describe("createChatModel", () => {
  it("builds a strict Ollama model and disables thinking only where supported", async () => {
    const thinking = await createChatModel(fakeConfig(), { fetchImpl: respond(200, { capabilities: ["thinking"] }) });
    expect(thinking).toBeInstanceOf(StrictChatOllama);
    expect((thinking as StrictChatOllama).invocationParams()).toMatchObject({ truncate: false, think: false });

    const plain = await createChatModel(fakeConfig(), { fetchImpl: respond(200, { capabilities: ["completion"] }) });
    expect((plain as StrictChatOllama).invocationParams().think).toBeUndefined();
    expect((plain as StrictChatOllama).invocationParams().options).toMatchObject({ num_ctx: 4096, temperature: 0 });
  });

  it("builds hosted models lazily", async () => {
    process.env.ANTHROPIC_API_KEY ??= "test";
    process.env.OPENAI_API_KEY ??= "test";
    const claude = await createChatModel(fakeConfig({ provider: "anthropic", model: "claude-sonnet-5-5" }));
    expect(claude._llmType()).toBe("anthropic");
    const gpt = await createChatModel(fakeConfig({ provider: "openai", model: "gpt-5-mini" }));
    expect(gpt._llmType()).toBe("openai");
  });
});
