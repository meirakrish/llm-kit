import { HumanMessage } from "@langchain/core/messages";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { createLlmClient } from "../src/client.js";
import { ContextOverflowError } from "../src/errors.js";
import { FakeChatModel, fakeConfig } from "./helpers.js";

const Answer = z.object({ verdict: z.enum(["ok", "bad"]), note: z.string() });
const msgs = [new HumanMessage("hi")];

describe("createLlmClient.structured", () => {
  it("returns the validated answer and meters usage", async () => {
    const model = new FakeChatModel(['{"verdict":"ok","note":"fine"}']);
    const llm = await createLlmClient(fakeConfig(), { model });

    expect(await llm.structured(Answer, msgs, { name: "answer" })).toEqual({ verdict: "ok", note: "fine" });
    expect(llm.takeUsage()).toEqual({ calls: 1, promptTokens: 100, outputTokens: 10, peakPromptTokens: 100 });
    expect(llm.takeUsage().calls).toBe(0);
  });

  it("retries once on an answer that does not match the schema", async () => {
    const logs: string[] = [];
    const model = new FakeChatModel(['{"verdict":"maybe"}', '{"verdict":"bad","note":"x"}']);
    const llm = await createLlmClient(fakeConfig(), { model, log: (m) => logs.push(m) });

    expect(await llm.structured(Answer, msgs, { name: "answer" })).toEqual({ verdict: "bad", note: "x" });
    expect(logs).toHaveLength(1);
    expect(logs[0]).toMatch(/Structured answer "answer" failed, retrying once/);
  });

  it("gives up after the second bad answer", async () => {
    const model = new FakeChatModel(["not json", "still not json"]);
    const llm = await createLlmClient(fakeConfig(), { model });
    await expect(llm.structured(Answer, msgs, { name: "answer" })).rejects.toThrow();
    expect(model.seen).toHaveLength(2);
  });

  it("does not retry a context overflow and explains it", async () => {
    const overflow = new Error('{"error":"request (20000 tokens) exceeds the available context size (16384 tokens)"}');
    const model = new FakeChatModel([overflow, '{"verdict":"ok","note":"never reached"}']);
    const llm = await createLlmClient(fakeConfig(), { model, overflowHint: "raise LLM_NUM_CTX" });

    const err = await llm.structured(Answer, msgs, { name: "answer" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ContextOverflowError);
    expect((err as Error).message).toBe(
      "prompt (20000 tokens) does not fit in the context window (16384); raise LLM_NUM_CTX",
    );
    expect(model.seen).toHaveLength(1);
  });

  it("does not retry when the server is unreachable", async () => {
    const down = Object.assign(new TypeError("fetch failed"), { cause: { code: "ECONNREFUSED" } });
    const model = new FakeChatModel([down, '{"verdict":"ok","note":"x"}']);
    const llm = await createLlmClient(fakeConfig(), { model });
    await expect(llm.structured(Answer, msgs, { name: "answer" })).rejects.toThrow("fetch failed");
    expect(model.seen).toHaveLength(1);
  });

  it("reports an injected model as custom", async () => {
    const llm = await createLlmClient(fakeConfig(), { model: new FakeChatModel([]) });
    expect([llm.provider, llm.model]).toEqual(["custom", "fake-model"]);
  });
});

describe("createLlmClient.chatWithTools", () => {
  it("does not retry an injected model's errors (no Ollama GPU retry)", async () => {
    const model = new FakeChatModel([new Error("CUDA error"), "ok"]);
    const llm = await createLlmClient(fakeConfig(), { model });
    await expect(llm.chatWithTools(msgs, [])).rejects.toThrow("CUDA error");
  });

  it("meters every call", async () => {
    const model = new FakeChatModel(["a", "b"]);
    const llm = await createLlmClient(fakeConfig(), { model });
    await llm.chatWithTools(msgs, []);
    const reply = await llm.chatWithTools(msgs, []);
    expect(reply.content).toBe("b");
    expect(llm.takeUsage()).toEqual({ calls: 2, promptTokens: 300, outputTokens: 20, peakPromptTokens: 200 });
  });
});
