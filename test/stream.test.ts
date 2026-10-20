import { HumanMessage } from "@langchain/core/messages";
import { describe, expect, it, vi } from "vitest";
import { createLlmClient } from "../src/client.js";
import { scriptedLlm } from "../src/testing.js";
import { fakeConfig, FakeStreamingModel } from "./helpers.js";

const ask = [new HumanMessage("hi")];

async function collect(stream: AsyncIterable<string>): Promise<string[]> {
  const out: string[] = [];
  for await (const piece of stream) out.push(piece);
  return out;
}

describe("createLlmClient.stream", () => {
  it("yields text pieces as they arrive and meters usage", async () => {
    const model = new FakeStreamingModel([["Hel", "lo", " world"]]);
    const llm = await createLlmClient(fakeConfig(), { model });

    expect(await collect(llm.stream(ask))).toEqual(["Hel", "lo", " world"]);
    expect(llm.takeUsage()).toMatchObject({ calls: 1, promptTokens: 50 });
  });

  it("times out when the answer stalls, but allows a slow first piece", async () => {
    // First piece after 150 ms (under timeoutMs), then a stall longer than the idle limit.
    const model = new FakeStreamingModel([[{ delayMs: 150 }, "a", { delayMs: 500 }, "b"]]);
    const llm = await createLlmClient(fakeConfig({ timeoutMs: 1000 }), { model });
    const seen: string[] = [];

    const err = await (async () => {
      for await (const piece of llm.stream(ask, { idleTimeoutMs: 100 })) seen.push(piece);
    })().catch((e: unknown) => e);

    expect(seen).toEqual(["a"]);
    expect((err as Error).name).toBe("TimeoutError");
    expect((err as Error).message).toMatch(/no output for 100 ms/);
  });

  it("times out when nothing arrives within timeoutMs", async () => {
    const model = new FakeStreamingModel([[{ delayMs: 500 }, "late"]]);
    const llm = await createLlmClient(fakeConfig({ timeoutMs: 100 }), { model });

    await expect(collect(llm.stream(ask))).rejects.toMatchObject({ name: "TimeoutError" });
  });

  it("cancels the request when the caller stops early", async () => {
    const model = new FakeStreamingModel([["a", { delayMs: 5000 }, "never"]]);
    const llm = await createLlmClient(fakeConfig(), { model });

    for await (const piece of llm.stream(ask)) {
      expect(piece).toBe("a");
      break;
    }
    await vi.waitFor(() => expect(model.aborted).toBe(1));
  });

  it("honours the caller's signal", async () => {
    const model = new FakeStreamingModel([["a", { delayMs: 5000 }, "never"]]);
    const llm = await createLlmClient(fakeConfig(), { model });
    const controller = new AbortController();

    const run = collect(llm.stream(ask, { signal: controller.signal }));
    setTimeout(() => controller.abort(), 20);
    await expect(run).rejects.toMatchObject({ name: "AbortError" });
  });

  it("does not retry an injected model's errors, before or after text", async () => {
    const before = new FakeStreamingModel([[new Error("boom")], ["ok"]]);
    const llm = await createLlmClient(fakeConfig(), { model: before });
    await expect(collect(llm.stream(ask))).rejects.toThrow("boom");
    expect(before.calls).toBe(1);

    const midway = new FakeStreamingModel([["a", new Error("cut off")]]);
    const llm2 = await createLlmClient(fakeConfig(), { model: midway });
    const seen: string[] = [];
    await expect(
      (async () => {
        for await (const p of llm2.stream(ask)) seen.push(p);
      })(),
    ).rejects.toThrow("cut off");
    expect(seen).toEqual(["a"]);
  });

  it("turns an Ollama context overflow into a ContextOverflowError", async () => {
    const overflow = new Error("request (20000 tokens) exceeds the available context size (16384 tokens)");
    const model = new FakeStreamingModel([[overflow]]);
    const llm = await createLlmClient(fakeConfig(), { model, overflowHint: "send fewer passages" });

    await expect(collect(llm.stream(ask))).rejects.toThrow(/20000 tokens.*16384.*send fewer passages/);
  });
});

describe("scriptedLlm.stream", () => {
  it("plays back pieces and records calls", async () => {
    const llm = scriptedLlm({ stream: [["The answer", " is [1]."], new Error("down")] });

    expect(await collect(llm.stream(ask))).toEqual(["The answer", " is [1]."]);
    await expect(collect(llm.stream(ask))).rejects.toThrow("down");
    expect(llm.calls.map((c) => c.kind)).toEqual(["stream", "stream"]);
    expect(llm.takeUsage().calls).toBe(2);
  });
});
