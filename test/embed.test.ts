import { describe, expect, it, vi } from "vitest";
import { createEmbedder, resolveEmbedConfig, type EmbedConfig } from "../src/embed.js";
import { ContextOverflowError } from "../src/errors.js";
import { fakeEmbedder } from "../src/testing.js";

describe("resolveEmbedConfig", () => {
  it("defaults to local Ollama with qwen3-embedding", () => {
    expect(resolveEmbedConfig({}, {})).toEqual({
      provider: "ollama",
      model: "qwen3-embedding:0.6b",
      dimensions: undefined,
      ollamaHost: "http://localhost:11434",
      batchSize: 32,
      timeoutMs: 120000,
    });
  });

  it("picks OpenAI from its API key, but not Anthropic from its key", () => {
    expect(resolveEmbedConfig({}, { OPENAI_API_KEY: "k" })).toMatchObject({
      provider: "openai",
      model: "text-embedding-3-small",
    });
    expect(resolveEmbedConfig({}, { ANTHROPIC_API_KEY: "k" }).provider).toBe("ollama");
  });

  it("lets flags win over env, and env over defaults", () => {
    const env = { EMBED_MODEL: "env-model", EMBED_DIMENSIONS: "1024", OLLAMA_HOST: "0.0.0.0:11434" };
    expect(resolveEmbedConfig({ model: "flag-model" }, env)).toMatchObject({
      model: "flag-model",
      dimensions: 1024,
      ollamaHost: "http://127.0.0.1:11434",
    });
  });

  it("treats empty env values as unset", () => {
    expect(resolveEmbedConfig({}, { EMBED_DIMENSIONS: "", EMBED_MODEL: " " })).toMatchObject({
      model: "qwen3-embedding:0.6b",
      dimensions: undefined,
    });
  });

  it("rejects bad values with a readable error", () => {
    expect(() => resolveEmbedConfig({}, { EMBED_PROVIDER: "anthropic" })).toThrow(/Invalid embedding configuration/);
    expect(() => resolveEmbedConfig({ dimensions: 0 })).toThrow(/dimensions/);
    expect(() => resolveEmbedConfig({ batchSize: "many" })).toThrow(/batchSize/);
  });
});

const cfg = (over: Partial<EmbedConfig> = {}): EmbedConfig => ({
  ...resolveEmbedConfig({}, {}),
  ...over,
});

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

/** A fake Ollama that embeds each text as [length, index-in-batch]. */
function fakeOllama(...overrides: Array<Response | Error>) {
  const requests: Array<{ url: string; body: Record<string, unknown> }> = [];
  const fetchImpl = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body)) as { input: string[] };
    requests.push({ url: String(url), body });
    const next = overrides.shift();
    if (next instanceof Error) throw next;
    return next ?? json(200, { embeddings: body.input.map((t, i) => [t.length, i]) });
  });
  return { fetchImpl: fetchImpl as unknown as typeof fetch, requests };
}

describe("createEmbedder (Ollama)", () => {
  it("sends truncate: false and the dimensions, and keeps input order across batches", async () => {
    const { fetchImpl, requests } = fakeOllama();
    const embedder = await createEmbedder(cfg({ batchSize: 2, dimensions: 2 }), { fetchImpl });

    const vectors = await embedder.embed(["a", "bb", "ccc"]);

    expect(vectors).toEqual([
      [1, 0],
      [2, 1],
      [3, 0],
    ]);
    expect(requests.map((r) => r.body.input)).toEqual([["a", "bb"], ["ccc"]]);
    expect(requests[0]).toMatchObject({
      url: "http://localhost:11434/api/embed",
      body: { model: "qwen3-embedding:0.6b", truncate: false, dimensions: 2 },
    });
  });

  it("makes no request for no texts", async () => {
    const { fetchImpl, requests } = fakeOllama();
    expect(await (await createEmbedder(cfg(), { fetchImpl })).embed([])).toEqual([]);
    expect(requests).toEqual([]);
  });

  it("rejects vectors of the wrong length", async () => {
    const { fetchImpl } = fakeOllama();
    const embedder = await createEmbedder(cfg({ dimensions: 1024 }), { fetchImpl });
    await expect(embedder.embed(["a"])).rejects.toThrow(/2-dimensional vector, expected 1024/);
  });

  it("rejects vectors of mixed lengths when no dimensions are set", async () => {
    const { fetchImpl } = fakeOllama(json(200, { embeddings: [[1, 2], [3]] }));
    const embedder = await createEmbedder(cfg(), { fetchImpl });
    await expect(embedder.embed(["a", "b"])).rejects.toThrow(/1-dimensional vector, expected 2/);
  });

  it("rejects a response with the wrong number of vectors", async () => {
    const { fetchImpl } = fakeOllama(json(200, { embeddings: [[1, 2]] }));
    const embedder = await createEmbedder(cfg(), { fetchImpl });
    await expect(embedder.embed(["a", "b"])).rejects.toThrow(/returned 1 vectors for 2 texts/);
  });

  it("turns an over-long input into a ContextOverflowError, without retrying", async () => {
    const { fetchImpl, requests } = fakeOllama(json(400, { error: "the input length exceeds the context length" }));
    const embedder = await createEmbedder(cfg(), { fetchImpl, overflowHint: "lower the chunk size" });

    const err = await embedder.embed(["long"]).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ContextOverflowError);
    expect((err as Error).message).toMatch(/lower the chunk size/);
    expect(requests).toHaveLength(1);
  });

  it("retries a transient server error once", async () => {
    const log = vi.fn();
    const { fetchImpl, requests } = fakeOllama(json(500, { error: "CUDA error: unspecified launch failure" }));
    const embedder = await createEmbedder(cfg(), { fetchImpl, log });

    expect(await embedder.embed(["a"])).toEqual([[1, 0]]);
    expect(requests).toHaveLength(2);
    expect(log).toHaveBeenCalledWith(expect.stringMatching(/Embed call failed, retrying once: .*CUDA error/));
  });

  it("gives up after the second failure", async () => {
    const { fetchImpl, requests } = fakeOllama(json(500, { error: "boom" }), json(500, { error: "boom again" }));
    const embedder = await createEmbedder(cfg(), { fetchImpl });

    await expect(embedder.embed(["a"])).rejects.toThrow(/Ollama embed failed \(500\): boom again/);
    expect(requests).toHaveLength(2);
  });

  it("does not retry when Ollama is unreachable", async () => {
    const down = Object.assign(new TypeError("fetch failed"), { cause: { code: "ECONNREFUSED" } });
    const { fetchImpl, requests } = fakeOllama(down);
    const embedder = await createEmbedder(cfg(), { fetchImpl });

    await expect(embedder.embed(["a"])).rejects.toThrow(/fetch failed/);
    expect(requests).toHaveLength(1);
  });

  it("passes the caller's signal through", async () => {
    const controller = new AbortController();
    let seen: AbortSignal | undefined;
    const fetchImpl = (async (_url: string, init?: RequestInit) => {
      seen = init?.signal ?? undefined;
      return json(200, { embeddings: [[1]] });
    }) as unknown as typeof fetch;
    const embedder = await createEmbedder(cfg(), { fetchImpl });

    await embedder.embed(["a"], { signal: controller.signal });
    controller.abort();
    expect(seen?.aborted).toBe(true);
  });
});

describe("fakeEmbedder", () => {
  it("returns deterministic unit vectors and records calls", async () => {
    const embedder = fakeEmbedder({ dimensions: 4 });
    const [a1, b] = await embedder.embed(["alpha", "beta"]);
    const [a2] = await embedder.embed(["alpha"]);

    expect(a1).toHaveLength(4);
    expect(a1).toEqual(a2);
    expect(a1).not.toEqual(b);
    expect(Math.hypot(...a1!)).toBeCloseTo(1);
    expect(embedder.calls).toEqual([["alpha", "beta"], ["alpha"]]);
  });

  it("can be told to fail", async () => {
    const embedder = fakeEmbedder({ fail: (t) => (t.includes("bad") ? new Error("nope") : undefined) });
    await expect(embedder.embed(["ok", "bad text"])).rejects.toThrow("nope");
  });
});
