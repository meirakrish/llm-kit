import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";
import type { LlmClient, StructuredOptions } from "./client.js";
import type { Embedder } from "./embed.js";
import { emptyUsage } from "./usage.js";

export interface ScriptedLlmScript {
  /** Replies to chatWithTools, in order. An Error is thrown instead of returned. */
  chat?: Array<AIMessage | Error>;
  /**
   * Answers to structured(), in order, or one function deciding per call. An Error is thrown.
   * Answers are validated against the schema, like the real client does.
   */
  structured?: Array<unknown> | ((name: string, messages: BaseMessage[]) => unknown);
}

export interface RecordedCall {
  kind: "chat" | "structured";
  messages: BaseMessage[];
  tools?: string[];
  name?: string;
}

/** An LlmClient that plays back a script and records what it was asked. For unit tests. */
export interface ScriptedLlm extends LlmClient {
  readonly calls: RecordedCall[];
}

export function scriptedLlm(script: ScriptedLlmScript = {}): ScriptedLlm {
  const chat = [...(script.chat ?? [])];
  const answers = Array.isArray(script.structured) ? [...script.structured] : null;
  const calls: RecordedCall[] = [];
  let usage = emptyUsage();
  const count = () => {
    usage = { ...usage, calls: usage.calls + 1 };
  };

  return {
    provider: "custom",
    model: "scripted",
    calls,
    async chatWithTools(messages: BaseMessage[], tools: StructuredToolInterface[]) {
      calls.push({ kind: "chat", messages, tools: tools.map((t) => t.name) });
      count();
      const next = chat.shift();
      if (next instanceof Error) throw next;
      return next ?? new AIMessage("(script exhausted)");
    },
    async structured(schema, messages, opts: StructuredOptions) {
      calls.push({ kind: "structured", messages, name: opts.name });
      count();
      const next = answers ? answers.shift() : (script.structured as Function | undefined)?.(opts.name, messages);
      if (next instanceof Error) throw next;
      if (next === undefined) throw new Error(`scriptedLlm: no structured answer left for "${opts.name}"`);
      return schema.parse(next);
    },
    takeUsage() {
      const u = usage;
      usage = emptyUsage();
      return u;
    },
  };
}

/** A deterministic Embedder that records what it was asked. For unit tests. */
export interface FakeEmbedder extends Embedder {
  readonly dimensions: number;
  /** The texts of each embed() call. */
  readonly calls: string[][];
}

/**
 * The same text always gets the same unit vector; different texts almost always get different ones.
 * Vectors carry no meaning, so this suits storage and plumbing tests, not retrieval quality.
 * `fail` makes the call that contains a matching text throw instead.
 */
export function fakeEmbedder(
  opts: { dimensions?: number; fail?: (text: string) => Error | undefined } = {},
): FakeEmbedder {
  const dimensions = opts.dimensions ?? 8;
  const calls: string[][] = [];
  return {
    provider: "custom",
    model: "fake-embedder",
    dimensions,
    calls,
    async embed(texts) {
      calls.push(texts);
      for (const text of texts) {
        const err = opts.fail?.(text);
        if (err) throw err;
      }
      return texts.map((text) => hashVector(text, dimensions));
    },
  };
}

function hashVector(text: string, dimensions: number): number[] {
  // FNV-1a seed, then xorshift: cheap, deterministic, spread over [-1, 1).
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 0x01000193);
  const v = Array.from({ length: dimensions }, () => {
    h ^= h << 13;
    h ^= h >>> 17;
    h ^= h << 5;
    return (h >>> 0) / 0x80000000 - 1;
  });
  const norm = Math.hypot(...v) || 1;
  return v.map((x) => x / norm);
}
