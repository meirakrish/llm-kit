import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { StructuredToolInterface } from "@langchain/core/tools";
import type { LlmClient, StructuredOptions } from "./client.js";
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
