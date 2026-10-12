import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, type BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { RunnableLambda } from "@langchain/core/runnables";

type Reply = string | Error;

/**
 * A LangChain chat model that plays back replies. Structured output parses the reply text as JSON,
 * so tests exercise the real client code paths (callbacks, retries, validation).
 */
export class FakeChatModel extends BaseChatModel {
  model = "fake-model";
  readonly seen: BaseMessage[][] = [];
  readonly structuredTargets: unknown[] = [];

  constructor(private replies: Reply[]) {
    super({});
  }

  _llmType() {
    return "fake";
  }

  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    this.seen.push(messages);
    const next = this.replies.shift() ?? "(no reply)";
    if (next instanceof Error) throw next;
    const message = new AIMessage({
      content: next,
      usage_metadata: { input_tokens: 100 * this.seen.length, output_tokens: 10, total_tokens: 0 },
    });
    return { generations: [{ text: next, message }] };
  }

  override bindTools() {
    return this as never;
  }

  override withStructuredOutput(target: unknown) {
    this.structuredTargets.push(target);
    return this.pipe(
      RunnableLambda.from((m: AIMessage) => JSON.parse(String(m.content)) as Record<string, unknown>),
    ) as never;
  }
}

export function fakeConfig(overrides = {}) {
  return {
    provider: "ollama" as const,
    model: "m",
    ollamaHost: "http://localhost:11434",
    numCtx: 4096,
    maxOutputTokens: 512,
    timeoutMs: 5000,
    ...overrides,
  };
}

export function respond(status: number, body: unknown): typeof fetch {
  return (async () => new Response(JSON.stringify(body), { status })) as typeof fetch;
}
