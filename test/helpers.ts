import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
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

type StreamStep = string | Error | { delayMs: number };

/**
 * A chat model that streams scripted pieces. Each call plays the next script: strings are yielded as
 * chunks, `{ delayMs }` waits (aborting if the call's signal fires), an Error is thrown. The last chunk
 * carries usage, as Ollama's does.
 */
export class FakeStreamingModel extends BaseChatModel {
  model = "fake-stream";
  calls = 0;
  aborted = 0;

  constructor(private scripts: StreamStep[][]) {
    super({});
  }

  _llmType() {
    return "fake-stream";
  }

  async _generate(): Promise<ChatResult> {
    throw new Error("FakeStreamingModel only streams");
  }

  override async *_streamResponseChunks(
    _messages: BaseMessage[],
    options: this["ParsedCallOptions"],
  ): AsyncGenerator<ChatGenerationChunk> {
    const steps = this.scripts[this.calls++] ?? [];
    const signal = options.signal;
    for (const step of steps) {
      if (step instanceof Error) throw step;
      if (typeof step === "object") {
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, step.delayMs);
          signal?.addEventListener("abort", () => {
            clearTimeout(timer);
            this.aborted++;
            reject(Object.assign(new Error("Aborted"), { name: "AbortError" }));
          });
        });
        continue;
      }
      yield new ChatGenerationChunk({ text: step, message: new AIMessageChunk({ content: step }) });
    }
    yield new ChatGenerationChunk({
      text: "",
      message: new AIMessageChunk({
        content: "",
        usage_metadata: { input_tokens: 50, output_tokens: steps.length, total_tokens: 50 + steps.length },
      }),
    });
  }
}
