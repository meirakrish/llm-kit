# llm-kit

The LLM layer shared by two DevOps tools:

- [**k8s-devops-agent**](https://github.com/meirakrish/k8s-devops-agent): read-only Kubernetes health checks with LLM triage and root-cause investigation
- [**iac-pr-reviewer**](https://github.com/meirakrish/iac-pr-reviewer): risk review of Terraform plans in pull requests

Both tools follow the same pattern: deterministic rules first, an LLM on top to reason about what the rules found, and a rule-based fallback when the LLM fails. This package is the part they have in common: talking to a model reliably, with local models treated as a first-class target rather than an afterthought.

## What it handles

| Problem                                                            | What llm-kit does                                                                                                                                    |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Switching between local and hosted models                          | One config for Ollama, Anthropic and OpenAI. A present API key picks its provider; otherwise local Ollama. Hosted SDKs load only when used.          |
| Ollama silently drops the start of prompts longer than its context | Sends `truncate: false`, so an oversized prompt fails with a clear `ContextOverflowError` ("prompt (20000 tokens) does not fit…; raise LLM_NUM_CTX") |
| Small models write one string forever                              | Length limits on structured-output fields, enforced by Ollama's constrained decoding, plus an output token cap                                       |
| Thinking models spend the output budget reasoning                  | Turns thinking off, only for models that support it (others reject the option)                                                                       |
| Invalid or truncated JSON answers                                  | Validates every structured answer against its zod schema and retries once                                                                            |
| Transient local GPU failures (e.g. CUDA errors under WSL)          | Retries Ollama calls once; never retries what will fail the same way again (overflow, timeout, server down)                                          |
| Knowing what a run cost                                            | Meters token usage per call, including calls inside structured output, with a peak-prompt figure to compare against the context window               |
| Testing code that calls an LLM                                     | A small `LlmClient` interface and a scripted fake in `@meirakrish/llm-kit/testing`                                                                   |

It also defines the conventions both tools share: one severity scale (`low` < `medium` < `high` < `critical`), `--fail-on` thresholds, and the LLM metadata block in JSON reports.

## Usage

```ts
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";
import { createLlmClient, resolveLlmConfig, Severity } from "@meirakrish/llm-kit";

const Review = z.object({
  riskLevel: Severity,
  summary: z.string(),
  findings: z.array(z.object({ resource: z.string(), issue: z.string() })),
});

const llm = await createLlmClient(resolveLlmConfig({ model: flags.model }), {
  overflowHint: "raise LLM_NUM_CTX or lower MAX_PLAN_CHARS",
});

const review = await llm.structured(Review, [new SystemMessage(SYSTEM), new HumanMessage(plan)], {
  name: "iac_review",
  limits: { summary: 700, findings: 20, "findings[].issue": 400 }, // applied for Ollama only
});

console.log(review.riskLevel, llm.takeUsage()); // { calls: 1, promptTokens: 5321, ... }
```

For an agent loop, `llm.chatWithTools(messages, tools)` runs one turn in which the model may call tools.

### Testing

```ts
import { scriptedLlm } from "@meirakrish/llm-kit/testing";

const llm = scriptedLlm({ structured: [{ riskLevel: "high", summary: "…", findings: [] }] });
await reviewPlan(plan, { llm });
expect(llm.calls[0]?.name).toBe("iac_review");
```

## Configuration

Flags passed to `resolveLlmConfig` win over environment variables, which win over defaults. Empty values count as unset.

| Variable                | Default                  | Notes                                                         |
| ----------------------- | ------------------------ | ------------------------------------------------------------- |
| `LLM_PROVIDER`          | detected                 | `ollama`, `anthropic` or `openai`                             |
| `LLM_MODEL`             | per provider             | `qwen3.5:9b`, `claude-sonnet-5-5`, `gpt-5-mini`               |
| `OLLAMA_HOST`           | `http://localhost:11434` | Ollama's own variable; `host:port` and `0.0.0.0` are handled  |
| `LLM_NUM_CTX`           | `16384`                  | Ollama context window                                         |
| `LLM_MAX_OUTPUT_TOKENS` | `4096`                   | Cap on generated tokens                                       |
| `LLM_TIMEOUT_MS`        | `180000`                 | Per request                                                   |
| `ANTHROPIC_API_KEY`     |                          | Selects Anthropic when no provider is set                     |
| `OPENAI_API_KEY`        |                          | Selects OpenAI when no provider is set (and no Anthropic key) |

Tool-specific settings belong to each tool, with its own prefix.

## Install

```bash
pnpm add @meirakrish/llm-kit @langchain/core zod
# only for hosted providers:
pnpm add @langchain/anthropic   # and/or @langchain/openai
```

`@langchain/core` and `zod` are peer dependencies, so the tool and llm-kit share one copy (messages and schemas cross the boundary).

## Development

```bash
pnpm install
pnpm typecheck && pnpm test && pnpm format:check
pnpm build
```

## License

MIT
