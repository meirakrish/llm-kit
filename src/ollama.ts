import { ChatOllama } from "@langchain/ollama";

/**
 * By default Ollama silently drops the start of a prompt longer than numCtx, and the model answers
 * from what is left. `truncate: false` makes Ollama reject the request instead (see explainOverflow).
 * ChatOllama has no option for it, but passes the request object through to Ollama.
 */
export class StrictChatOllama extends ChatOllama {
  override invocationParams(options?: this["ParsedCallOptions"]) {
    return { ...super.invocationParams(options), truncate: false } as ReturnType<ChatOllama["invocationParams"]>;
  }
}

export interface OllamaStatus {
  url: string;
  model: string;
  reachable: boolean;
  modelAvailable: boolean;
  installedModels: string[];
  error?: string;
}

/** Ollama treats "name" and "name:latest" as the same model. */
function normalizeModelName(name: string): string {
  return name.includes(":") ? name : `${name}:latest`;
}

/**
 * Check that the Ollama server answers and that `model` is pulled, for a preflight before a long run.
 * Uses GET /api/tags, which is cheap and does not load the model into memory.
 */
export async function checkOllama(
  url: string,
  model: string,
  opts: { timeoutMs?: number; fetchImpl?: typeof fetch } = {},
): Promise<OllamaStatus> {
  const { timeoutMs = 3000, fetchImpl = fetch } = opts;
  const base: OllamaStatus = { url, model, reachable: false, modelAvailable: false, installedModels: [] };
  try {
    const res = await fetchImpl(`${url}/api/tags`, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { ...base, error: `HTTP ${res.status} from ${url}/api/tags` };
    const body = (await res.json()) as { models?: { name: string }[] };
    const installedModels = (body.models ?? []).map((m) => m.name);
    const wanted = normalizeModelName(model);
    const modelAvailable = installedModels.some((m) => normalizeModelName(m) === wanted);
    return {
      ...base,
      reachable: true,
      installedModels,
      modelAvailable,
      error:
        installedModels.length === 0
          ? "no models installed"
          : modelAvailable
            ? undefined
            : `model ${model} is not pulled (run: ollama pull ${model})`,
    };
  } catch (err) {
    const reason = err instanceof Error && err.name === "TimeoutError" ? "timed out" : String(err);
    return { ...base, error: `cannot reach Ollama at ${url} (${reason})` };
  }
}

/** Ask Ollama what a model supports (e.g. "completion", "tools", "thinking"). Empty if it can't be determined. */
export async function ollamaCapabilities(
  url: string,
  model: string,
  fetchImpl: typeof fetch = fetch,
): Promise<string[]> {
  try {
    const res = await fetchImpl(`${url.replace(/\/$/, "")}/api/show`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model }),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return [];
    const body = (await res.json()) as { capabilities?: unknown };
    return Array.isArray(body.capabilities) ? body.capabilities.filter((c): c is string => typeof c === "string") : [];
  } catch {
    // Server unreachable: the real call will report the real error.
    return [];
  }
}
