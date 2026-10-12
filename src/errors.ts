/** The message of a thrown value, which may not be an Error. */
export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** A timeout or cancellation from an AbortSignal. */
export function isAbortError(err: unknown): boolean {
  const name = (err as { name?: string } | null)?.name;
  return name === "AbortError" || name === "TimeoutError";
}

/** The server could not be reached at all: retrying straight away will not help. */
export function isConnectionError(err: unknown): boolean {
  const e = err as { cause?: { code?: string }; message?: string } | null;
  const code = e?.cause?.code;
  return (
    code === "ECONNREFUSED" ||
    code === "ENOTFOUND" ||
    code === "EAI_AGAIN" ||
    /fetch failed|ECONNREFUSED|ENOTFOUND/.test(e?.message ?? "")
  );
}

/** A request that never produced a model answer (network, timeout). Distinct from a bad answer. */
export function isTransportError(err: unknown): boolean {
  return isAbortError(err) || isConnectionError(err) || Boolean((err as { cause?: { code?: string } })?.cause?.code);
}

/** Ollama rejected the prompt because it is longer than the context window (needs `truncate: false`). */
export function isContextOverflow(err: unknown): err is Error {
  return err instanceof Error && /exceeds the available context size/i.test(err.message);
}

/** Thrown when a prompt does not fit the model's context window. */
export class ContextOverflowError extends Error {
  override name = "ContextOverflowError";
  constructor(
    readonly promptTokens: number | undefined,
    readonly contextTokens: number | undefined,
    hint = "raise the context size or send less input",
  ) {
    const p = promptTokens ? ` (${promptTokens} tokens)` : "";
    const c = contextTokens ? ` (${contextTokens})` : "";
    super(`prompt${p} does not fit in the context window${c}; ${hint}`);
  }
}

/** Turn Ollama's JSON overflow blob into a ContextOverflowError; rethrow anything else unchanged. */
export function explainOverflow(err: unknown, hint?: string): never {
  if (!isContextOverflow(err)) throw err;
  const [, prompt, ctx] =
    /request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)/.exec(err.message) ?? [];
  throw new ContextOverflowError(prompt ? Number(prompt) : undefined, ctx ? Number(ctx) : undefined, hint);
}

/** One line for a report or log: what went wrong with the LLM, without stack traces or JSON blobs. */
export function describeLlmError(err: unknown): string {
  if (isAbortError(err)) return "request timed out";
  if (isConnectionError(err)) return "could not connect to the LLM server (is Ollama running?)";
  return (errorMessage(err).split("\n")[0] ?? "").slice(0, 200);
}
