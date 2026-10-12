export {
  createLlmClient,
  type CallOptions,
  type LlmClient,
  type LlmClientOptions,
  type StructuredOptions,
} from "./client.js";
export {
  DEFAULT_MODELS,
  DEFAULT_OLLAMA_HOST,
  detectProvider,
  LLM_ENV,
  normalizeHost,
  Provider,
  resolveLlmConfig,
  type LlmConfig,
  type LlmConfigInput,
} from "./config.js";
export {
  ContextOverflowError,
  describeLlmError,
  errorMessage,
  explainOverflow,
  isAbortError,
  isConnectionError,
  isContextOverflow,
  isTransportError,
} from "./errors.js";
export { checkOllama, ollamaCapabilities, StrictChatOllama, type OllamaStatus } from "./ollama.js";
export { createChatModel, type ChatModelOptions } from "./provider.js";
export { describeLlmRun, llmDisabled, LlmRunInfo, ReportMeta } from "./report.js";
export { boundedJsonSchema, type SchemaLimits } from "./schema.js";
export {
  compareSeverity,
  FailOn,
  maxSeverity,
  meetsThreshold,
  parseFailOn,
  Severity,
  SEVERITY_ORDER,
} from "./severity.js";
export { trimToSentence, truncateMiddle } from "./text.js";
export { addUsage, emptyUsage, LlmUsage, UsageMeter } from "./usage.js";
