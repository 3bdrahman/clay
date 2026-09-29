/**
 * Unified error hierarchy for the Clay RAG pipeline.
 * All errors extend RagError with typed code, cause, and retryable flag.
 * Enables programmatic handling and actionable user-facing messages.
 *
 * This file re-exports all error types and utilities from focused modules.
 */

// Base types and class
export {
  RagErrorCode,
  RagError,
} from './errors/base';
export type { RagErrorOptions } from './errors/base';

// Classification utilities
export {
  classifyError,
  classifyHttpError,
  isRetryable,
  getUserMessage,
  isLikelyCorsBlock,
} from './errors/classify';

// Configuration errors
export {
  NoProviderError,
  ModelCatalogEmptyError,
  ModelNotFoundError,
  LocalServerUrlMissingError,
} from './errors/configErrors';

// Provider/Network errors
export {
  ProviderUnreachableError,
  InvalidApiKeyError,
  RateLimitError,
  ProviderTimeoutError,
  CorsBlockedError,
  WebSearchProviderError,
} from './errors/providerErrors';

// Streaming/Generation errors
export {
  StreamInterruptedError,
  TokenBudgetExceededError,
  GenerationFailedError,
} from './errors/streamErrors';

// Analysis Pipeline errors
export {
  AnalysisBudgetExceededError,
  CodeExecutionError,
} from './errors/pipelineErrors';
export type {
  AnalysisBudgetKind,
  AnalysisBudgetExceededContext,
} from './errors/pipelineErrors';

// Vector Store errors
export {
  VectorStoreCorruptedError,
  VectorStoreQuotaExceededError,
} from './errors/storeErrors';

// Analysis Tool errors
export {
  AnalysisToolError,
} from './errors/analysisErrors';