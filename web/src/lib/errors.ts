/**
 * Unified error hierarchy for the Clay RAG pipeline.
 * All errors extend RagError with typed code, cause, and retryable flag.
 * Enables programmatic handling and actionable user-facing messages.
 */

// ============================================================================
// Error Codes (programmatic, stable)
// ============================================================================

export enum RagErrorCode {
  // Configuration errors
  NO_PROVIDER_CONFIGURED = 'NO_PROVIDER_CONFIGURED',
  EMBEDDING_MODEL_MISSING = 'EMBEDDING_MODEL_MISSING',
  MODEL_CATALOG_EMPTY = 'MODEL_CATALOG_EMPTY',
  MODEL_NOT_FOUND = 'MODEL_NOT_FOUND',
  LOCAL_SERVER_URL_MISSING = 'LOCAL_SERVER_URL_MISSING',

  // Provider/Network errors
  PROVIDER_UNREACHABLE = 'PROVIDER_UNREACHABLE',
  INVALID_API_KEY = 'INVALID_API_KEY',
  RATE_LIMIT_EXCEEDED = 'RATE_LIMIT_EXCEEDED',
  PROVIDER_TIMEOUT = 'PROVIDER_TIMEOUT',
  CORS_BLOCKED = 'CORS_BLOCKED',

  // Streaming/Generation errors
  STREAM_INTERRUPTED = 'STREAM_INTERRUPTED',
  TOKEN_BUDGET_EXCEEDED = 'TOKEN_BUDGET_EXCEEDED',
  GENERATION_FAILED = 'GENERATION_FAILED',

  // Vector store errors
  VECTOR_STORE_CORRUPTED = 'VECTOR_STORE_CORRUPTED',
  VECTOR_STORE_QUOTA_EXCEEDED = 'VECTOR_STORE_QUOTA_EXCEEDED',

  // Web search errors
  WEB_SEARCH_PROVIDER_FAILED = 'WEB_SEARCH_PROVIDER_FAILED',

  // Code execution errors
  CODE_EXECUTION_ERROR = 'CODE_EXECUTION_ERROR',

  // Generic/fallback
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
}

// ============================================================================
// Base Error Class
// ============================================================================

export interface RagErrorOptions {
  code: RagErrorCode;
  message: string;
  cause?: Error;
  retryable?: boolean;
  provider?: string;
  step?: string;
  context?: Record<string, unknown>;
}

export class RagError extends Error {
  public readonly code: RagErrorCode;
  public readonly retryable: boolean;
  public readonly provider?: string;
  public readonly step?: string;
  public readonly context?: Record<string, unknown>;

  constructor(options: RagErrorOptions) {
    super(options.message);
    this.name = this.constructor.name;
    this.code = options.code;
    this.cause = options.cause;
    this.retryable = options.retryable ?? false;
    this.provider = options.provider;
    this.step = options.step;
    this.context = options.context;

    // Maintains proper stack trace in V8 environments
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Returns a sanitized user-facing message without secrets.
   */
  toUserMessage(): string {
    return this.message;
  }

  /**
   * Returns a new RagError with the given step context, preserving the original
   * error's message, code, and cause. Use this when re-throwing an error
   * through orchestration layers that need to record which step produced it.
   */
  withStep(step: string): RagError {
    const next = new RagError({
      code: this.code,
      message: this.message,
      cause: this.cause instanceof Error ? this.cause : undefined,
      retryable: this.retryable,
      provider: this.provider,
      step,
      context: this.context,
    });
    return next;
  }

  /**
   * Returns a debug representation with full context.
   */
  toDebugObject(): Record<string, unknown> {
    const cause = this.cause;
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      retryable: this.retryable,
      provider: this.provider,
      step: this.step,
      context: this.context,
      stack: this.stack,
      cause: cause instanceof Error ? cause.message : cause,
    };
  }
}

// ============================================================================
// Configuration Errors
// ============================================================================

export class NoProviderError extends RagError {
  constructor(provider: string, cause?: Error) {
    const message =
      provider === 'nim'
        ? 'No NVIDIA NIM API key configured. Add your API key in Settings.'
        : provider === 'local' || provider === 'ollama'
        ? 'No local server URL configured. Set the server URL in Settings.'
        : `No ${provider} API key configured. Add your API key in Settings.`;

    super({
      code: RagErrorCode.NO_PROVIDER_CONFIGURED,
      message,
      cause,
      retryable: false,
      provider,
      context: { provider },
    });
  }
}

export class EmbeddingModelMissingError extends RagError {
  constructor(cause?: Error) {
    super({
      code: RagErrorCode.EMBEDDING_MODEL_MISSING,
      message: 'No embedding model selected. Choose an embedding model in Settings.',
      cause,
      retryable: false,
      context: {},
    });
  }
}

export class ModelCatalogEmptyError extends RagError {
  constructor(provider: string, cause?: Error) {
    const message = `${provider} model catalog is empty. Check your API key and click Refresh in Settings.`;

    super({
      code: RagErrorCode.MODEL_CATALOG_EMPTY,
      message,
      cause,
      retryable: true,
      provider,
      context: { provider },
    });
  }
}

export class ModelNotFoundError extends RagError {
  constructor(modelId: string, availableModels: string[], cause?: Error) {
    const suggestions = availableModels.slice(0, 5).join(', ');
    const hint = availableModels.length > 0
      ? ` Available models: ${suggestions}${availableModels.length > 5 ? '...' : ''}`
      : '';

    super({
      code: RagErrorCode.MODEL_NOT_FOUND,
      message: `Model "${modelId}" not found in the catalog.${hint} Re-pick or refresh the catalog.`,
      cause,
      retryable: false,
      context: { modelId, availableCount: availableModels.length },
    });
  }
}

export class LocalServerUrlMissingError extends RagError {
  constructor(cause?: Error) {
    super({
      code: RagErrorCode.LOCAL_SERVER_URL_MISSING,
      message: 'Local server URL is required. Set the OpenAI-compatible server URL in Settings (e.g., http://localhost:11434/v1).',
      cause,
      retryable: false,
      provider: 'local',
      context: {},
    });
  }
}

// ============================================================================
// Provider/Network Errors
// ============================================================================

export class ProviderUnreachableError extends RagError {
  constructor(
    provider: string,
    cause?: Error,
    options: { isTimeout?: boolean; retryable?: boolean; message?: string } = {}
  ) {
    const message = options.message
      ? options.message
      : options.isTimeout
      ? `Request to ${provider} timed out. Check network connectivity and server status.`
      : `Cannot reach ${provider}. Check network connectivity, CORS settings, or server availability.`;

    super({
      code: RagErrorCode.PROVIDER_UNREACHABLE,
      message,
      cause,
      retryable: options.retryable ?? true,
      provider,
      context: { isTimeout: options.isTimeout },
    });
  }
}

export class InvalidApiKeyError extends RagError {
  constructor(provider: string, statusCode: 401 | 403, cause?: Error) {
    const isForbidden = statusCode === 403;
    const message = isForbidden
      ? `API key rejected by ${provider} (403 Forbidden). The key may lack required permissions.`
      : `Invalid API key for ${provider} (401 Unauthorized). Check your API key in Settings.`;

    super({
      code: RagErrorCode.INVALID_API_KEY,
      message,
      cause,
      retryable: false,
      provider,
      context: { statusCode },
    });
  }
}

export class RateLimitError extends RagError {
  constructor(
    provider: string,
    retryAfterMs?: number,
    cause?: Error
  ) {
    const retryHint = retryAfterMs
      ? ` Retry after ${Math.ceil(retryAfterMs / 1000)}s.`
      : ' Rate limit exceeded.';

    super({
      code: RagErrorCode.RATE_LIMIT_EXCEEDED,
      message: `${provider} rate limit exceeded.${retryHint} Reduce request frequency or upgrade your tier.`,
      cause,
      retryable: true,
      provider,
      context: { retryAfterMs },
    });
  }
}

export class ProviderTimeoutError extends RagError {
  constructor(provider: string, timeoutMs: number, cause?: Error) {
    super({
      code: RagErrorCode.PROVIDER_TIMEOUT,
      message: `${provider} request timed out after ${timeoutMs}ms. The server may be overloaded or the request too complex.`,
      cause,
      retryable: true,
      provider,
      context: { timeoutMs },
    });
  }
}

export class CorsBlockedError extends RagError {
  constructor(provider: string, cause?: Error, customMessage?: string) {
    const shortMessage = `Browser blocked request to ${provider} (CORS). ${provider} only allows requests from build.nvidia.com.`;
    const detailedMessage = customMessage
      ? customMessage
      : `Browser blocked the request to ${provider} due to CORS policy. ` +
        `${provider} does not allow requests from this origin. ` +
        `Solutions: (1) Switch to Local server in Settings (Ollama, LM Studio, etc.), or ` +
        `(2) Deploy an edge proxy (Cloudflare Worker, Vercel function, Netlify function) ` +
        `and set VITE_NIM_BASE_URL at build time. See README for details.`;

    super({
      code: RagErrorCode.CORS_BLOCKED,
      message: detailedMessage,
      cause,
      retryable: false,
      provider,
      context: { blockedBy: 'browser-cors', shortMessage },
    });
  }

  toUserMessage(): string {
    return (this.context?.shortMessage as string) ?? this.message;
  }
}

// ============================================================================
// Streaming/Generation Errors
// ============================================================================

export class StreamInterruptedError extends RagError {
  constructor(
    provider: string,
    partialContent: string,
    cause?: Error
  ) {
    super({
      code: RagErrorCode.STREAM_INTERRUPTED,
      message: `AI response from ${provider} was interrupted. Partial response received (${partialContent.length} chars).`,
      cause,
      retryable: true,
      provider,
      context: { partialLength: partialContent.length, wasAborted: cause?.name === 'AbortError' },
    });
  }
}

export class TokenBudgetExceededError extends RagError {
  constructor(requested: number, budget: number, cause?: Error) {
    super({
      code: RagErrorCode.TOKEN_BUDGET_EXCEEDED,
      message: `Token budget exceeded: requested ${requested}, budget ${budget}. Reduce context or use a model with larger context window.`,
      cause,
      retryable: false,
      context: { requested, budget },
    });
  }
}

export class GenerationFailedError extends RagError {
  constructor(provider: string, cause?: Error, options: { retryable?: boolean } = {}) {
    super({
      code: RagErrorCode.GENERATION_FAILED,
      message: `Failed to generate response from ${provider}. The model may be overloaded or the input invalid.`,
      cause,
      retryable: options.retryable ?? true,
      provider,
      context: {},
    });
  }
}

// ============================================================================
// Vector Store Errors
// ============================================================================

export class VectorStoreCorruptedError extends RagError {
  constructor(reason: string, cause?: Error) {
    super({
      code: RagErrorCode.VECTOR_STORE_CORRUPTED,
      message: `Vector store corrupted: ${reason}. Try clearing data and re-indexing your documents.`,
      cause,
      retryable: false,
      context: { reason },
    });
  }
}

export class VectorStoreQuotaExceededError extends RagError {
  constructor(cause?: Error) {
    super({
      code: RagErrorCode.VECTOR_STORE_QUOTA_EXCEEDED,
      message: 'IndexedDB storage quota exceeded. Clear old documents or use a browser with higher storage limits.',
      cause,
      retryable: false,
      context: {},
    });
  }
}

// ============================================================================
// Web Search Errors
// ============================================================================

export class WebSearchProviderError extends RagError {
  constructor(
    provider: 'serper' | 'duckduckgo',
    reason: string,
    cause?: Error,
    options: { retryable?: boolean } = {}
  ) {
    const providerName = provider === 'serper' ? 'Serper (Google)' : 'DuckDuckGo';
    const message = `${providerName} search failed: ${reason}.`;

    super({
      code: RagErrorCode.WEB_SEARCH_PROVIDER_FAILED,
      message,
      cause,
      retryable: options.retryable ?? false,
      provider,
      context: { provider, reason },
    });
  }
}

// ============================================================================
// Code Execution Errors
// ============================================================================

export class CodeExecutionError extends RagError {
  constructor(
    reason: string,
    originalError: Error,
    options: { code?: string; retryable?: boolean } = {}
  ) {
    const message = `Code execution failed: ${reason}`;

    super({
      code: RagErrorCode.CODE_EXECUTION_ERROR,
      message,
      cause: originalError,
      retryable: options.retryable ?? false,
      context: { reason, codeSnippet: options.code?.slice(0, 200) },
    });
  }
}

// ============================================================================
// Error Classification Utilities
// ============================================================================

/**
 * Checks if the current origin is build.nvidia.com (the only origin NIM allows CORS from).
 */
function isBuildNvidiaOrigin(): boolean {
  try {
    if (typeof window === 'undefined') return false;
    return window.location.hostname === 'build.nvidia.com';
  } catch {
    return false;
  }
}

/**
 * Checks if the error is likely a CORS block for a given provider.
 * CORS failures manifest as TypeError: Failed to fetch with no response.
 */
function isLikelyCorsBlock(error: unknown, provider: string): boolean {
  if (!(error instanceof TypeError && error.message.includes('fetch'))) return false;
  if (import.meta.env.DEV) return false; // Dev uses Vite proxy, CORS is bypassed

  const nimProviders = ['NVIDIA NIM', 'OpenRouter', 'Groq', 'Together AI'];
  if (!nimProviders.includes(provider)) return false;
  if (provider === 'NVIDIA NIM' && isBuildNvidiaOrigin()) return false;
  return true;
}

/**
 * Classifies a generic error into a typed RagError.
 * Used as a safety net when calling external APIs.
 */
export function classifyError(
  error: unknown,
  provider: string,
  step: string
): RagError {
  if (error instanceof RagError) return error;

  if (error instanceof Response) {
    // Fetch Response object (rare, but handle it)
    return classifyHttpError(error, provider, step);
  }

  if (error instanceof Error) {
    // Network/Abort errors
    if (error.name === 'AbortError' || error.name === 'TimeoutError') {
      return new ProviderUnreachableError(provider, error, {
        isTimeout: error.name === 'TimeoutError',
      });
    }

    // TypeError usually means network failure in fetch
    if (error instanceof TypeError && error.message.includes('fetch')) {
      // Check if this is likely a CORS block
      if (isLikelyCorsBlock(error, provider)) {
        return new CorsBlockedError(provider, error);
      }
      return new ProviderUnreachableError(provider, error);
    }

    // DOMException for abort
    if (error.name === 'AbortError') {
      return new StreamInterruptedError(provider, '', error);
    }
  }

  // Fallback
  return new RagError({
    code: RagErrorCode.UNKNOWN_ERROR,
    message: `Unexpected error in ${step}: ${error instanceof Error ? error.message : String(error)}`,
    cause: error instanceof Error ? error : undefined,
    retryable: false,
    provider,
    step,
  });
}

/**
 * Classifies HTTP response errors from fetch.
 */
export function classifyHttpError(
  response: Response,
  provider: string,
  step: string,
  modelHint?: string
): RagError {
  const status = response.status;

  if (status === 401 || status === 403) {
    return new InvalidApiKeyError(provider, status as 401 | 403);
  }

  if (status === 429) {
    const retryAfter = response.headers.get('retry-after');
    const retryAfterMs = retryAfter ? parseInt(retryAfter, 10) * 1000 : undefined;
    return new RateLimitError(provider, retryAfterMs);
  }

  if (status >= 500) {
    return new ProviderUnreachableError(provider, new Error(`${status} ${response.statusText}`), {
      retryable: true,
    });
  }

  if (status === 404) {
    return new ModelNotFoundError(modelHint ?? '(unspecified)', [], new Error(`${status} ${response.statusText}`));
  }

  return new RagError({
    code: RagErrorCode.UNKNOWN_ERROR,
    message: `${provider} returned ${status} ${response.statusText} during ${step}`,
    cause: new Error(`${status} ${response.statusText}`),
    retryable: false,
    provider,
    step,
  });
}

/**
 * Checks if an error is retryable.
 */
export function isRetryable(error: unknown): boolean {
  if (error instanceof RagError) return error.retryable;
  return false;
}

/**
 * Extracts user-facing message from any error.
 */
export function getUserMessage(error: unknown): string {
  if (error instanceof RagError) return error.toUserMessage();
  if (error instanceof Error) return error.message;
  return String(error);
}

export { isLikelyCorsBlock };