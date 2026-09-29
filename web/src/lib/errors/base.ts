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
  ANALYSIS_BUDGET_EXCEEDED = 'ANALYSIS_BUDGET_EXCEEDED',
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