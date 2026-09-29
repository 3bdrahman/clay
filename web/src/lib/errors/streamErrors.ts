import { RagError, RagErrorCode } from './base';

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