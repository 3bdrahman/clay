import { RagError, RagErrorCode } from './base';

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
    const shortMessage = `Browser blocked request to ${provider} (CORS). ${provider} does not allow requests from this origin.`;
    const detailedMessage = customMessage
      ? customMessage
      : `Browser blocked the request to ${provider} due to CORS policy. ` +
        `${provider} does not allow requests from this origin. ` +
        `Solutions: (1) Switch to Local server in Settings (Ollama, LM Studio, etc.), or ` +
        `(2) Add ${provider} to the app's CSP connect-src by setting VITE_CSP_EXTRA_CONNECT_SRC at build time. ` +
        `See README for details.`;

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
// Web Search Errors
// ============================================================================

export class WebSearchProviderError extends RagError {
  constructor(
    provider: 'serper' | 'mwmbl',
    reason: string,
    cause?: Error,
    options: { retryable?: boolean } = {}
  ) {
    const providerName = provider === 'serper' ? 'Serper (Google)' : 'Mwmbl';
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
