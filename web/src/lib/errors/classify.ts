import { RagError, RagErrorCode } from './base';
import {
  ProviderUnreachableError,
  InvalidApiKeyError,
  RateLimitError,
  CorsBlockedError,
} from './providerErrors';
import { StreamInterruptedError } from './streamErrors';
import { ModelNotFoundError } from './configErrors';

/**
 * Checks if the error is likely a CORS block for a given provider.
 * CORS failures manifest as TypeError: Failed to fetch with no response.
 */
export function isLikelyCorsBlock(error: unknown, provider: string): boolean {
  if (!(error instanceof TypeError && error.message.includes('fetch'))) return false;
  if (import.meta.env.DEV) return false; // Dev uses Vite proxy, CORS is bypassed

  const corsAwareProviders = ['OpenRouter', 'NVIDIA NIM'];
  if (!corsAwareProviders.includes(provider)) return false;
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
