import { describe, it, expect } from 'vitest';
import {
  RagError,
  RagErrorCode,
  NoProviderError,
  EmbeddingModelMissingError,
  ModelCatalogEmptyError,
  ModelNotFoundError,
  LocalServerUrlMissingError,
  ProviderUnreachableError,
  InvalidApiKeyError,
  RateLimitError,
  ProviderTimeoutError,
  StreamInterruptedError,
  TokenBudgetExceededError,
  GenerationFailedError,
  VectorStoreCorruptedError,
  VectorStoreQuotaExceededError,
  WebSearchProviderError,
  CodeExecutionError,
  isRetryable,
  getUserMessage,
} from './errors';

describe('RagError subclasses', () => {
  describe('NoProviderError', () => {
    it('creates error for NIM provider', () => {
      const err = new NoProviderError('nim');
      expect(err.code).toBe(RagErrorCode.NO_PROVIDER_CONFIGURED);
      expect(err.message).toContain('NVIDIA NIM API key');
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe('nim');
    });

    it('creates error for local provider', () => {
      const err = new NoProviderError('local');
      expect(err.code).toBe(RagErrorCode.NO_PROVIDER_CONFIGURED);
      expect(err.message).toContain('local server URL');
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe('local');
    });

    it('includes cause', () => {
      const cause = new Error('root cause');
      const err = new NoProviderError('nim', cause);
      expect(err.cause).toBe(cause);
    });
  });

  describe('EmbeddingModelMissingError', () => {
    it('creates error with correct code and message', () => {
      const err = new EmbeddingModelMissingError();
      expect(err.code).toBe(RagErrorCode.EMBEDDING_MODEL_MISSING);
      expect(err.message).toContain('embedding model');
      expect(err.retryable).toBe(false);
    });
  });

  describe('ModelCatalogEmptyError', () => {
    it('creates error for NIM provider', () => {
      const err = new ModelCatalogEmptyError('nim');
      expect(err.code).toBe(RagErrorCode.MODEL_CATALOG_EMPTY);
      expect(err.message).toContain('NVIDIA NIM model catalog');
      expect(err.retryable).toBe(true);
      expect(err.provider).toBe('nim');
    });

    it('creates error for local provider', () => {
      const err = new ModelCatalogEmptyError('local');
      expect(err.code).toBe(RagErrorCode.MODEL_CATALOG_EMPTY);
      expect(err.message).toContain('Local model catalog');
      expect(err.retryable).toBe(true);
      expect(err.provider).toBe('local');
    });
  });

  describe('ModelNotFoundError', () => {
    it('creates error with model ID and available models', () => {
      const available = ['model-a', 'model-b', 'model-c'];
      const err = new ModelNotFoundError('model-x', available);
      expect(err.code).toBe(RagErrorCode.MODEL_NOT_FOUND);
      expect(err.message).toContain('model-x');
      expect(err.message).toContain('model-a');
      expect(err.message).toContain('model-b');
      expect(err.retryable).toBe(false);
      expect(err.context?.modelId).toBe('model-x');
      expect(err.context?.availableCount).toBe(3);
    });

    it('handles empty available models', () => {
      const err = new ModelNotFoundError('model-x', []);
      expect(err.message).toContain('model-x');
      expect(err.context?.availableCount).toBe(0);
    });
  });

  describe('LocalServerUrlMissingError', () => {
    it('creates error with correct message', () => {
      const err = new LocalServerUrlMissingError();
      expect(err.code).toBe(RagErrorCode.LOCAL_SERVER_URL_MISSING);
      expect(err.message).toContain('Local server URL');
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe('local');
    });
  });

  describe('ProviderUnreachableError', () => {
    it('creates error for network failure', () => {
      const err = new ProviderUnreachableError('NVIDIA NIM');
      expect(err.code).toBe(RagErrorCode.PROVIDER_UNREACHABLE);
      expect(err.message).toContain('Cannot reach NVIDIA NIM');
      expect(err.retryable).toBe(true);
      expect(err.provider).toBe('NVIDIA NIM');
    });

    it('creates error for timeout', () => {
      const err = new ProviderUnreachableError('local', undefined, { isTimeout: true });
      expect(err.code).toBe(RagErrorCode.PROVIDER_UNREACHABLE);
      expect(err.message).toContain('timed out');
      expect(err.retryable).toBe(true);
      expect(err.context?.isTimeout).toBe(true);
    });

    it('sets retryable to false when specified', () => {
      const err = new ProviderUnreachableError('provider', undefined, { retryable: false });
      expect(err.retryable).toBe(false);
    });
  });

  describe('InvalidApiKeyError', () => {
    it('creates error for 401', () => {
      const err = new InvalidApiKeyError('NVIDIA NIM', 401);
      expect(err.code).toBe(RagErrorCode.INVALID_API_KEY);
      expect(err.message).toContain('Invalid API key');
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe('NVIDIA NIM');
      expect(err.context?.statusCode).toBe(401);
    });

    it('creates error for 403', () => {
      const err = new InvalidApiKeyError('local', 403);
      expect(err.code).toBe(RagErrorCode.INVALID_API_KEY);
      expect(err.message).toContain('API key rejected');
      expect(err.retryable).toBe(false);
      expect(err.context?.statusCode).toBe(403);
    });
  });

  describe('RateLimitError', () => {
    it('creates error with retryAfterMs', () => {
      const err = new RateLimitError('provider', 5000);
      expect(err.code).toBe(RagErrorCode.RATE_LIMIT_EXCEEDED);
      expect(err.message).toContain('rate limit exceeded');
      expect(err.message).toContain('5s');
      expect(err.retryable).toBe(true);
      expect(err.context?.retryAfterMs).toBe(5000);
    });

    it('creates error without retryAfterMs', () => {
      const err = new RateLimitError('provider');
      expect(err.message).toContain('rate limit exceeded');
      expect(err.context?.retryAfterMs).toBeUndefined();
    });
  });

  describe('ProviderTimeoutError', () => {
    it('creates error with timeoutMs', () => {
      const err = new ProviderTimeoutError('provider', 30000);
      expect(err.code).toBe(RagErrorCode.PROVIDER_TIMEOUT);
      expect(err.message).toContain('30000ms');
      expect(err.retryable).toBe(true);
      expect(err.context?.timeoutMs).toBe(30000);
    });
  });

  describe('StreamInterruptedError', () => {
    it('creates error with partial content length', () => {
      const err = new StreamInterruptedError('provider', 'partial response content');
      expect(err.code).toBe(RagErrorCode.STREAM_INTERRUPTED);
      expect(err.message).toContain('interrupted');
      expect(err.message).toContain('24 chars');
      expect(err.retryable).toBe(true);
      expect(err.provider).toBe('provider');
      expect(err.context?.partialLength).toBe(24);
    });

    it('marks wasAborted when cause is AbortError', () => {
      const abortError = new DOMException('Aborted', 'AbortError');
      const err = new StreamInterruptedError('provider', 'content', abortError);
      expect(err.context?.wasAborted).toBe(true);
    });
  });

  describe('TokenBudgetExceededError', () => {
    it('creates error with requested and budget', () => {
      const err = new TokenBudgetExceededError(10000, 8000);
      expect(err.code).toBe(RagErrorCode.TOKEN_BUDGET_EXCEEDED);
      expect(err.message).toContain('10000');
      expect(err.message).toContain('8000');
      expect(err.retryable).toBe(false);
      expect(err.context?.requested).toBe(10000);
      expect(err.context?.budget).toBe(8000);
    });
  });

  describe('GenerationFailedError', () => {
    it('creates error with provider', () => {
      const err = new GenerationFailedError('provider', new Error('gen failed'));
      expect(err.code).toBe(RagErrorCode.GENERATION_FAILED);
      expect(err.message).toContain('Failed to generate');
      expect(err.retryable).toBe(true);
      expect(err.provider).toBe('provider');
    });
  });

  describe('VectorStoreCorruptedError', () => {
    it('creates error with reason', () => {
      const err = new VectorStoreCorruptedError('dimension mismatch');
      expect(err.code).toBe(RagErrorCode.VECTOR_STORE_CORRUPTED);
      expect(err.message).toContain('Vector store corrupted');
      expect(err.message).toContain('dimension mismatch');
      expect(err.retryable).toBe(false);
      expect(err.context?.reason).toBe('dimension mismatch');
    });
  });

  describe('VectorStoreQuotaExceededError', () => {
    it('creates error with correct message', () => {
      const err = new VectorStoreQuotaExceededError();
      expect(err.code).toBe(RagErrorCode.VECTOR_STORE_QUOTA_EXCEEDED);
      expect(err.message).toContain('storage quota exceeded');
      expect(err.retryable).toBe(false);
    });
  });

  describe('WebSearchProviderError', () => {
    it('creates error for serper', () => {
      const err = new WebSearchProviderError('serper', 'Invalid API key');
      expect(err.code).toBe(RagErrorCode.WEB_SEARCH_PROVIDER_FAILED);
      expect(err.message).toContain('Serper (Google)');
      expect(err.message).toContain('Invalid API key');
      expect(err.retryable).toBe(false);
      expect(err.provider).toBe('serper');
    });

    it('creates error for duckduckgo with retryable true', () => {
      const err = new WebSearchProviderError('duckduckgo', 'Network error', undefined, { retryable: true });
      expect(err.retryable).toBe(true);
      expect(err.provider).toBe('duckduckgo');
    });
  });

  describe('CodeExecutionError', () => {
    it('creates error with original error and retryable flag', () => {
      const original = new Error('Syntax error');
      const err = new CodeExecutionError('Syntax error in generated code', original, {
        code: 'const x =',
        retryable: false,
      });
      expect(err.code).toBe(RagErrorCode.CODE_EXECUTION_ERROR);
      expect(err.message).toContain('Syntax error in generated code');
      expect(err.cause).toBe(original);
      expect(err.retryable).toBe(false);
      expect(err.context?.codeSnippet).toBe('const x =');
    });
  });

  describe('isRetryable', () => {
    it('returns true for retryable RagError', () => {
      const err = new ProviderUnreachableError('provider');
      expect(isRetryable(err)).toBe(true);
    });

    it('returns false for non-retryable RagError', () => {
      const err = new InvalidApiKeyError('provider', 401);
      expect(isRetryable(err)).toBe(false);
    });

    it('returns false for non-RagError', () => {
      expect(isRetryable(new Error('generic'))).toBe(false);
      expect(isRetryable(null)).toBe(false);
      expect(isRetryable(undefined)).toBe(false);
    });
  });

  describe('getUserMessage', () => {
    it('returns message from RagError', () => {
      const err = new InvalidApiKeyError('provider', 401);
      expect(getUserMessage(err)).toBe(err.message);
    });

    it('returns message from generic Error', () => {
      const err = new Error('generic error');
      expect(getUserMessage(err)).toBe('generic error');
    });

    it('returns string for non-Error', () => {
      expect(getUserMessage('string error')).toBe('string error');
      expect(getUserMessage(123)).toBe('123');
    });
  });

  describe('toUserMessage and toDebugObject', () => {
    it('toUserMessage returns the message', () => {
      const err = new ProviderUnreachableError('provider');
      expect(err.toUserMessage()).toBe(err.message);
    });

    it('toDebugObject includes all fields', () => {
      const err = new InvalidApiKeyError('provider', 401, new Error('cause'));
      const debug = err.toDebugObject();
      expect(debug.name).toBe('InvalidApiKeyError');
      expect(debug.code).toBe(RagErrorCode.INVALID_API_KEY);
      expect(debug.retryable).toBe(false);
      expect(debug.provider).toBe('provider');
      expect(debug.cause).toBe('cause');
      expect(debug.stack).toBeDefined();
    });
  });
});