import { useCallback } from 'react';
import { classifyError, type RagError } from '../lib/errors';
import { reportError } from '../lib/globalErrorHandler';

export interface ErrorHandlerOptions {
  provider?: string;
  step?: string;
  onError?: (error: RagError) => void;
}

export interface UseErrorHandlerReturn {
  handleError: (error: unknown, step?: string) => RagError;
  clearError: () => void;
  lastError: RagError | null;
}

export function useErrorHandler(options: ErrorHandlerOptions = {}): UseErrorHandlerReturn {
  const { provider = 'app', step: defaultStep, onError } = options;
  let lastError: RagError | null = null;

  const handleError = useCallback(
    (error: unknown, step?: string): RagError => {
      const actualStep = step ?? defaultStep ?? 'component';
      const ragError = classifyError(error, provider, actualStep);
      lastError = ragError;

      // Report to global handler
      reportError(ragError);

      // Call optional callback
      onError?.(ragError);

      return ragError;
    },
    [provider, defaultStep, onError]
  );

  const clearError = useCallback(() => {
    lastError = null;
  }, []);

  return {
    handleError,
    clearError,
    get lastError() {
      return lastError;
    },
  };
}

// Convenience hook for async operations
export function useAsyncErrorHandler(options: ErrorHandlerOptions = {}) {
  const { handleError, clearError } = useErrorHandler(options);

  const wrapAsync = useCallback(
    <T,>(promise: Promise<T>, step?: string): Promise<T> => {
      return promise.catch((error) => {
        handleError(error, step);
        throw error;
      });
    },
    [handleError]
  );

  return {
    handleError,
    clearError,
    wrapAsync,
  };
}