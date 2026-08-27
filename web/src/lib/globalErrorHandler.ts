/**
 * Global error handling setup for unhandled promise rejections and errors.
 * Call `initGlobalErrorHandler()` once at app startup.
 */

import { classifyError, RagError } from './errors';

let isInitialized = false;

export function initGlobalErrorHandler(): void {
  if (isInitialized) return;
  isInitialized = true;

  // Handle unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    const error = event.reason instanceof Error
      ? event.reason
      : new Error(String(event.reason));

    const ragError = classifyError(error, 'global', 'unhandled-rejection');
    logError(ragError);

    // Prevent default browser behavior (logging to console)
    event.preventDefault();
  });

  // Handle uncaught errors
  window.addEventListener('error', (event) => {
    const error = event.error instanceof Error
      ? event.error
      : new Error(event.message);

    const ragError = classifyError(error, 'global', 'uncaught-error');
    logError(ragError);
  });

  // Handle resource loading errors (scripts, styles, images)
  window.addEventListener('error', (event) => {
    if (event.target !== window && event.target instanceof HTMLScriptElement) {
      const error = new Error(`Failed to load script: ${event.target.src}`);
      const ragError = classifyError(error, 'global', 'resource-load');
      logError(ragError);
    }
  }, true);
}

function logError(error: RagError): void {
  console.error('[GlobalErrorHandler]', {
    code: error.code,
    message: error.message,
    provider: error.provider,
    step: error.step,
    retryable: error.retryable,
    context: error.context,
    stack: error.stack,
  });
}

export function reportError(
  error: unknown,
  provider: string = 'app',
  step: string = 'manual-report'
): RagError {
  const ragError = error instanceof RagError
    ? error
    : classifyError(error, provider, step);

  logError(ragError);
  return ragError;
}

export function isRetryableError(error: unknown): boolean {
  if (error instanceof RagError) return error.retryable;
  return false;
}