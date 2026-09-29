/**
 * Retry logic for the workflow orchestrator — extracted from orchestrator.ts for modularity.
 * Handles exponential backoff retry with step trace recording.
 */

import type { StepTrace } from '../lib/types';
import { RagError, isRetryable, GenerationFailedError } from '../lib/errors';

export const MAX_STEP_RETRIES = 2;
export const BASE_RETRY_DELAY_MS = 1000;

export interface RetryContext {
  steps: StepTrace[];
  emitSteps: () => void;
}

export async function withRetry<T>(
  stepName: string,
  fn: () => Promise<T>,
  retryCtx: RetryContext,
  signal?: AbortSignal
): Promise<T> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= MAX_STEP_RETRIES; attempt++) {
    if (signal?.aborted) {
      throw new GenerationFailedError('orchestrator', new Error('Aborted'), { retryable: false });
    }
    try {
      return await fn();
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));

      // Don't retry on abort or non-retryable errors
      if (signal?.aborted || !isRetryable(lastError)) {
        if (lastError instanceof RagError) {
          throw lastError.withStep(stepName);
        }
        throw new GenerationFailedError('orchestrator', lastError, {
          retryable: false,
        }).withStep(stepName);
      }

      // Retry with exponential backoff
      if (attempt < MAX_STEP_RETRIES) {
        const delay = BASE_RETRY_DELAY_MS * Math.pow(2, attempt);
        // Record retry event in the currently-running step (last step with status 'running')
        for (let i = retryCtx.steps.length - 1; i >= 0; i--) {
          const step = retryCtx.steps[i];
          if (step.status === 'running') {
            if (!step.retries) step.retries = [];
            step.retries.push({ attempt: attempt + 1, error: lastError.message, delayMs: delay });
            retryCtx.emitSteps();
            break;
          }
        }
        await new Promise(r => setTimeout(r, delay));
        if (import.meta.env.DEV) {
          console.warn(`[orchestrator] Retrying ${stepName} (attempt ${attempt + 1}/${MAX_STEP_RETRIES}):`, lastError.message);
        }
      }
    }
  }
  if (lastError instanceof RagError) {
    throw lastError.withStep(stepName);
  }
  throw new GenerationFailedError('orchestrator', lastError!, { retryable: false }).withStep(stepName);
}