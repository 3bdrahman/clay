/**
 * Step management for the workflow orchestrator — extracted from orchestrator.ts for modularity.
 * Handles step lifecycle, error recording, and callback emission.
 */

import type { StepTrace, WorkflowState } from '../lib/types';
import type { WorkflowCallbacks } from './orchestrator';
import { RagError, RagErrorCode, getUserMessage, isRetryable } from '../lib/errors';

export interface StepManager {
  steps: StepTrace[];
  beginStep: (node: string, label: string) => void;
  endStep: (node: string, opts?: { status?: 'done' | 'error'; detail?: string; meta?: Record<string, unknown> }) => void;
  setError: (err: Error, step: string) => void;
  endAllRunningSteps: (detail: string) => void;
  emitSteps: () => void;
}

export function createStepManager(
  state: WorkflowState,
  callbacks: WorkflowCallbacks
): StepManager {
  const steps: StepTrace[] = [];
  let emitScheduled = false;

  function beginStep(node: string, label: string): void {
    const step: StepTrace = {
      id: `${node}-${crypto.randomUUID()}`,
      node,
      label,
      status: 'running',
      startedAt: Date.now(),
    };
    steps.push(step);
    emitSteps();
  }

  function endStep(node: string, opts: { status?: 'done' | 'error'; detail?: string; meta?: Record<string, unknown> } = {}): void {
    for (let i = steps.length - 1; i >= 0; i--) {
      const step = steps[i];
      if (step && step.node === node && step.status === 'running') {
        step.status = opts.status ?? 'done';
        step.finishedAt = Date.now();
        step.durationMs = step.finishedAt - (step.startedAt || step.finishedAt);
        if (opts.detail) step.detail = opts.detail;
        if (opts.meta) step.meta = opts.meta;
        emitSteps();
        return;
      }
    }
  }

  function setError(err: Error, step: string): void {
    const message = getUserMessage(err);
    const code = err instanceof RagError ? err.code : RagErrorCode.UNKNOWN_ERROR;
    const retryable = isRetryable(err);

    state.error = {
      code,
      message,
      step,
      retryable,
    };

    callbacks.onError?.(err);
    emitSteps();
  }

  function endAllRunningSteps(detail: string): void {
    const now = Date.now();
    for (const step of steps) {
      if (step.status === 'running') {
        step.status = 'skipped';
        step.finishedAt = now;
        step.durationMs = now - (step.startedAt || now);
        step.detail = detail;
      }
    }
    emitSteps();
  }

  function emitSteps(): void {
    state.steps = [...steps];
    callbacks.onStepUpdate?.(state.steps);
    if (!emitScheduled) {
      emitScheduled = true;
      requestAnimationFrame(() => {
        emitScheduled = false;
        callbacks.onPartialUpdate?.({ ...state });
      });
    }
  }

  return {
    steps,
    beginStep,
    endStep,
    setError,
    endAllRunningSteps,
    emitSteps,
  };
}