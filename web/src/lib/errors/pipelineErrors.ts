import { RagError, RagErrorCode } from './base';

// ============================================================================
// Analysis Pipeline Errors
// ============================================================================

/** Which analyst tool-loop budget was exceeded: iteration count, wall-clock time, or cumulative tokens. */
export type AnalysisBudgetKind = 'iterations' | 'time' | 'tokens';

export interface AnalysisBudgetExceededContext {
  iterations: number;
  elapsedMs: number;
  tokensUsed: number;
  tripped: AnalysisBudgetKind;
  limit: number;
}

export class AnalysisBudgetExceededError extends RagError {
  constructor(context: AnalysisBudgetExceededContext, cause?: Error) {
    const budgetName =
      context.tripped === 'iterations' ? 'iteration' :
      context.tripped === 'time' ? 'time' : 'token';
    super({
      code: RagErrorCode.ANALYSIS_BUDGET_EXCEEDED,
      message:
        `Analysis budget exceeded: ${budgetName} budget (limit ${context.limit}) tripped — ` +
        `${context.iterations} iterations, ${context.elapsedMs}ms elapsed, ${context.tokensUsed} tokens used. ` +
        `Try a narrower question or increase the budget.`,
      cause,
      retryable: false,
      context: { ...context },
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