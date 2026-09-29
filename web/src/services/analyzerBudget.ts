/**
 * Analyzer budget — token/time/iteration limits and tracking.
 * Extracted from analyzerToolLoop.ts for modularity.
 */

export const MAX_TOOL_ITERATIONS = 8;
export const MAX_TOOL_LOOP_MS = 120_000;
export const DEFAULT_TOOL_LOOP_TOKENS = 100_000;
export const MAX_TOOL_RESULT_CHARS = 4000;
export const ARGS_SUMMARY_MAX_CHARS = 120;
export const MAX_CONSECUTIVE_MALFORMED = 2;
export const MAX_LLM_CALL_RETRIES = 3;
export const LLM_CALL_BASE_BACKOFF_MS = 1000;
export const SALVAGE_MAX_COMPLETION_TOKENS = 1024;
export const PER_CALL_COMPLETION_CAP = 2048;
export const MIN_PER_CALL_TOKENS = 64;

export interface BudgetContext {
  tokenBudget: number;
  tokensUsed: number;
  iterations: number;
  startTime: number;
}

export function deriveTokenBudget(maxToolLoopTokens: number | undefined): number {
  return maxToolLoopTokens ?? DEFAULT_TOOL_LOOP_TOKENS;
}

export function computePerCallMaxTokens(remainingBudget: number): number {
  return Math.min(PER_CALL_COMPLETION_CAP, Math.max(MIN_PER_CALL_TOKENS, remainingBudget));
}

export function checkTimeBudget(elapsedMs: number): { exceeded: boolean; limit: number } {
  return { exceeded: elapsedMs > MAX_TOOL_LOOP_MS, limit: MAX_TOOL_LOOP_MS };
}

export function checkTokenBudget(tokensUsed: number, tokenBudget: number): { exceeded: boolean; limit: number } {
  return { exceeded: tokensUsed > tokenBudget, limit: tokenBudget };
}

export function checkIterationBudget(iterations: number): { exceeded: boolean; limit: number } {
  return { exceeded: iterations >= MAX_TOOL_ITERATIONS, limit: MAX_TOOL_ITERATIONS };
}

export function buildBudgetExceededContext(
  ctx: BudgetContext,
  tripped: 'time' | 'tokens' | 'iterations'
): { iterations: number; elapsedMs: number; tokensUsed: number; tripped: 'time' | 'tokens' | 'iterations'; limit: number } {
  const elapsedMs = performance.now() - ctx.startTime;
  const limit = tripped === 'time' ? MAX_TOOL_LOOP_MS : tripped === 'tokens' ? ctx.tokenBudget : MAX_TOOL_ITERATIONS;
  return {
    iterations: ctx.iterations,
    elapsedMs,
    tokensUsed: ctx.tokensUsed,
    tripped,
    limit,
  };
}