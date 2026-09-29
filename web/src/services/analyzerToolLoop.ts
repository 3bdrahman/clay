/**
 * Analyzer tool loop — entry point for the agentic analysis tool loop.
 * Re-exports public API from budget, executor, and salvage modules.
 */

import type { LLMMessage, ChartConfig } from '../lib/types';
import { TOOL_SCHEMAS } from '../lib/analysisTools';
import { CodeExecutionError, AnalysisBudgetExceededError, GenerationFailedError, RateLimitError } from '../lib/errors';
import { buildSystemPrompt, buildInitialUserMessage } from './analyzerPrompts';

// Re-export types
export type {
  ToolTraceEntry,
  LoopResult,
  AnalyzerToolLoopDeps,
  AnalyzerToolLoopOptions,
} from './analyzerToolLoopTypes';

// Re-export budget constants and functions
export {
  MAX_TOOL_ITERATIONS,
  MAX_TOOL_LOOP_MS,
  DEFAULT_TOOL_LOOP_TOKENS,
  MAX_TOOL_RESULT_CHARS,
  ARGS_SUMMARY_MAX_CHARS,
  MAX_CONSECUTIVE_MALFORMED,
  MAX_LLM_CALL_RETRIES,
  LLM_CALL_BASE_BACKOFF_MS,
  SALVAGE_MAX_COMPLETION_TOKENS,
  PER_CALL_COMPLETION_CAP,
  MIN_PER_CALL_TOKENS,
  deriveTokenBudget,
  computePerCallMaxTokens,
  checkTimeBudget,
  checkTokenBudget,
  checkIterationBudget,
  buildBudgetExceededContext,
  type BudgetContext,
} from './analyzerBudget';

// Re-export tool executor functions
export {
  buildToolTrace,
  recordToolTrace,
  summarizeArgs,
  truncateResult,
  executeToolCallWithTracing,
  validateToolCallArguments,
  type ToolExecutionResult,
  type ExecuteToolCallDeps,
} from './analyzerToolExecutor';

// Re-export salvage functions and types
export {
  FallbackTriggered,
  parseSynthesisJson,
  normalizeInsightConfidence,
  isValidChartConfig,
  runSalvageSynthesis,
  type SalvageDeps,
  type SalvageOptions,
} from './analyzerSalvage';

import {
  MAX_TOOL_ITERATIONS,
  MAX_CONSECUTIVE_MALFORMED,
  MAX_LLM_CALL_RETRIES,
  LLM_CALL_BASE_BACKOFF_MS,
  deriveTokenBudget,
  computePerCallMaxTokens,
  checkTimeBudget,
  checkTokenBudget,
  buildBudgetExceededContext,
  type BudgetContext,
} from './analyzerBudget';

import {
  buildToolTrace,
  executeToolCallWithTracing,
  validateToolCallArguments,
  truncateResult,
  type ExecuteToolCallDeps,
} from './analyzerToolExecutor';

import {
  FallbackTriggered,
  runSalvageSynthesis,
  parseSynthesisJson,
  normalizeInsightConfidence,
  isValidChartConfig,
} from './analyzerSalvage';

import type { LoopResult, AnalyzerToolLoopDeps, AnalyzerToolLoopOptions } from './analyzerToolLoopTypes';

export async function runToolLoop(
  question: string,
  relevant: string[],
  deps: AnalyzerToolLoopDeps,
  options: AnalyzerToolLoopOptions = {}
): Promise<LoopResult> {
  const { signal, hooks, previousContext } = options;
  const start = performance.now();
  const tokenBudget = deriveTokenBudget(deps.maxToolLoopTokens);

  const messages: LLMMessage[] = [
    { role: 'user', content: buildInitialUserMessage({ question, relevant, metadata: deps.metadata, previousContext }) },
  ];

  const toolTraceMap = new Map<string, { calls: number; durationMs: number; tokensUsed: number }>();
  let tokensUsed = 0;
  let iterations = 0;
  let consecutiveMalformed = 0;

  const budgetCtx: BudgetContext = {
    tokenBudget,
    tokensUsed: 0,
    iterations: 0,
    startTime: start,
  };

  async function runSalvage(fallbackReason: string): Promise<LoopResult> {
    return runSalvageSynthesis(messages, { llm: deps.llm, codeGenModel: deps.codeGenModel }, { signal, hooks }, fallbackReason, toolTraceMap, iterations, tokensUsed);
  }

  try {
    while (iterations < MAX_TOOL_ITERATIONS) {
      // Check abort before each invoke
      if (signal?.aborted) {
        throw new CodeExecutionError('Analysis aborted', new Error('AbortSignal triggered'), {
          code: '',
          retryable: false,
        });
      }

      let resp: Awaited<ReturnType<typeof deps.llm.invoke>>;
      let llmCallAttempt = 0;
      while (true) {
        try {
          const remainingBudget = tokenBudget - tokensUsed;
          const maxTokens = computePerCallMaxTokens(remainingBudget);
          resp = await deps.llm.invoke({
            system: buildSystemPrompt(),
            messages,
            tools: TOOL_SCHEMAS,
            toolChoice: 'auto',
            temperature: 0,
            model: deps.codeGenModel,
            maxTokens,
          }, signal);
          break;
        } catch (e) {
          if (iterations === 0 && e instanceof GenerationFailedError) {
            throw new FallbackTriggered('provider-rejected-tools');
          }
          if (e instanceof RateLimitError && llmCallAttempt < MAX_LLM_CALL_RETRIES) {
            llmCallAttempt++;
            const delay = LLM_CALL_BASE_BACKOFF_MS * Math.pow(2, llmCallAttempt - 1);
            if (import.meta.env.DEV) {
              console.warn(`[analyzer] Rate limited, retrying LLM call (attempt ${llmCallAttempt}/${MAX_LLM_CALL_RETRIES}) after ${delay}ms`);
            }
            await new Promise(r => setTimeout(r, delay));
            if (signal?.aborted) {
              throw new CodeExecutionError('Analysis aborted', new Error('AbortSignal triggered'), {
                code: '',
                retryable: false,
              });
            }
            continue;
          }
          throw e;
        }
      }

      // Check abort after invoke
      if (signal?.aborted) {
        throw new CodeExecutionError('Analysis aborted', new Error('AbortSignal triggered'), {
          code: '',
          retryable: false,
        });
      }

      iterations++;
      budgetCtx.iterations = iterations;
      const iterationTokens = resp.usage?.totalTokens ?? 0;
      tokensUsed += iterationTokens;
      budgetCtx.tokensUsed = tokensUsed;

      // Budget checks
      const elapsedMs = performance.now() - start;
      const timeCheck = checkTimeBudget(elapsedMs);
      if (timeCheck.exceeded) {
        throw new AnalysisBudgetExceededError(buildBudgetExceededContext(budgetCtx, 'time'));
      }
      const tokenCheck = checkTokenBudget(tokensUsed, tokenBudget);
      if (tokenCheck.exceeded) {
        throw new AnalysisBudgetExceededError(buildBudgetExceededContext(budgetCtx, 'tokens'));
      }

      const reflection = (resp.content || '').trim();
      if (reflection) {
        try {
          hooks?.onIteration?.({
            iteration: iterations,
            reflection,
            tokensUsed: iterationTokens,
          });
        } catch (hookErr) {
          if (import.meta.env.DEV) console.warn('[analyzer] onIteration hook threw:', hookErr);
        }
      }

      const toolCalls = resp.toolCalls?.map(c => ({
        ...c,
        // Some models emit tool calls without an id; the OpenAI-compatible
        // protocol requires a non-empty tool_call_id on the assistant message
        // and the matching tool message, and providers 400 without it.
        // Synthesize one so the linkage stays valid.
        id: typeof c.id === 'string' && c.id ? c.id : crypto.randomUUID(),
      }));
      const finishReason = resp.finishReason;

      if (toolCalls && toolCalls.length > 0) {
        // Assistant message with tool calls
        messages.push({
          role: 'assistant',
          content: resp.content || '',
          toolCalls,
        });

        const tokensPerCall = Math.round(iterationTokens / toolCalls.length);

        const executeDeps: ExecuteToolCallDeps = {
          datasets: deps.datasets,
          metadata: deps.metadata,
          executeUserCode: deps.executeUserCode,
        };

        for (const call of toolCalls) {
          const validation = validateToolCallArguments(call);
          if (validation.error) {
            consecutiveMalformed++;
            if (import.meta.env.DEV) console.warn('[analyzer] Malformed tool call:', validation.error);
            // Tool result message for malformed call
            messages.push({
              role: 'tool',
              content: truncateResult(JSON.stringify({ error: validation.error })),
              toolCallId: call.id,
            });
          } else {
            // Valid call - reset malformed counter
            consecutiveMalformed = 0;

            const result = await executeToolCallWithTracing(
              executeDeps,
              validation.toolName,
              validation.args,
              tokensPerCall,
              toolTraceMap,
              hooks
            );

            // Tool result message
            messages.push({
              role: 'tool',
              content: truncateResult(JSON.stringify(result.toolResult)),
              toolCallId: call.id,
            });
          }

          // Check consecutive malformed limit - Trigger (c)
          if (consecutiveMalformed >= MAX_CONSECUTIVE_MALFORMED) {
            throw new FallbackTriggered('malformed-tool-calls');
          }
        }

        continue; // Next iteration
      }

      // No tool calls - check finish reason
      // Trigger (b): first iteration, no tool calls, finishReason 'stop'
      if (iterations === 1 && finishReason === 'stop') {
        throw new FallbackTriggered('no-first-tool-call');
      }

      if (finishReason === 'stop' || !toolCalls || toolCalls.length === 0) {
        const synthesis = parseSynthesisJson(resp.content || '');
        // FIX 2a: Normalize insight confidence to 'high' | 'medium' | 'low' enum
        const insights = normalizeInsightConfidence(synthesis.insights);

        // FIX 2b: Validate synthesis chart shape before trusting it
        let chart: ChartConfig | undefined = synthesis.chart;
        if (chart && !isValidChartConfig(chart)) {
          chart = undefined;
        }

        return {
          answer: synthesis.answer,
          insights,
          chart,
          tokensUsed,
          iterations,
          toolTrace: buildToolTrace(toolTraceMap),
        };
      }

      // Other finish reasons (length, content_filter, etc.) - treat as stop
      return {
        answer: resp.content || '',
        insights: [],
        chart: undefined,
        tokensUsed,
        iterations,
        toolTrace: buildToolTrace(toolTraceMap),
      };
    }

    // Should not reach here due to budget checks, but safety net
    throw new AnalysisBudgetExceededError(buildBudgetExceededContext(budgetCtx, 'iterations'));
  } catch (e) {
    // Salvage synthesis for mid-loop budget exhaustion or malformed tool calls (when iterations > 0)
    if (iterations > 0 && (
      e instanceof AnalysisBudgetExceededError ||
      (e instanceof FallbackTriggered && e.reason === 'malformed-tool-calls')
    )) {
      if (import.meta.env.DEV) {
        console.warn('[analyzer] Running salvage synthesis due to:', e instanceof AnalysisBudgetExceededError ? 'budget exceeded' : 'malformed tool calls');
      }
      return runSalvage(e instanceof AnalysisBudgetExceededError ? `salvaged-${e.context?.tripped}` : 'salvaged-malformed');
    }
    throw e;
  }
}