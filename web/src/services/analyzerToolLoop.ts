/**
 * Analyzer tool loop — extracted from analyzer.ts for modularity.
 * Contains the agentic tool loop logic for data analysis.
 */

import type { LLMClient } from '../lib/llm';
import type { LLMMessage, ChartConfig, Insight } from '../lib/types';
import { TOOL_SCHEMAS, executeToolCall, type AnalysisToolContext } from '../lib/analysisTools';
import { CodeExecutionError, AnalysisBudgetExceededError, GenerationFailedError, RateLimitError } from '../lib/errors';
import { buildSystemPrompt, buildInitialUserMessage } from './analyzerPrompts';

export interface ToolTraceEntry {
  tool: string;
  calls: number;
  durationMs: number;
  tokensUsed: number;
}

export interface LoopResult {
  answer: string;
  insights: Insight[];
  chart: ChartConfig | undefined;
  tokensUsed: number;
  iterations: number;
  toolTrace: ToolTraceEntry[];
  partial?: boolean;
  fallbackReason?: string;
}

export interface AnalyzerToolLoopDeps {
  llm: LLMClient;
  datasets: Map<string, unknown>;
  metadata: { [datasetName: string]: { columns: string[]; rowCount: number } };
  codeGenModel?: string;
  maxToolLoopTokens?: number;
  executeUserCode: (code: string) => Promise<unknown>;
}

export interface AnalyzerToolLoopOptions {
  signal?: AbortSignal;
  hooks?: {
    onToolStart?: (info: { tool: string; argsSummary: string; startedAt: number }) => void;
    onToolEnd?: (info: { tool: string; durationMs: number; error?: string }) => void;
    onIteration?: (info: { iteration: number; reflection: string; tokensUsed: number }) => void;
    onSynthesisToken?: (token: string) => void;
  };
  previousContext?: string;
}

const MAX_TOOL_ITERATIONS = 8;
const MAX_TOOL_LOOP_MS = 120_000;
const DEFAULT_TOOL_LOOP_TOKENS = 100_000;
const MAX_TOOL_RESULT_CHARS = 4000;
const ARGS_SUMMARY_MAX_CHARS = 120;
const MAX_CONSECUTIVE_MALFORMED = 2;
const MAX_LLM_CALL_RETRIES = 3;
const LLM_CALL_BASE_BACKOFF_MS = 1000;
const SALVAGE_MAX_COMPLETION_TOKENS = 1024;
const PER_CALL_COMPLETION_CAP = 2048;
const MIN_PER_CALL_TOKENS = 64;

export class FallbackTriggered extends Error {
  constructor(public readonly reason: 'provider-rejected-tools' | 'no-first-tool-call' | 'malformed-tool-calls') {
    super(`Fallback triggered: ${reason}`);
    this.name = 'FallbackTriggered';
  }
}

function summarizeArgs(args: Record<string, unknown>): string {
  const json = JSON.stringify(args);
  if (json.length <= ARGS_SUMMARY_MAX_CHARS) return json;
  return json.slice(0, ARGS_SUMMARY_MAX_CHARS) + '…';
}

function truncateResult(content: string): string {
  if (content.length <= MAX_TOOL_RESULT_CHARS) return content;
  return content.slice(0, MAX_TOOL_RESULT_CHARS) + '… [truncated]';
}

interface ToolTraceAcc { calls: number; durationMs: number; tokensUsed: number; }

const VALID_CHART_TYPES = new Set(['bar', 'line', 'pie'] as const);

/**
 * Parse a synthesis response into {answer, insights, chart}. Strips markdown
 * fences when present; unparseable content is treated as the whole answer.
 */
function parseSynthesisJson(content: string): { answer: string; insights?: Insight[]; chart?: ChartConfig } {
  let stripped = content;
  const fenceMatch = stripped.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch) stripped = fenceMatch[1].trim();
  try {
    const parsed = JSON.parse(stripped);
    return { answer: parsed.answer ?? stripped, insights: parsed.insights ?? [], chart: parsed.chart };
  } catch {
    return { answer: content, insights: [] };
  }
}

/** Coerce insight confidence values outside 'high'|'medium'|'low' down to 'low'. */
function normalizeInsightConfidence(insights: Insight[] | undefined): Insight[] {
  const validConfidence = new Set(['high', 'medium', 'low'] as const);
  if (!Array.isArray(insights)) return insights ?? [];
  for (const insight of insights) {
    if (!validConfidence.has(insight.confidence)) {
      insight.confidence = 'low';
    }
  }
  return insights;
}

/** Structural validation for an LLM-supplied chart before it reaches the UI. */
function isValidChartConfig(chart: ChartConfig | undefined | null): chart is ChartConfig {
  return (
    typeof chart === 'object' &&
    chart !== null &&
    VALID_CHART_TYPES.has(chart.type) &&
    typeof chart.title === 'string' &&
    typeof chart.xKey === 'string' &&
    Array.isArray(chart.yKeys) &&
    chart.yKeys.length > 0 &&
    chart.yKeys.every(k => typeof k === 'string') &&
    Array.isArray(chart.data) &&
    chart.data.length > 0 &&
    chart.data.every(d => d && typeof d === 'object')
  );
}

function buildToolTrace(toolTraceMap: Map<string, ToolTraceAcc>): ToolTraceEntry[] {
  return Array.from(toolTraceMap.entries()).map(([tool, data]) => ({
    tool,
    calls: data.calls,
    durationMs: data.durationMs,
    tokensUsed: data.tokensUsed,
  }));
}

function recordToolTrace(toolTraceMap: Map<string, ToolTraceAcc>, toolName: string, durationMs: number, tokensPerCall: number): void {
  const trace = toolTraceMap.get(toolName) || { calls: 0, durationMs: 0, tokensUsed: 0 };
  trace.calls += 1;
  trace.durationMs += durationMs;
  trace.tokensUsed += tokensPerCall;
  toolTraceMap.set(toolName, trace);
}

export async function runToolLoop(
  question: string,
  relevant: string[],
  deps: AnalyzerToolLoopDeps,
  options: AnalyzerToolLoopOptions = {}
): Promise<LoopResult> {
  const { signal, hooks, previousContext } = options;
  const start = performance.now();
  const tokenBudget = deps.maxToolLoopTokens ?? DEFAULT_TOOL_LOOP_TOKENS;

  const messages: LLMMessage[] = [
    { role: 'user', content: buildInitialUserMessage({ question, relevant, metadata: deps.metadata, previousContext }) },
  ];

  const toolTraceMap = new Map<string, ToolTraceAcc>();
  let tokensUsed = 0;
  let iterations = 0;
  let consecutiveMalformed = 0;

  async function runSalvageSynthesis(fallbackReason: string): Promise<LoopResult> {
    let salvageContent = '';
    const salvageResp = await deps.llm.stream({
      system: buildSystemPrompt(),
      messages,
      temperature: 0,
      model: deps.codeGenModel,
      maxTokens: SALVAGE_MAX_COMPLETION_TOKENS,
    }, (token: string) => {
      salvageContent += token;
      hooks?.onSynthesisToken?.(token);
    }, signal);

    const salvageTokens = salvageResp.usage?.totalTokens ?? 0;
    tokensUsed += salvageTokens;

    // Parse salvage response tolerantly — markdown fences stripped, same
    // contract as the in-loop synthesis path
    const parsed = parseSynthesisJson(salvageResp.content || salvageContent || '');
    const insights = normalizeInsightConfidence(parsed.insights);
    let chart: ChartConfig | undefined = parsed.chart;
    if (chart && !isValidChartConfig(chart)) {
      chart = undefined;
    }

    return {
      answer: parsed.answer,
      insights,
      chart,
      tokensUsed,
      iterations,
      toolTrace: buildToolTrace(toolTraceMap),
      partial: true,
      fallbackReason,
    };
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
          const maxTokens = Math.min(PER_CALL_COMPLETION_CAP, Math.max(MIN_PER_CALL_TOKENS, remainingBudget));
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
      const iterationTokens = resp.usage?.totalTokens ?? 0;
      tokensUsed += iterationTokens;

      // Budget checks
      const elapsedMs = performance.now() - start;
      if (elapsedMs > MAX_TOOL_LOOP_MS) {
        throw new AnalysisBudgetExceededError({
          iterations,
          elapsedMs,
          tokensUsed,
          tripped: 'time',
          limit: MAX_TOOL_LOOP_MS,
        });
      }
      if (tokensUsed > tokenBudget) {
        throw new AnalysisBudgetExceededError({
          iterations,
          elapsedMs,
          tokensUsed,
          tripped: 'tokens',
          limit: tokenBudget,
        });
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

        for (const call of toolCalls) {
          const toolName = call.function.name;
          let toolResult: unknown;
          let toolError: string | undefined;
          const toolStart = Date.now();

          // Validate tool call
          let args: Record<string, unknown> = {};
          try {
            args = JSON.parse(call.function.arguments || '{}');
          } catch {
            // Malformed arguments
            consecutiveMalformed++;
            toolError = `malformed tool call: invalid JSON arguments`;
            toolResult = { error: toolError };
            if (import.meta.env.DEV) console.warn('[analyzer] Malformed tool call arguments:', call.function.arguments);
          }

          if (!toolError) {
            if (!toolName || typeof toolName !== 'string') {
              consecutiveMalformed++;
              toolError = 'malformed tool call: missing function name';
              toolResult = { error: toolError };
              if (import.meta.env.DEV) console.warn('[analyzer] Malformed tool call: missing name');
            }
          }

          if (!toolError) {
            // Valid call - reset malformed counter
            consecutiveMalformed = 0;

            // Fire onToolStart hook
            try {
              hooks?.onToolStart?.({
                tool: toolName,
                argsSummary: summarizeArgs(args),
                startedAt: toolStart,
              });
            } catch (hookErr) {
              if (import.meta.env.DEV) console.warn('[analyzer] onToolStart hook threw:', hookErr);
            }

            // Execute tool
            try {
              const ctx: AnalysisToolContext = {
                datasets: deps.datasets,
                metadata: deps.metadata,
              };
              toolResult = await executeToolCall(ctx, toolName, args, deps.executeUserCode);
            } catch (e) {
              const err = e instanceof Error ? e : new Error(String(e));
              toolError = err.message;
              toolResult = { error: toolError };
              if (import.meta.env.DEV) console.warn(`[analyzer] Tool ${toolName} failed:`, toolError);
            }

            const durationMs = Date.now() - toolStart;
            recordToolTrace(toolTraceMap, toolName, durationMs, tokensPerCall);

            try {
              hooks?.onToolEnd?.({
                tool: toolName,
                durationMs,
                error: toolError,
              });
            } catch (hookErr) {
              if (import.meta.env.DEV) console.warn('[analyzer] onToolEnd hook threw:', hookErr);
            }
          } else {
            recordToolTrace(toolTraceMap, toolName, Date.now() - toolStart, tokensPerCall);
          }

          // Tool result message
          messages.push({
            role: 'tool',
            content: truncateResult(JSON.stringify(toolResult)),
            toolCallId: call.id,
          });
        }

        // Check consecutive malformed limit - Trigger (c)
        if (consecutiveMalformed >= MAX_CONSECUTIVE_MALFORMED) {
          throw new FallbackTriggered('malformed-tool-calls');
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
    throw new AnalysisBudgetExceededError({
      iterations,
      elapsedMs: performance.now() - start,
      tokensUsed,
      tripped: 'iterations',
      limit: MAX_TOOL_ITERATIONS,
    });
  } catch (e) {
    // Salvage synthesis for mid-loop budget exhaustion or malformed tool calls (when iterations > 0)
    if (iterations > 0 && (
      e instanceof AnalysisBudgetExceededError ||
      (e instanceof FallbackTriggered && e.reason === 'malformed-tool-calls')
    )) {
      if (import.meta.env.DEV) {
        console.warn('[analyzer] Running salvage synthesis due to:', e instanceof AnalysisBudgetExceededError ? 'budget exceeded' : 'malformed tool calls');
      }
      return runSalvageSynthesis(e instanceof AnalysisBudgetExceededError ? `salvaged-${e.context?.tripped}` : 'salvaged-malformed');
    }
    throw e;
  }
}