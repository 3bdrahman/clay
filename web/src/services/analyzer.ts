// Data analyzer — replaces Python exec() with safe in-browser code generation
// Generates Arquero-compatible code (a pandas-like dataframe library)

import type { ChartConfig, DataAnalysisResult, DatasetSummary, LLMMessage } from '../lib/types';
import type { LLMClient } from '../lib/llm';
import { CodeExecutionError, GenerationFailedError, RateLimitError } from '../lib/errors';
import { TOOL_SCHEMAS, executeToolCall, type AnalysisToolContext } from '../lib/analysisTools';
import { AnalysisBudgetExceededError } from '../lib/errors';

export interface DatasetMeta {
  [datasetName: string]: {
    columns: string[];
    rowCount: number;
  };
}

export interface AnalyzerHooks {
  onToolStart?: (info: { tool: string; argsSummary: string; startedAt: number }) => void;
  onToolEnd?: (info: { tool: string; durationMs: number; error?: string }) => void;
  onIteration?: (info: { iteration: number; reflection: string; tokensUsed: number }) => void;
}

export interface DataAnalyzer {
  analyze(question: string, signal?: AbortSignal, hooks?: AnalyzerHooks, previousContext?: string): Promise<DataAnalysisResult>;
  listDatasets(): DatasetSummary[];
  getDatasetSummary(name: string): DatasetSummary | undefined;
}

export interface DataAnalyzerDeps {
  llm: LLMClient;
  datasets: Map<string, unknown>;
  metadata: DatasetMeta;
  codeGenModel?: string;
  maxToolLoopTokens?: number; // cumulative tool-loop token budget; undefined = default
}

class FallbackTriggered extends Error {
  constructor(public readonly reason: 'provider-rejected-tools' | 'no-first-tool-call' | 'malformed-tool-calls') {
    super(`Fallback triggered: ${reason}`);
    this.name = 'FallbackTriggered';
  }
}

/**
 * Create a data analyzer that generates and executes Arquero code for CSV analysis.
 * Uses LLM to generate JavaScript, executes safely via new Function(), detects chart config.
 * @param deps - LLM client, dataset Map, metadata, optional codeGenModel, optional maxToolLoopTokens
 * @returns DataAnalyzer with analyze(), listDatasets(), getDatasetSummary()
 */
export function createDataAnalyzer(deps: DataAnalyzerDeps): DataAnalyzer {
  const { llm, metadata } = deps;

  const NAME_TOKEN_MATCH_SCORE = 4;
  const COLUMN_TOKEN_MATCH_SCORE = 2;
  const MAX_RELEVANT_DATASETS = 4;
  const MAX_CHART_ROWS = 12;
  const MAX_ANALYSIS_ATTEMPTS = 2;

  // Named constants for the agentic tool loop (AGENTS.md: no magic numbers)
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

  function relevantDatasets(question: string): string[] {
    const q = question.toLowerCase().replace(/[^a-z0-9_\s]/g, ' ');
    const tokens = new Set(q.split(/\s+/).filter(t => t.length >= 3));
    const matches: Array<{ name: string; score: number }> = [];
    for (const [name, meta] of Object.entries(metadata)) {
      let score = 0;
      const nameTokens = name.toLowerCase().split(/[_\s]+/);
      for (const t of nameTokens) {
        if (t.length >= 3 && tokens.has(t)) score += NAME_TOKEN_MATCH_SCORE;
      }
      for (const col of meta.columns) {
        const colTokens = col.toLowerCase().split(/[_\s]+/);
        for (const t of colTokens) {
          if (t.length >= 3 && tokens.has(t)) score += COLUMN_TOKEN_MATCH_SCORE;
        }
      }
      if (score > 0) matches.push({ name, score });
    }
    matches.sort((a, b) => b.score - a.score);
    return matches.slice(0, MAX_RELEVANT_DATASETS).map(m => m.name);
  }

  function buildPrompt(question: string, relevant: string[]): string {
    const datasetInfo = relevant
      .map(name => {
        const meta = metadata[name];
        return `- ${name} (${meta?.rowCount || '?'} rows): columns = ${JSON.stringify(meta?.columns || [])}`;
      })
      .join('\n');
    return `Question: ${question}

Datasets available as Arquero tables (variable name = dataset name):
${datasetInfo}

Generate JavaScript using the Arquero library (available as 'aq'). Datasets are loaded as variables named after their dataset name and are real Arquero tables.

Example patterns:
- Filter: employees.filter(d => d.department === 'Engineering')
- Group count: employees.groupby('department').count()
- Sort top N: projects.orderby('budget_usd', 'desc').limit(5)
- Aggregate: projects.groupby('status').rollup({ total: d => op.sum(d.budget_usd) })
- Join: feedback.join(projects, ['project_id'])

Return JSON with literal text inside the code block (no outer braces):
{"code": "<your JavaScript code, ending with result = ...>", "explanation": "<brief explanation of the analysis>"}`;
  }

  function buildRetryPrompt(question: string, lastError: string): string {
    return `The previous code failed with: ${lastError}

Question: ${question}

Generate FIXED JavaScript code using Arquero (loaded as 'aq'). Available datasets: ${Object.keys(deps.datasets).filter(n => n !== 'aq').join(', ')}.

Common pitfalls to avoid:
- Don't use pandas syntax (no .iloc, no pd, no Python f-strings)
- Use Arquero verbs: .filter(), .groupby(), .count(), .orderby(), .limit(), .rollup(), .join()
- Access columns as d.columnName or d['column name with spaces']
- Store result in a variable named result
- Return small JSON-safe results (objects, arrays of objects, or primitives)

Return JSON: {"code": "...", "explanation": "..."}`;
  }

  /**
   * SECURITY: Execute LLM-generated JavaScript in a constrained sandbox.
   *
   * ── Trust boundary ────────────────────────────────────────────────────────
   * The source of `code` is an LLM completion, NOT user-typed input. The LLM
   * has been system-prompted to emit only Arquero transformations, but the
   * underlying threat is prompt injection: a malicious document ingested via
   * the vectorstore (or a user-chosen CSV column header) could contain
   * instructions the LLM dutifully echoes into the generated code, after which
   * `new Function(...)` executes them in the host page context.
   *
   * ── Mitigations (current) ─────────────────────────────────────────────────
   *  - Strict mode (`"use strict"`) forbids accidental globals / `with` /
   *    undeclared assignments.
   *  - Limited scope. The only external references reachable from the
   *    generated code are:
   *        • the `aq` Arquero namespace (frozen — see below)
   *        • `op` Arquero operators object (frozen — see below)
   *        • one parameter per loaded dataset table (Arquero `ColumnTable`)
   *        • the `result` slot reserved for the return value
   *    No `window`, `globalThis`, `document`, `fetch`, `eval`, `import`,
   *    `require`, `process`, or any DOM/network primitive is passed in. The
   *    generated code can still *reach* the global scope via property chains
   *    such as `(() => {}).constructor.constructor("...")()` — that is the
   *    inherent risk of `new Function` and the reason the document-ingestion
   *    path lives in the same browser tab rather than a worker.
   *
   *  - Input discipline. CSV column names are NOT injected as identifier names
   *    (they're accessed as `d['column name with spaces']`). The only
   *    identifierss derived from user-controlled data are dataset names, which
   *    are themselves produced by `deriveName()` in `services/files.ts` and
   *    stripped to `[a-zA-Z0-9_]`.
   *
   * ── Mitigations NOT applied (and why) ─────────────────────────────────────
   *  - Web Worker isolation. Moving execution to a Worker would buy true
   *    wall-clock isolation (no access to `window`, no synchronous DOM), at
   *    the cost of postMessage serialization of Arquero tables on every call.
   *    Tracked as a follow-up — this function is the only call site that
   *    would move.
   *  - CSP `unsafe-eval` removal. The Vite dev build and the GitHub Pages
   *    deploy both rely on `new Function`; turning it off would also disable
   *    React refresh in dev. A proper sandbox (`quickjs-emscripten`,
   *    `proxy-tree-walker`) is the long-term fix and is incompatible with the
   *    current "no backend, no wasm" deploy profile.
   *  - Static code validation. We could AST-scan generated code for forbidden
   *    references before execution, but a determined prompt-injection can
   *    construct the same refs dynamically (`[][`constructor`]` etc.). The
   *    retry loop already rejects `SyntaxError` and timeouts; runtime failures
   *    return an error result to the UI instead of crashing the chat.
   *
   * ── Hardening applied here ────────────────────────────────────────────────
   * `op` is passed as a fresh shallow clone so generated code cannot mutate
   * the real `aq.op` and corrupt subsequent Arquero verbs. we do NOT freeze
   * `aq` because Arquero internally relies on the namespace being mutable
   * (attempted and reverted after test regression). Dataset tables are
   * Arquero `ColumnTable` instances (immutable by contract) and are re-read
   * from `deps.datasets` on every `analyze()` call, so generated code cannot
   * poison the in-memory store for the next question.
   *
   * @throws CodeExecutionError for syntax errors, runtime errors, timeouts
   */
  function executeUserCode(code: string): unknown {
    const aq = deps.datasets.get('aq') as { op?: Record<string, unknown>; from?: unknown; fromCSV?: unknown } | undefined;
    const datasetsObj: Record<string, unknown> = {};
    for (const [name, table] of deps.datasets) {
      if (name === 'aq') continue;
      datasetsObj[name] = table;
    }
    const opRaw = aq?.op || {};
    // Create a request-scoped aq wrapper to isolate the namespace per execution
    const aqRef = {
      from: aq?.from,
      fromCSV: aq?.fromCSV,
    };
    const opRef = { ...opRaw };
    const argNames = Object.keys(datasetsObj);
    const argValues = Object.values(datasetsObj);

    // eslint-disable-next-line @typescript-eslint/no-implied-eval, no-new-func
    const fn = new Function(
      ...argNames,
      'aq',
      'op',
      '"use strict"; let result; ' + code + '; return result;'
    );

    try {
      const result = fn(...argValues, aqRef, opRef);
      if (result && typeof result === 'object' && typeof (result as { objects?: () => unknown[] }).objects === 'function') {
        return (result as { objects: () => unknown[] }).objects();
      }
      return result;
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));

      const isSyntaxError = error instanceof SyntaxError ||
        error.name === 'SyntaxError' ||
        error.message.includes('SyntaxError') ||
        error.message.includes('Unexpected token') ||
        error.message.includes('Unexpected end of input');

      const isTimeout = error.name === 'TimeoutError' ||
        error.message.includes('timeout') ||
        error.message.includes('timed out');

      throw new CodeExecutionError(
        isSyntaxError ? 'Syntax error in generated code' :
        isTimeout ? 'Code execution timed out' :
        'Runtime error in generated code',
        error,
        {
          code,
          retryable: !isSyntaxError && !isTimeout,
        }
      );
    }
  }

  function tryDetectChart(rows: unknown[]): ChartConfig | undefined {
    if (rows.length === 0) return undefined;
    const first = rows[0] as Record<string, unknown>;
    if (!first || typeof first !== 'object') return undefined;
    const keys = Object.keys(first);
    const numericKeys = keys.filter(k => typeof first[k] === 'number');
    if (numericKeys.length === 0) return undefined;
    const data = rows.slice(0, MAX_CHART_ROWS) as Array<Record<string, unknown>>;
    const xKey = keys.find(k => typeof first[k] === 'string') || keys[0];
    return {
      type: 'bar',
      title: 'Analysis Result',
      xKey,
      yKeys: numericKeys.slice(0, 2),
      data,
    };
  }

  function formatResult(
    question: string,
    result: unknown,
    code: string,
    explanation: string,
    attempts: number,
    start: number,
    extra?: {
      mode: 'tools' | 'single-shot';
      insights?: Insight[];
      toolTrace?: Array<{ tool: string; calls: number; durationMs: number; tokensUsed: number }>;
    }
  ): DataAnalysisResult {
    let resultType: DataAnalysisResult['resultType'] = 'scalar';
    let chartConfig: ChartConfig | undefined;
    let displayResult: unknown = result;

    // Detect Arquero ColumnTable and convert to array of objects
    const isColumnTable = result && typeof result === 'object' &&
      typeof (result as { objects?: () => unknown[] }).objects === 'function';
    const resultArray = isColumnTable
      ? (result as { objects: () => unknown[] }).objects()
      : null;

    if (resultArray) {
      resultType = 'table';
      displayResult = resultArray;
      chartConfig = tryDetectChart(resultArray);
      if (chartConfig) resultType = 'chart';
    } else if (Array.isArray(result)) {
      resultType = 'table';
      displayResult = result;
      chartConfig = tryDetectChart(result);
      if (chartConfig) resultType = 'chart';
    } else if (result && typeof result === 'object') {
      const obj = result as Record<string, unknown>;
      const keys = Object.keys(obj);
      const values = Object.values(obj);

      if (values.every(v => typeof v === 'number') && keys.length > 1) {
        resultType = 'chart';
        const data = keys.map(k => ({ name: k, value: obj[k] as number }));
        chartConfig = { type: 'bar', title: 'Result', xKey: 'name', yKeys: ['value'], data };
        displayResult = obj;
      }
      else if (keys.length > 0 && Array.isArray(obj[keys[0]])) {
        resultType = 'table';
        displayResult = obj;
      }
      else if (keys.length === 1 && typeof obj[keys[0]] === 'number') {
        resultType = 'chart';
        const data = [{ name: keys[0], value: obj[keys[0]] as number }];
        chartConfig = { type: 'bar', title: 'Result', xKey: 'name', yKeys: ['value'], data };
        displayResult = obj;
      }
      else {
        resultType = 'scalar';
        displayResult = obj;
      }
    } else {
      resultType = 'scalar';
      displayResult = result;
    }

    return {
      type: 'data_analysis',
      question,
      code,
      explanation,
      resultType,
      result: displayResult,
      chartConfig,
      attempts,
      durationMs: performance.now() - start,
      timestamp: Date.now(),
      insights: extra?.insights,
      mode: extra?.mode,
      toolTrace: extra?.toolTrace,
    };
  }

  // ============================================================================
  // Single-shot analysis (capability fallback for todo 7)
  // ============================================================================

  async function _analyzeSingleShot(question: string, signal?: AbortSignal): Promise<DataAnalysisResult> {
    const start = performance.now();
    const relevant = relevantDatasets(question);
    const maxAttempts = MAX_ANALYSIS_ATTEMPTS;
    let lastError: string | null = null;

    for (let attempt = 0; attempt <= maxAttempts; attempt++) {
      // Check for abort signal
      if (signal?.aborted) {
        throw new CodeExecutionError('Analysis aborted', new Error('AbortSignal triggered'), {
          code: '',
          retryable: false,
        });
      }

      let code = '';
      try {
        const prompt = attempt === 0
          ? buildPrompt(question, relevant)
          : buildRetryPrompt(question, lastError || 'unknown error');

        const resp = await llm.invoke({
          system:
            'You are a data analyst. Generate JavaScript code that operates on a pre-loaded "aq" (Arquero) variable and any of these datasets as Arquero tables: ' +
            Object.keys(deps.datasets).join(', ') +
            '. Always store the final answer in a variable named result. Return JSON with code (the JS source) and explanation.',
          messages: [{ role: 'user', content: prompt }],
          jsonMode: true,
          temperature: 0,
          model: deps.codeGenModel,
        });

        const parsed = JSON.parse(resp.content || '{"code":"","explanation":""}');
        code = parsed.code || '';
        if (!code) throw new Error('Empty code from LLM');
        const result = executeUserCode(code);
        return formatResult(question, result, code, parsed.explanation, attempt + 1, start);
      } catch (e) {
        const error = e instanceof Error ? e : new Error(String(e));
        lastError = error.message;

        // If it's already a CodeExecutionError, check if retryable
        if (error instanceof CodeExecutionError && !error.retryable) {
          // Non-retryable error (syntax, timeout) - fail immediately
          if (import.meta.env.DEV) console.warn(`Analysis attempt ${attempt + 1} failed (non-retryable):`, lastError);
          return formatErrorResult(question, code, error, attempt + 1, start);
        }

        if (import.meta.env.DEV) console.warn(`Analysis attempt ${attempt + 1} failed:`, lastError);
        if (attempt >= maxAttempts) {
          return formatErrorResult(question, code, error, attempt + 1, start);
        }
      }
    }
    return formatErrorResult(question, '', new Error('Unknown error'), 0, start);
  }

  function formatErrorResult(
    question: string,
    code: string,
    error: Error,
    attempts: number,
    start: number
  ): DataAnalysisResult {
    return {
      type: 'data_analysis',
      question,
      code,
      explanation: error instanceof CodeExecutionError ? error.message : 'Code execution failed',
      resultType: 'error',
      result: error.message,
      chartConfig: undefined,
      attempts,
      durationMs: performance.now() - start,
      timestamp: Date.now(),
    };
  }

  // ============================================================================
  // Agentic tool loop (primary analyze path)
  // ============================================================================

  interface ToolTraceEntry {
    tool: string;
    calls: number;
    durationMs: number;
    tokensUsed: number;
  }

  interface LoopResult {
    answer: string;
    insights: Insight[];
    chart: ChartConfig | undefined;
    tokensUsed: number;
    iterations: number;
    toolTrace: ToolTraceEntry[];
    partial?: boolean;
    fallbackReason?: string;
  }

  interface Insight {
    finding: string;
    evidence: string;
    confidence: 'high' | 'medium' | 'low';
    implication?: string;
  }

  function buildSystemPrompt(): string {
    return `You are a senior data analyst. You have tools to inspect the actual data — begin EVERY analysis by calling list_datasets, then profile_column on the relevant columns BEFORE reasoning. Quantify every claim with numbers from tool results. Verify patterns against the actual data (use filter_sample to inspect rows). Compute correlations where meaningful. State limitations and confidence honestly — never overclaim causation from correlation. When the evidence is sufficient, STOP calling tools and return the final synthesis as JSON with this exact structure:
{
  "answer": "your final answer text",
  "insights": [
    {"finding": "...", "evidence": "...", "confidence": "high|medium|low", "implication": "..."}
  ],
  "chart": {"type": "bar|line|pie", "title": "...", "xKey": "...", "yKeys": [...], "data": [...]}
}

The chart field is optional. Include it only when a visualization adds value.

CRITICAL: After EVERY tool call response, you MUST include a one-line reflection in your next message — what the results told you and what you will do next. This reflection is required for every iteration.`;
  }

  function buildInitialUserMessage(question: string, relevant: string[], previousContext?: string): string {
    const datasetInfo = relevant
      .map(name => {
        const meta = metadata[name];
        return `- ${name} (${meta?.rowCount || '?'} rows): columns = ${JSON.stringify(meta?.columns || [])}`;
      })
      .join('\n');
    let message = `Question: ${question}

Available datasets:
${datasetInfo}

Use the tools to explore the data and answer the question.`;
    if (previousContext) {
      message = `Previous analysis context (from the last data question):\n${previousContext}\n\n---\n\n${message}`;
    }
    return message;
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

  async function runToolLoop(
    question: string,
    relevant: string[],
    signal?: AbortSignal,
    hooks?: AnalyzerHooks,
    previousContext?: string
  ): Promise<LoopResult> {
    const start = performance.now();
    const tokenBudget = deps.maxToolLoopTokens ?? DEFAULT_TOOL_LOOP_TOKENS;

    const messages: LLMMessage[] = [
      { role: 'user', content: buildInitialUserMessage(question, relevant, previousContext) },
    ];

    const toolTraceMap = new Map<string, { calls: number; durationMs: number; tokensUsed: number }>();
    let tokensUsed = 0;
    let iterations = 0;
    let consecutiveMalformed = 0;

    async function runSalvageSynthesis(fallbackReason: string): Promise<LoopResult> {
      // Run one salvage synthesis invoke with accumulated messages, no tools, jsonMode false
      const salvageResp = await llm.invoke({
        system: buildSystemPrompt(),
        messages,
        temperature: 0,
        model: deps.codeGenModel,
        maxTokens: SALVAGE_MAX_COMPLETION_TOKENS,
        // No tools, no jsonMode - free-form text response
      }, signal);

      const salvageTokens = salvageResp.usage?.totalTokens ?? 0;
      tokensUsed += salvageTokens;

      // Parse salvage response tolerantly
      let answer = salvageResp.content || '';
      let insights: Insight[] = [];
      let chart: ChartConfig | undefined = undefined;

      try {
        // Try to parse as JSON first
        const parsed = JSON.parse(answer);
        answer = parsed.answer ?? answer;
        insights = parsed.insights ?? [];
        chart = parsed.chart;
      } catch {
        // Unparseable - treat whole content as answer
      }

      // Normalize insight confidence
      const validConfidence = new Set(['high', 'medium', 'low'] as const);
      if (Array.isArray(insights)) {
        for (const insight of insights) {
          if (!validConfidence.has(insight.confidence)) {
            insight.confidence = 'low';
          }
        }
      }

      // Validate chart
      const validChartTypes = new Set(['bar', 'line', 'pie'] as const);
      if (chart) {
        const isValidChart =
          typeof chart === 'object' &&
          chart !== null &&
          validChartTypes.has(chart.type) &&
          typeof chart.title === 'string' &&
          typeof chart.xKey === 'string' &&
          Array.isArray(chart.yKeys) &&
          chart.yKeys.length > 0 &&
          chart.yKeys.every(k => typeof k === 'string') &&
          Array.isArray(chart.data) &&
          chart.data.length > 0 &&
          chart.data.every(d => d && typeof d === 'object');
        if (!isValidChart) {
          chart = undefined;
        }
      }

      const toolTrace: ToolTraceEntry[] = Array.from(toolTraceMap.entries()).map(([tool, data]) => ({
        tool,
        calls: data.calls,
        durationMs: data.durationMs,
        tokensUsed: data.tokensUsed,
      }));

      return {
        answer,
        insights,
        chart,
        tokensUsed,
        iterations,
        toolTrace,
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

        let resp: Awaited<ReturnType<typeof llm.invoke>>;
        let llmCallAttempt = 0;
        while (true) {
          try {
            const remainingBudget = tokenBudget - tokensUsed;
            const maxTokens = Math.min(PER_CALL_COMPLETION_CAP, Math.max(MIN_PER_CALL_TOKENS, remainingBudget));
            resp = await llm.invoke({
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
        if (iterations > MAX_TOOL_ITERATIONS) {
          throw new AnalysisBudgetExceededError({
            iterations,
            elapsedMs,
            tokensUsed,
            tripped: 'iterations',
            limit: MAX_TOOL_ITERATIONS,
          });
        }
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

        const toolCalls = resp.toolCalls;
        const finishReason = resp.finishReason;

        if (toolCalls && toolCalls.length > 0) {
          // Assistant message with tool calls
          messages.push({
            role: 'assistant',
            content: resp.content || '',
            toolCalls,
          });

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
                  metadata,
                };
                toolResult = executeToolCall(ctx, toolName, args, executeUserCode);
              } catch (e) {
                const err = e instanceof Error ? e : new Error(String(e));
                toolError = err.message;
                toolResult = { error: toolError };
                if (import.meta.env.DEV) console.warn(`[analyzer] Tool ${toolName} failed:`, toolError);
              }

              const durationMs = Date.now() - toolStart;

              const tokensPerCall = toolCalls.length > 0 ? Math.round(iterationTokens / toolCalls.length) : 0;
              const trace = toolTraceMap.get(toolName) || { calls: 0, durationMs: 0, tokensUsed: 0 };
              trace.calls += 1;
              trace.durationMs += durationMs;
              trace.tokensUsed += tokensPerCall;
              toolTraceMap.set(toolName, trace);

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
              const durationMs = Date.now() - toolStart;
              const tokensPerCall = toolCalls.length > 0 ? Math.round(iterationTokens / toolCalls.length) : 0;
              const trace = toolTraceMap.get(toolName) || { calls: 0, durationMs: 0, tokensUsed: 0 };
              trace.calls += 1;
              trace.durationMs += durationMs;
              trace.tokensUsed += tokensPerCall;
              toolTraceMap.set(toolName, trace);
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
          // Final synthesis
          let synthesis: { answer: string; insights?: Insight[]; chart?: ChartConfig } = {
            answer: resp.content || '',
            insights: [],
          };

          try {
            // Strip markdown fences if present
            let content = resp.content || '';
            const fenceMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
            if (fenceMatch) content = fenceMatch[1].trim();
            const parsed = JSON.parse(content);
            synthesis = {
              answer: parsed.answer ?? content,
              insights: parsed.insights ?? [],
              chart: parsed.chart,
            };
          } catch {
            // Unparseable - treat whole content as answer
            synthesis = { answer: resp.content || '', insights: [] };
          }

          // FIX 2a: Normalize insight confidence to 'high' | 'medium' | 'low' enum
          const validConfidence = new Set(['high', 'medium', 'low'] as const);
          if (Array.isArray(synthesis.insights)) {
            for (const insight of synthesis.insights) {
              if (!validConfidence.has(insight.confidence)) {
                insight.confidence = 'low';
              }
            }
          }

          // FIX 2b: Validate synthesis chart shape before trusting it
          const validChartTypes = new Set(['bar', 'line', 'pie'] as const);
          let chart: ChartConfig | undefined = synthesis.chart;
          if (chart) {
            const isValidChart =
              typeof chart === 'object' &&
              chart !== null &&
              validChartTypes.has(chart.type) &&
              typeof chart.title === 'string' &&
              typeof chart.xKey === 'string' &&
              Array.isArray(chart.yKeys) &&
              chart.yKeys.length > 0 &&
              chart.yKeys.every(k => typeof k === 'string') &&
              Array.isArray(chart.data) &&
              chart.data.length > 0 &&
              chart.data.every(d => d && typeof d === 'object');
            if (!isValidChart) {
              chart = undefined;
            }
          }

          const toolTrace: ToolTraceEntry[] = Array.from(toolTraceMap.entries()).map(([tool, data]) => ({
            tool,
            calls: data.calls,
            durationMs: data.durationMs,
            tokensUsed: data.tokensUsed,
          }));

          return {
            answer: synthesis.answer,
            insights: synthesis.insights ?? [],
            chart,
            tokensUsed,
            iterations,
            toolTrace,
          };
        }

        // Other finish reasons (length, content_filter, etc.) - treat as stop
        const toolTrace: ToolTraceEntry[] = Array.from(toolTraceMap.entries()).map(([tool, data]) => ({
          tool,
          calls: data.calls,
          durationMs: data.durationMs,
          tokensUsed: data.tokensUsed,
        }));

        return {
          answer: resp.content || '',
          insights: [],
          chart: undefined,
          tokensUsed,
          iterations,
          toolTrace,
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

  async function analyze(question: string, signal?: AbortSignal, hooks?: AnalyzerHooks, previousContext?: string): Promise<DataAnalysisResult> {
    const start = performance.now();

    // FIX 1: Empty datasets guard — return early with helpful message if no real datasets loaded
    const realDatasetNames = [...deps.datasets.keys()].filter(n => n !== 'aq');
    if (realDatasetNames.length === 0) {
      return {
        type: 'data_analysis',
        question,
        code: '',
        explanation: 'No datasets are loaded yet. Upload a CSV in the Data panel first, then ask again.',
        resultType: 'scalar',
        result: 'No datasets loaded',
        chartConfig: undefined,
        attempts: 0,
        durationMs: performance.now() - start,
        timestamp: Date.now(),
      };
    }

    const relevant = relevantDatasets(question);

    try {
      const loopResult = await runToolLoop(question, relevant, signal, hooks, previousContext);

      const toolTrace: Array<{ tool: string; calls: number; durationMs: number; tokensUsed: number }> = Array.from(
        loopResult.toolTrace
      );

      // For tool loop, the result is the synthesis answer (string), and chart comes from synthesis
      const resultType = loopResult.chart ? 'chart' : 'scalar';
      const isPartial = loopResult.partial === true;
      return {
        type: 'data_analysis',
        question,
        code: '',
        explanation: loopResult.answer,
        resultType,
        result: loopResult.answer,
        chartConfig: loopResult.chart,
        insights: loopResult.insights,
        attempts: loopResult.iterations,
        durationMs: performance.now() - start,
        timestamp: Date.now(),
        mode: isPartial ? ('fallback' as const) : ('tools' as const),
        fallbackReason: isPartial ? loopResult.fallbackReason : undefined,
        partial: isPartial,
        toolTrace,
      };
    } catch (e) {
      if (e instanceof FallbackTriggered) {
        const singleShotResult = await _analyzeSingleShot(question, signal);
        return {
          ...singleShotResult,
          mode: 'single-shot' as const,
          fallbackReason: e.reason,
        };
      }
      throw e;
    }
  }

  function getDatasetSummary(name: string): DatasetSummary | undefined {
    const meta = metadata[name];
    if (!meta) return undefined;
    const table = deps.datasets.get(name);
    let rowCount = 0;
    let columns: string[] = meta.columns;
    if (table && typeof (table as { numRows?: () => number }).numRows === 'function') {
      try {
        rowCount = (table as { numRows: () => number }).numRows();
        const names = (table as { columnNames?: () => string[] }).columnNames?.();
        if (Array.isArray(names) && names.length > 0) columns = names;
      } catch (e) {
        if (import.meta.env.DEV) {
          console.warn('[analyzer] getDatasetSummary live-table inspection failed (falling back to metadata):', e);
        }
        rowCount = meta.rowCount;
      }
    }
    return { name, rowCount, columns };
  }

  function listDatasets(): DatasetSummary[] {
    return Object.keys(metadata)
      .map(name => getDatasetSummary(name))
      .filter((s): s is DatasetSummary => !!s);
  }

  return { analyze, listDatasets, getDatasetSummary };
}