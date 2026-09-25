// Single-shot analysis fallback for models without tool support. Used when
// the provider rejects the tools param or never emits a tool call.

import type { ChartConfig, DataAnalysisResult } from '../lib/types';
import type { LLMClient } from '../lib/llm';
import { CodeExecutionError } from '../lib/errors';
import { buildPrompt, buildRetryPrompt } from './analyzerPrompts';
import type { DatasetMeta } from './analyzer';

const MAX_CHART_ROWS = 12;
const MAX_ANALYSIS_ATTEMPTS = 2;

export interface SingleShotDeps {
  llm: LLMClient;
  datasets: Map<string, unknown>;
  metadata: DatasetMeta;
  codeGenModel?: string;
  relevantDatasets: (question: string) => string[];
  executeUserCode: (code: string) => unknown;
}

export function createSingleShotAnalyzer(
  deps: SingleShotDeps,
): (question: string, signal?: AbortSignal) => Promise<DataAnalysisResult> {
  const { llm, metadata, relevantDatasets, executeUserCode } = deps;

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
    start: number
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
    };
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

  async function analyzeSingleShot(question: string, signal?: AbortSignal): Promise<DataAnalysisResult> {
    const start = performance.now();
    const relevant = relevantDatasets(question);
    const maxAttempts = MAX_ANALYSIS_ATTEMPTS;
    let lastError: string | null = null;

    // Get dataset names (excluding 'aq' namespace) for prompts
    const datasetNames = [...deps.datasets.keys()].filter(n => n !== 'aq');

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
          ? buildPrompt({ question, relevant, metadata })
          : buildRetryPrompt({ question, lastError, datasetNames });

        const resp = await llm.invoke({
          system:
            'You are a data analyst. Generate JavaScript code that operates on a pre-loaded "aq" (Arquero) variable and any of these datasets as Arquero tables: ' +
            datasetNames.join(', ') +
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
          if (import.meta.env.DEV) console.warn(`[analyzer] Analysis attempt ${attempt + 1} failed (non-retryable):`, lastError);
          return formatErrorResult(question, code, error, attempt + 1, start);
        }

        if (import.meta.env.DEV) console.warn(`[analyzer] Analysis attempt ${attempt + 1} failed:`, lastError);
        if (attempt >= maxAttempts) {
          return formatErrorResult(question, code, error, attempt + 1, start);
        }
      }
    }
    return formatErrorResult(question, '', new Error('Analysis retry loop exited without a result'), 0, start);
  }

  return analyzeSingleShot;
}
