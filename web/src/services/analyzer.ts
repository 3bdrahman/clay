// Data analyzer — replaces Python exec() with safe in-browser code generation
// Generates Arquero-compatible code (a pandas-like dataframe library)

import type { ChartConfig, DataAnalysisResult, DatasetSummary } from '../lib/types';
import type { EmbeddingsClient } from '../lib/embeddings';
import type { LLMClient } from '../lib/llm';
import { CodeExecutionError } from '../lib/errors';

export interface DatasetMeta {
  [datasetName: string]: {
    columns: string[];
    rowCount: number;
  };
}

export interface DataAnalyzer {
  analyze(question: string, signal?: AbortSignal): Promise<DataAnalysisResult>;
  listDatasets(): DatasetSummary[];
  getDatasetSummary(name: string): DatasetSummary | undefined;
}

export interface DataAnalyzerDeps {
  llm: LLMClient;
  embeddings: EmbeddingsClient;
  datasets: Map<string, unknown>;
  metadata: DatasetMeta;
  codeGenModel?: string;
}

/**
 * Create a data analyzer that generates and executes Arquero code for CSV analysis.
 * Uses LLM to generate JavaScript, executes safely via new Function(), detects chart config.
 * @param deps - LLM client, embeddings client, dataset Map, metadata, optional codeGenModel
 * @returns DataAnalyzer with analyze(), listDatasets(), getDatasetSummary()
 */
export function createDataAnalyzer(deps: DataAnalyzerDeps): DataAnalyzer {
  const { llm, metadata } = deps;

  function relevantDatasets(question: string): string[] {
    const q = question.toLowerCase().replace(/[^a-z0-9_\s]/g, ' ');
    const tokens = new Set(q.split(/\s+/).filter(t => t.length >= 3));
    const matches: Array<{ name: string; score: number }> = [];
    for (const [name, meta] of Object.entries(metadata)) {
      let score = 0;
      const nameTokens = name.toLowerCase().split(/[_\s]+/);
      for (const t of nameTokens) {
        if (t.length >= 3 && tokens.has(t)) score += 4;
      }
      for (const col of meta.columns) {
        const colTokens = col.toLowerCase().split(/[_\s]+/);
        for (const t of colTokens) {
          if (t.length >= 3 && tokens.has(t)) score += 2;
        }
      }
      if (score > 0) matches.push({ name, score });
    }
    matches.sort((a, b) => b.score - a.score);
    return matches.slice(0, 4).map(m => m.name);
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

Generate FIXED JavaScript code using Arquero (loaded as 'aq'). Available datasets: ${Object.keys(deps.datasets).join(', ')}.

Common pitfalls to avoid:
- Don't use pandas syntax (no .iloc, no pd, no Python f-strings)
- Use Arquero verbs: .filter(), .groupby(), .count(), .orderby(), .limit(), .rollup(), .join()
- Access columns as d.columnName or d['column name with spaces']
- Store result in a variable named result
- Return small JSON-safe results (objects, arrays of objects, or primitives)

Return JSON: {"code": "...", "explanation": "..."}`;
  }

  /**
   * Execute generated code in a sandboxed environment.
   * @throws CodeExecutionError for syntax errors, runtime errors, timeouts
   */
  function execute(code: string): unknown {
    const aq = deps.datasets.get('aq') as { op?: Record<string, unknown> };
    const datasetsObj: Record<string, unknown> = {};
    for (const [name, table] of deps.datasets) {
      if (name === 'aq') continue;
      datasetsObj[name] = table;
    }
    const op = aq?.op || {};
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
      const result = fn(...argValues, aq, op);
      // Convert Arquero table to array of objects if needed
      if (result && typeof result === 'object' && typeof (result as { objects?: () => unknown[] }).objects === 'function') {
        return (result as { objects: () => unknown[] }).objects();
      }
      return result;
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));

      // Classify error type for retry logic
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
          retryable: !isSyntaxError && !isTimeout, // Retry runtime errors, not syntax or timeout
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
    const data = rows.slice(0, 12) as Array<Record<string, unknown>>;
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
      question: code,
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

  async function analyze(question: string, signal?: AbortSignal): Promise<DataAnalysisResult> {
    const start = performance.now();
    const relevant = relevantDatasets(question);
    const maxAttempts = 2;
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
        const result = execute(code);
        return formatResult(result, code, parsed.explanation, attempt + 1, start);
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
      } catch {
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