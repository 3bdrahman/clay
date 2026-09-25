// Data analyzer — replaces Python exec() with safe in-browser code generation
// Generates Arquero-compatible code (a pandas-like dataframe library)

import type { DataAnalysisResult, DatasetSummary } from '../lib/types';
import type { LLMClient } from '../lib/llm';
import { createUserCodeExecutor } from './analyzerSandbox';
import { createSingleShotAnalyzer } from './analyzerSingleShot';
import { runToolLoop, FallbackTriggered } from './analyzerToolLoop';

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
  onSynthesisToken?: (token: string) => void;
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

const NAME_TOKEN_MATCH_SCORE = 4;
const COLUMN_TOKEN_MATCH_SCORE = 2;
const MAX_RELEVANT_DATASETS = 4;

/**
 * Create a data analyzer that generates and executes Arquero code for CSV analysis.
 * Uses LLM to generate JavaScript, executes safely via new Function(), detects chart config.
 * @param deps - LLM client, dataset Map, metadata, optional codeGenModel, optional maxToolLoopTokens
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

  const executeUserCode = createUserCodeExecutor(deps.datasets);
  const runSingleShot = createSingleShotAnalyzer({
    llm,
    datasets: deps.datasets,
    metadata,
    codeGenModel: deps.codeGenModel,
    relevantDatasets,
    executeUserCode,
  });

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
      const loopResult = await runToolLoop(question, relevant, {
        llm,
        datasets: deps.datasets,
        metadata,
        codeGenModel: deps.codeGenModel,
        maxToolLoopTokens: deps.maxToolLoopTokens,
        executeUserCode,
      }, {
        signal,
        hooks,
        previousContext,
      });

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
        const singleShotResult = await runSingleShot(question, signal);
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
