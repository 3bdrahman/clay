/**
 * Analyzer tool loop — shared type definitions.
 * Extracted from analyzerToolLoop.ts for modularity.
 */

import type { LLMClient } from '../lib/llm';
import type { ChartConfig, Insight } from '../lib/types';

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