/**
 * Analyzer salvage — error recovery and fallback synthesis paths.
 * Extracted from analyzerToolLoop.ts for modularity.
 */

import type { LLMClient } from '../lib/llm';
import type { LLMMessage, ChartConfig, Insight } from '../lib/types';
import { buildSystemPrompt } from './analyzerPrompts';
import { SALVAGE_MAX_COMPLETION_TOKENS } from './analyzerBudget';

const VALID_CHART_TYPES = new Set(['bar', 'line', 'pie'] as const);

export class FallbackTriggered extends Error {
  constructor(public readonly reason: 'provider-rejected-tools' | 'no-first-tool-call' | 'malformed-tool-calls') {
    super(`Fallback triggered: ${reason}`);
    this.name = 'FallbackTriggered';
  }
}

/**
 * Parse a synthesis response into {answer, insights, chart}. Strips markdown
 * fences when present; unparseable content is treated as the whole answer.
 */
export function parseSynthesisJson(content: string): { answer: string; insights?: Insight[]; chart?: ChartConfig } {
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
export function normalizeInsightConfidence(insights: Insight[] | undefined): Insight[] {
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
export function isValidChartConfig(chart: ChartConfig | undefined | null): chart is ChartConfig {
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

export interface SalvageDeps {
  llm: LLMClient;
  codeGenModel?: string;
}

export interface SalvageOptions {
  signal?: AbortSignal;
  hooks?: {
    onSynthesisToken?: (token: string) => void;
  };
}

export async function runSalvageSynthesis(
  messages: LLMMessage[],
  deps: SalvageDeps,
  options: SalvageOptions,
  fallbackReason: string,
  toolTraceMap: Map<string, { calls: number; durationMs: number; tokensUsed: number }>,
  iterations: number,
  tokensUsed: number
): Promise<{
  answer: string;
  insights: Insight[];
  chart: ChartConfig | undefined;
  tokensUsed: number;
  iterations: number;
  toolTrace: { tool: string; calls: number; durationMs: number; tokensUsed: number }[];
  partial: boolean;
  fallbackReason: string;
}> {
  let salvageContent = '';
  const salvageResp = await deps.llm.stream({
    system: buildSystemPrompt(),
    messages,
    temperature: 0,
    model: deps.codeGenModel,
    maxTokens: SALVAGE_MAX_COMPLETION_TOKENS,
  }, (token: string) => {
    salvageContent += token;
    options.hooks?.onSynthesisToken?.(token);
  }, options.signal);

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
    toolTrace: Array.from(toolTraceMap.entries()).map(([tool, data]) => ({
      tool,
      calls: data.calls,
      durationMs: data.durationMs,
      tokensUsed: data.tokensUsed,
    })),
    partial: true,
    fallbackReason,
  };
}