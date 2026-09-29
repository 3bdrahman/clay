/**
 * Analyzer tool executor — per-iteration tool call validation, execution, and tracing.
 * Extracted from analyzerToolLoop.ts for modularity.
 */

import type { AnalysisToolContext } from '../lib/analysisTools';
import { executeToolCall } from '../lib/analysisTools';
import { MAX_TOOL_RESULT_CHARS, ARGS_SUMMARY_MAX_CHARS } from './analyzerBudget';

export interface ToolTraceEntry {
  tool: string;
  calls: number;
  durationMs: number;
  tokensUsed: number;
}

interface ToolTraceAcc {
  calls: number;
  durationMs: number;
  tokensUsed: number;
}

export function buildToolTrace(toolTraceMap: Map<string, ToolTraceAcc>): ToolTraceEntry[] {
  return Array.from(toolTraceMap.entries()).map(([tool, data]) => ({
    tool,
    calls: data.calls,
    durationMs: data.durationMs,
    tokensUsed: data.tokensUsed,
  }));
}

export function recordToolTrace(
  toolTraceMap: Map<string, ToolTraceAcc>,
  toolName: string,
  durationMs: number,
  tokensPerCall: number
): void {
  const trace = toolTraceMap.get(toolName) || { calls: 0, durationMs: 0, tokensUsed: 0 };
  trace.calls += 1;
  trace.durationMs += durationMs;
  trace.tokensUsed += tokensPerCall;
  toolTraceMap.set(toolName, trace);
}

export function summarizeArgs(args: Record<string, unknown>): string {
  const json = JSON.stringify(args);
  if (json.length <= ARGS_SUMMARY_MAX_CHARS) return json;
  return json.slice(0, ARGS_SUMMARY_MAX_CHARS) + '…';
}

export function truncateResult(content: string): string {
  if (content.length <= MAX_TOOL_RESULT_CHARS) return content;
  return content.slice(0, MAX_TOOL_RESULT_CHARS) + '… [truncated]';
}

export interface ToolExecutionResult {
  toolResult: unknown;
  toolError: string | undefined;
  durationMs: number;
}

export interface ExecuteToolCallDeps {
  datasets: Map<string, unknown>;
  metadata: { [datasetName: string]: { columns: string[]; rowCount: number } };
  executeUserCode: (code: string) => Promise<unknown>;
}

export async function executeToolCallWithTracing(
  deps: ExecuteToolCallDeps,
  toolName: string,
  args: Record<string, unknown>,
  tokensPerCall: number,
  toolTraceMap: Map<string, ToolTraceAcc>,
  hooks?: {
    onToolStart?: (info: { tool: string; argsSummary: string; startedAt: number }) => void;
    onToolEnd?: (info: { tool: string; durationMs: number; error?: string }) => void;
  }
): Promise<ToolExecutionResult> {
  const toolStart = Date.now();
  let toolResult: unknown;
  let toolError: string | undefined;

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

  return { toolResult, toolError, durationMs };
}

export function validateToolCallArguments(call: { function: { arguments?: string; name?: string } }): {
  args: Record<string, unknown>;
  toolName: string;
  error?: string;
} {
  let args: Record<string, unknown> = {};
  try {
    args = JSON.parse(call.function.arguments || '{}');
  } catch {
    return { args: {}, toolName: '', error: 'malformed tool call: invalid JSON arguments' };
  }

  const toolName = call.function.name;
  if (!toolName || typeof toolName !== 'string') {
    return { args, toolName: '', error: 'malformed tool call: missing function name' };
  }

  return { args, toolName };
}