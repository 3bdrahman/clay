// Core orchestrator types, constants, and shared utilities — shared by orchestratorSteps and pathSteps.
// Extracted from orchestratorSteps.ts for modularity.

import type {
  DataAnalysisResult,
  Document,
  SourceType,
  StepTrace,
  WorkflowState,
} from '../lib/types';
import type { OrchestratorDeps, WorkflowCallbacks } from './orchestrator';
import {
  DOC_GRADER_INSTRUCTIONS,
  DOC_GRADER_PROMPT,
} from './orchestratorPrompts';

export interface OrchestratorStepContext {
  question: string;
  deps: OrchestratorDeps;
  callbacks: WorkflowCallbacks;
  state: WorkflowState;
  steps: StepTrace[];
  beginStep: (node: string, label: string) => void;
  endStep: (node: string, opts?: { status?: 'done' | 'error'; detail?: string; meta?: Record<string, unknown> }) => void;
  emitSteps: () => void;
  withRetry: <T>(stepName: string, fn: () => Promise<T>, signal?: AbortSignal) => Promise<T>;
}

export const NODE_LABELS: Record<string, string> = {
  start: 'Start',
  route: 'Routing Question',
  retrieve: 'Accessing Vector DB',
  grade_docs: 'Grading Documents',
  decide: 'Decide Source',
  analyze: 'Analyzing Data',
  web_search: 'Web Search',
  generate: 'Generating Answer',
  evaluate: 'Evaluating Quality',
  end: 'Done',
};

export const EVAL_TEMPERATURE = 0;

const DEFAULT_VECTORSTORE_INITIAL_K = 8;
const RERANK_K = 4;
const GRADE_EARLY_EXIT_AT = 4;
const WEB_SEARCH_RESULT_COUNT = 4;
const MAX_REFLECTIONS = 8;
const MAX_REWRITE_CHARS = 500;

export {
  DEFAULT_VECTORSTORE_INITIAL_K,
  RERANK_K,
  GRADE_EARLY_EXIT_AT,
  WEB_SEARCH_RESULT_COUNT,
  MAX_REFLECTIONS,
  MAX_REWRITE_CHARS,
};

export type { SourceType };

export async function gradeDocRelevance(
  ctx: OrchestratorStepContext,
  doc: Document,
): Promise<'relevant' | 'irrelevant' | 'keep-on-error'> {
  try {
    const resp = await ctx.deps.llm.invoke({
      system: DOC_GRADER_INSTRUCTIONS,
      messages: [
        {
          role: 'user',
          content: DOC_GRADER_PROMPT
            // Function replacement: a document containing `$&`/`` $` ``/`$'`
            // must not be interpreted as a replacement pattern.
            .replace('{document}', () => doc.content.slice(0, 1500))
            .replace('{question}', () => ctx.question),
        },
      ],
      jsonMode: true,
      temperature: EVAL_TEMPERATURE,
      model: ctx.deps.pickedModels.chat,
    });
    try {
      const parsed = JSON.parse(resp.content || '{}');
      return (parsed.binary_score || '').toLowerCase() === 'yes' ? 'relevant' : 'irrelevant';
    } catch (e) {
      if (import.meta.env.DEV) {
        console.warn('[orchestrator] gradeDocRelevance JSON parse failed (marking "irrelevant"):', e);
      }
      return 'irrelevant';
    }
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[orchestrator] gradeDocRelevance LLM invoke failed (marking "keep-on-error"):', e);
    }
    return 'keep-on-error';
  }
}

export function formatPreviousContext(analysis: DataAnalysisResult): string {
  const parts: string[] = [];
  if (analysis.insights && analysis.insights.length > 0) {
    parts.push('Insights:');
    for (const insight of analysis.insights) {
      parts.push(`- ${insight.finding} (${insight.confidence}): ${insight.evidence}`);
    }
  }
  if (analysis.toolTrace && analysis.toolTrace.length > 0) {
    parts.push('Tools used:');
    for (const tool of analysis.toolTrace) {
      parts.push(`- ${tool.tool} (${tool.calls} calls, ${Math.round(tool.durationMs)}ms)`);
    }
  }
  if (analysis.explanation) {
    parts.push(`Summary: ${analysis.explanation.slice(0, 500)}`);
  }
  return parts.join('\n');
}