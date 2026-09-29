// Orchestrator step functions — entry point re-exporting the public surface.
// Core types and shared utilities are in orchestratorCore.ts.
// Path-related step helpers are in pathSteps.ts.

// OrchestratorStepContext and SourceType are type-only: value re-exports of
// type-only names pass tsc but fail rolldown's worker bundling (MISSING_EXPORT).
export type {
  OrchestratorStepContext,
  SourceType,
} from './orchestratorCore';

export {
  NODE_LABELS,
  EVAL_TEMPERATURE,
  DEFAULT_VECTORSTORE_INITIAL_K,
  RERANK_K,
  GRADE_EARLY_EXIT_AT,
  WEB_SEARCH_RESULT_COUNT,
  MAX_REFLECTIONS,
  MAX_REWRITE_CHARS,
  gradeDocRelevance,
  formatPreviousContext,
} from './orchestratorCore';

export {
  runVectorstorePath,
  runPythonPath,
  runWebSearchStep,
  rewriteQuestionForSource,
} from './pathSteps';

import type { OrchestratorStepContext } from './orchestratorCore';
import {
  NODE_LABELS,
  EVAL_TEMPERATURE,
} from './orchestratorCore';
import {
  runVectorstorePath,
  runPythonPath,
  runWebSearchStep,
} from './pathSteps';
import {
  RAG_PROMPT,
  HALLUCINATION_INSTRUCTIONS,
  HALLUCINATION_PROMPT,
  ANSWER_INSTRUCTIONS,
  ANSWER_PROMPT,
} from './orchestratorPrompts';
import { formatHeadingCitation } from './orchestratorHelpers';
import { getUserMessage } from '../lib/errors';
import type { Citation, SourceType } from '../lib/types';

export async function runPath(
  ctx: OrchestratorStepContext,
  source: SourceType,
  signal?: AbortSignal,
  questionOverride?: string,
): Promise<void> {
  switch (source) {
    case 'vectorstore':
      await runVectorstorePath(ctx, signal, questionOverride);
      break;
    case 'python':
      await runPythonPath(ctx, signal, questionOverride);
      break;
    case 'websearch':
      await runWebSearchStep(ctx, signal, questionOverride);
      break;
  }
}

function buildContextForEval(ctx: OrchestratorStepContext): string {
  const parts: string[] = [];
  if (ctx.state.documents.length > 0) parts.push(ctx.state.documents.map(d => d.content).join('\n---\n'));
  if (ctx.state.webResults.length > 0) parts.push(ctx.state.webResults.map(w => w.content).join('\n---\n'));
  if (ctx.state.dataAnalysis && ctx.state.dataAnalysis.resultType !== 'error') {
    parts.push(JSON.stringify(ctx.state.dataAnalysis.result));
  }
  return parts.join('\n\n');
}

function buildCitations(ctx: OrchestratorStepContext): void {
  const citations: Citation[] = [];
  if (ctx.state.documents.length > 0) {
    for (const d of ctx.state.documents) {
      citations.push(formatHeadingCitation(d));
    }
  }
  if (ctx.state.webResults.length > 0) {
    for (const w of ctx.state.webResults) {
      citations.push({ source: w.title, excerpt: w.content.slice(0, 200), type: 'websearch' });
    }
  }
  if (ctx.state.dataAnalysis && ctx.state.dataAnalysis.resultType !== 'error') {
    citations.push({ source: 'Data Analysis', excerpt: ctx.state.dataAnalysis.explanation, type: 'python' });
  }
  ctx.state.citations = citations;
}

export async function generate(ctx: OrchestratorStepContext): Promise<void> {
  ctx.beginStep('generate', NODE_LABELS.generate);
  const sections: string[] = [];

  if (ctx.state.documents.length > 0) {
    const docText = ctx.state.documents
      .map(d => `[${d.source}${d.page ? ' p.' + d.page : ''}]\n${d.content}`)
      .join('\n\n---\n\n');
    sections.push(`DOCUMENTS:\n${docText}`);
  }
  if (ctx.state.dataAnalysis && ctx.state.dataAnalysis.resultType !== 'error') {
    const da = ctx.state.dataAnalysis;
    let dataAnalysisSection = `DATA ANALYSIS:\n` +
      `Question: ${ctx.question}\n` +
      `Code:\n\`\`\`js\n${da.code}\n\`\`\`\n` +
      `Result: ${JSON.stringify(da.result, null, 2)}\n` +
      `Explanation: ${da.explanation}`;

    if (da.insights && da.insights.length > 0) {
      const insightsText = da.insights
        .map(
          (insight) =>
            `- ${insight.finding} — ${insight.evidence} (confidence: ${insight.confidence})${insight.implication ? ` — ${insight.implication}` : ''}`
        )
        .join('\n');
      dataAnalysisSection += `\nInsights:\n${insightsText}`;
    }

    if (da.toolTrace && da.toolTrace.length > 0) {
      const toolsText = da.toolTrace
        .map((t) => `${t.tool} x${t.calls} (${Math.round(t.durationMs)}ms)`)
        .join(', ');
      dataAnalysisSection += `\nTools used: ${toolsText}`;
    }

    sections.push(dataAnalysisSection);
  }
  if (ctx.state.webResults.length > 0) {
    const webText = ctx.state.webResults
      .map(r => `[${r.title}]\n${r.content}`)
      .join('\n\n---\n\n');
    sections.push(`WEB SEARCH RESULTS:\n${webText}`);
  }
  if (sections.length === 0) {
    ctx.state.answer = "I couldn't find relevant information to answer your question.";
    ctx.endStep('generate', { detail: 'no context' });
    return;
  }

  const context = sections.join('\n\n============\n\n');
  const prompt = RAG_PROMPT.replace('{context}', () => context).replace('{question}', () => ctx.question);

  try {
    const resp = await ctx.withRetry('llm-stream', () =>
      ctx.deps.llm.stream(
        {
          system: 'You are a helpful assistant that answers questions based solely on the provided context. Cite sources inline using [1], [2], etc., and include a References: section at the end.',
          messages: [{ role: 'user', content: prompt }],
          temperature: ctx.deps.settings.temperature ?? 0,
          model: ctx.deps.pickedModels.chat,
        },
        ctx.callbacks.onToken ?? (() => {}),
      ),
    );
    ctx.state.answer = resp.content;
    buildCitations(ctx);
    ctx.endStep('generate', { detail: 'complete', meta: { tokensUsed: resp.usage?.totalTokens ?? 0 } });
  } catch (e) {
    const err = e instanceof Error ? e : new Error(String(e));
    ctx.state.answer = `Error generating answer: ${getUserMessage(err)}`;
    ctx.endStep('generate', { status: 'error', detail: getUserMessage(err) });
    throw err;
  }
}

export async function evaluate(ctx: OrchestratorStepContext): Promise<boolean> {
  ctx.beginStep('evaluate', NODE_LABELS.evaluate);
  const context = buildContextForEval(ctx);
  if (!context.trim()) {
    ctx.endStep('evaluate', { detail: 'no context' });
    return false;
  }

  let totalTokensUsed = 0;
  try {
    const halluc = await ctx.withRetry('llm-invoke-hallucination', () =>
      ctx.deps.llm.invoke({
        system: HALLUCINATION_INSTRUCTIONS,
        messages: [
          {
            role: 'user',
            content: HALLUCINATION_PROMPT
              .replace('{documents}', () => context)
              .replace('{generation}', () => ctx.state.answer || ''),
          },
        ],
        jsonMode: true,
        temperature: EVAL_TEMPERATURE,
        model: ctx.deps.pickedModels.chat,
      }),
    );
    totalTokensUsed += halluc.usage?.totalTokens ?? 0;
    const hallucParsed = JSON.parse(halluc.content || '{}');
    if ((hallucParsed.binary_score || '').toLowerCase() !== 'yes') {
      ctx.endStep('evaluate', { detail: 'hallucination', meta: { tokensUsed: totalTokensUsed } });
      return false;
    }

    const ansResp = await ctx.withRetry('llm-invoke-answer', () =>
      ctx.deps.llm.invoke({
        system: ANSWER_INSTRUCTIONS,
        messages: [
          {
            role: 'user',
            content: ANSWER_PROMPT
              .replace('{question}', () => ctx.question)
              .replace('{generation}', () => ctx.state.answer || ''),
          },
        ],
        jsonMode: true,
        temperature: EVAL_TEMPERATURE,
        model: ctx.deps.pickedModels.chat,
      }),
    );
    totalTokensUsed += ansResp.usage?.totalTokens ?? 0;
    const ansParsed = JSON.parse(ansResp.content || '{}');
    const useful = (ansParsed.binary_score || '').toLowerCase() === 'yes';
    ctx.endStep('evaluate', { detail: useful ? 'useful' : 'not useful', meta: { tokensUsed: totalTokensUsed } });
    return useful;
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[orchestrator] evaluate failed (treating as useful to avoid infinite retry):', e);
    }
    // The quality gate silently disabled on eval failure — surface both the
    // status and the reason in the trace so a degraded answer is visible.
    ctx.endStep('evaluate', {
      status: 'error',
      detail: `eval-error: ${getUserMessage(e)}`,
      meta: { tokensUsed: totalTokensUsed },
    });
    return true;
  }
}