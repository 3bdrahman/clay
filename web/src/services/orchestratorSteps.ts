// Orchestrator step functions — extracted from orchestrator.ts for modularity.
// Each step receives an OrchestratorStepContext carrying the shared workflow
// state, deps, callbacks, and the step-tracking helpers owned by the
// orchestrator closure.

import type {
  Citation,
  DataAnalysisResult,
  Document,
  SourceType,
  StepTrace,
  WorkflowState,
} from '../lib/types';
import type { AnalyzerHooks } from './analyzer';
import type { OrchestratorDeps, WorkflowCallbacks } from './orchestrator';
import {
  expandHyDE,
  parallelFanOut,
  parallelGrade,
  formatHeadingCitation,
  type HyDEResult,
} from './orchestratorHelpers';
import {
  DOC_GRADER_INSTRUCTIONS,
  DOC_GRADER_PROMPT,
  RAG_PROMPT,
  HALLUCINATION_INSTRUCTIONS,
  HALLUCINATION_PROMPT,
  ANSWER_INSTRUCTIONS,
  ANSWER_PROMPT,
} from './orchestratorPrompts';
import { getUserMessage } from '../lib/errors';

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

const DEFAULT_VECTORSTORE_INITIAL_K = 8;
const RERANK_K = 4;
const GRADE_EARLY_EXIT_AT = 4;
const WEB_SEARCH_RESULT_COUNT = 4;
const MAX_REFLECTIONS = 8;
const MAX_REWRITE_CHARS = 500;

export const EVAL_TEMPERATURE = 0;

function formatPreviousContext(analysis: DataAnalysisResult): string {
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
            .replace('{document}', doc.content.slice(0, 1500))
            .replace('{question}', ctx.question),
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

export async function runVectorstorePath(
  ctx: OrchestratorStepContext,
  signal?: AbortSignal,
  questionOverride?: string,
): Promise<void> {
  if (signal?.aborted) return;
  ctx.beginStep('retrieve', NODE_LABELS.retrieve);

  const effectiveQuestion = questionOverride ?? ctx.question;

  try {
    const hydeResult: HyDEResult = await expandHyDE(effectiveQuestion, { llm: ctx.deps.llm, model: ctx.deps.pickedModels.chat });
    const initialK = ctx.deps.settings.vectorstoreInitialK ?? DEFAULT_VECTORSTORE_INITIAL_K;
    const rerankK = RERANK_K;

    const docs = await ctx.withRetry('vectorstore-similaritySearch', () =>
      parallelFanOut(
        (q, k) => ctx.deps.vectorstore.similaritySearch(q, k),
        effectiveQuestion,
        hydeResult.hypothetical,
        initialK,
        rerankK,
      ),
      signal
    );
    ctx.state.documents = docs;
    ctx.endStep('retrieve', { detail: `${docs.length} docs`, meta: { count: docs.length, tokensUsed: hydeResult.tokensUsed } });

    ctx.beginStep('grade_docs', NODE_LABELS.grade_docs);
    const filtered = await parallelGrade(docs, (doc) => gradeDocRelevance(ctx, doc), {
      earlyExitAt: GRADE_EARLY_EXIT_AT,
      signal,
    });
    ctx.state.documents = filtered;
    ctx.endStep('grade_docs', { detail: `${filtered.length}/${docs.length} relevant` });
  } catch (e) {
    ctx.endStep('retrieve', { status: 'error', detail: getUserMessage(e) });
    ctx.endStep('grade_docs', { status: 'error' });
    throw e;
  }
}

export async function runPythonPath(
  ctx: OrchestratorStepContext,
  signal?: AbortSignal,
  questionOverride?: string,
): Promise<void> {
  if (signal?.aborted) return;
  ctx.beginStep('analyze', NODE_LABELS.analyze);
  try {
    const reflections: string[] = [];
    let plan: string | undefined;
    const hooks: AnalyzerHooks = {
      onToolStart: (info) => {
        ctx.steps.push({
          id: `analyze:${info.tool}-${crypto.randomUUID()}`,
          node: `analyze:${info.tool}`,
          label: info.tool,
          status: 'running',
          startedAt: info.startedAt,
        });
        ctx.emitSteps();
      },
      onToolEnd: (info) => {
        for (let i = ctx.steps.length - 1; i >= 0; i--) {
          const step = ctx.steps[i]!;
          if (step.node === `analyze:${info.tool}` && step.status === 'running') {
            step.status = info.error ? 'error' : 'done';
            step.finishedAt = Date.now();
            step.durationMs = info.durationMs;
            if (info.error) step.detail = info.error;
            break;
          }
        }
        ctx.emitSteps();
      },
      onIteration: (info) => {
        const text = info.reflection.trim();
        if (text.startsWith('PLAN:')) {
          plan = text.slice(5).trim();
        } else if (text.startsWith('REFLECTION:')) {
          if (reflections.length < MAX_REFLECTIONS) {
            reflections.push(text.slice(11).trim());
          }
        } else {
          if (reflections.length < MAX_REFLECTIONS) {
            reflections.push(text);
          }
        }
      },
      onSynthesisToken: (token) => {
        ctx.callbacks.onToken?.(token);
      },
    };
    const previousContext = ctx.deps.previousAnalysis
      ? formatPreviousContext(ctx.deps.previousAnalysis)
      : undefined;
    const effectiveQuestion = questionOverride ?? ctx.question;
    const result = await ctx.withRetry('analyzer-analyze', () => ctx.deps.analyzer.analyze(effectiveQuestion, signal, hooks, previousContext), signal);
    ctx.state.dataAnalysis = result;
    ctx.endStep('analyze', {
      detail: result.fallbackReason
        ? `fallback: ${result.fallbackReason}`
        : result.resultType === 'error'
        ? 'error'
        : 'complete',
      meta: {
        mode: result.mode,
        fallbackReason: result.fallbackReason,
        toolCount: result.toolTrace?.length ?? 0,
        iterations: result.attempts,
        durationMs: Math.round(result.durationMs),
        plan,
        reflections: reflections.length > 0 ? reflections : undefined,
        partial: result.partial,
      },
    });
  } catch (e) {
    ctx.endStep('analyze', { status: 'error', detail: getUserMessage(e) });
    throw e;
  }
}

export async function runWebSearchStep(
  ctx: OrchestratorStepContext,
  signal?: AbortSignal,
  questionOverride?: string,
): Promise<void> {
  if (signal?.aborted) return;
  ctx.beginStep('web_search', NODE_LABELS.web_search);
  try {
    const effectiveQuestion = questionOverride ?? ctx.question;
    const results = await ctx.withRetry('webSearch-search', () => ctx.deps.webSearch.search(effectiveQuestion, WEB_SEARCH_RESULT_COUNT), signal);
    ctx.state.webResults = results;
    ctx.endStep('web_search', { detail: `${results.length} results` });
  } catch (e) {
    ctx.endStep('web_search', { status: 'error', detail: getUserMessage(e) });
    throw e;
  }
}

export function clearSourceData(ctx: OrchestratorStepContext, source: SourceType): void {
  if (source === 'vectorstore') ctx.state.documents = [];
  else if (source === 'python') ctx.state.dataAnalysis = undefined;
  else if (source === 'websearch') ctx.state.webResults = [];
}

export async function rewriteQuestionForSource(
  ctx: OrchestratorStepContext,
  originalQuestion: string,
  source: SourceType,
  signal?: AbortSignal,
): Promise<string> {
  const prompt = `Rewrite this question to maximize retrieval effectiveness for ${source} search. Return JSON {question: string}. Keep the same information need.

Original question: ${originalQuestion}`;

  try {
    const resp = await ctx.withRetry('llm-invoke-rewrite', () =>
      ctx.deps.llm.invoke({
        system: 'You are a query rewriter that optimizes questions for specific retrieval sources. Return only valid JSON.',
        messages: [{ role: 'user', content: prompt }],
        jsonMode: true,
        temperature: 0,
        model: ctx.deps.pickedModels.chat,
      }),
      signal
    );
    const parsed = JSON.parse(resp.content || '{}');
    const rewritten = (parsed.question || '').trim();
    if (rewritten && rewritten.length > 0) {
      return rewritten.slice(0, MAX_REWRITE_CHARS);
    }
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[orchestrator] Query rewrite failed, using original question:', e);
    }
  }
  return originalQuestion;
}

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
  const prompt = RAG_PROMPT.replace('{context}', context).replace('{question}', ctx.question);

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
            content: HALLUCINATION_PROMPT.replace('{documents}', context).replace('{generation}', ctx.state.answer || ''),
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
            content: ANSWER_PROMPT.replace('{question}', ctx.question).replace('{generation}', ctx.state.answer || ''),
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
    ctx.endStep('evaluate', { detail: 'eval-error', meta: { tokensUsed: totalTokensUsed } });
    return true;
  }
}
