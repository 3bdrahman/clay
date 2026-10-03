// Path-related step helpers — extracted from orchestratorSteps.ts for modularity.
// Each path function handles one retrieval source: vectorstore, python (data analysis), or websearch.

import type { OrchestratorStepContext } from './orchestratorCore';
import {
  NODE_LABELS,
  DEFAULT_VECTORSTORE_INITIAL_K,
  RERANK_K,
  GRADE_EARLY_EXIT_AT,
  WEB_SEARCH_RESULT_COUNT,
  MAX_REFLECTIONS,
  MAX_REWRITE_CHARS,
  gradeDocRelevance,
  formatPreviousContext,
  type SourceType,
} from './orchestratorCore';
import {
  expandHyDE,
  parallelFanOut,
  parallelGrade,
  type HyDEResult,
} from './orchestratorHelpers';
import type { AnalyzerHooks } from './analyzer';
import { getUserMessage } from '../lib/errors';

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
    const results = await ctx.withRetry('webSearch-search', () => ctx.deps.webSearch.search(effectiveQuestion, WEB_SEARCH_RESULT_COUNT, signal), signal);
    ctx.state.webResults = results;
    ctx.endStep('web_search', { detail: `${results.length} results` });
  } catch (e) {
    ctx.endStep('web_search', { status: 'error', detail: getUserMessage(e) });
    throw e;
  }
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
