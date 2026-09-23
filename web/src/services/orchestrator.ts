import type {
  Citation,
  DataAnalysisResult,
  Document,
  Settings,
  SourceType,
  StepTrace,
  WorkflowState,
} from '../lib/types';
import type { LLMClient } from '../lib/llm';
import type { PickedModels } from '../lib/models';
import type { VectorStore } from '../lib/vectorstore';
import type { WebSearchClient } from '../lib/websearch';
import type { DataAnalyzer, AnalyzerHooks } from './analyzer';
import {
  expandHyDE,
  parallelFanOut,
  parallelGrade,
  formatHeadingCitation,
  type HyDEResult,
} from './orchestratorHelpers';
import { RagError, RagErrorCode, isRetryable, getUserMessage, GenerationFailedError } from '../lib/errors';

export interface WorkflowOrchestrator {
  run(signal?: AbortSignal): Promise<WorkflowState>;
}

export interface WorkflowCallbacks {
  onStepUpdate?: (steps: StepTrace[]) => void;
  onPartialUpdate?: (state: WorkflowState) => void;
  onError?: (err: Error) => void;
  onToken?: (token: string) => void;
}

export interface OrchestratorDeps {
  llm: LLMClient;
  vectorstore: VectorStore;
  webSearch: WebSearchClient;
  analyzer: DataAnalyzer;
  settings: Settings;
  pickedModels: PickedModels;
  previousAnalysis?: DataAnalysisResult;
}

const NODE_LABELS: Record<string, string> = {
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

const VALID_SOURCE_TYPES: SourceType[] = ['vectorstore', 'python', 'websearch'];

const MAX_STEP_RETRIES = 2;
const BASE_RETRY_DELAY_MS = 1000;
const DEFAULT_VECTORSTORE_INITIAL_K = 8;
const RERANK_K = 4;
const GRADE_EARLY_EXIT_AT = 4;
const WEB_SEARCH_RESULT_COUNT = 4;
const EVAL_TEMPERATURE = 0;
const MAX_REFLECTIONS = 8;
const MAX_REWRITE_CHARS = 500;
const ROUTER_CONFIDENCE_THRESHOLD = 0.6;

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

export function createWorkflowOrchestrator(
  question: string,
  deps: OrchestratorDeps,
  callbacks: WorkflowCallbacks
): WorkflowOrchestrator {
  const state: WorkflowState = {
    question,
    documents: [],
    webResults: [],
    citations: [],
    retryCount: 0,
    steps: [],
    startedAt: Date.now(),
  };
  let steps: StepTrace[] = [];
  let consecutiveProviderFailures = 0;
  const MAX_CONSECUTIVE_FAILURES = 3;

  function beginStep(node: string, label: string): void {
    const step: StepTrace = {
      id: `${node}-${crypto.randomUUID()}`,
      node,
      label,
      status: 'running',
      startedAt: Date.now(),
    };
    steps.push(step);
    emitSteps();
  }

  function endStep(node: string, opts: { detail?: string; meta?: Record<string, unknown> } = {}): void {
    for (let i = steps.length - 1; i >= 0; i--) {
      const step = steps[i];
      if (step.node === node && step.status === 'running') {
        step.status = opts.detail === 'error' ? 'error' : 'done';
        step.finishedAt = Date.now();
        step.durationMs = step.finishedAt - (step.startedAt || step.finishedAt);
        if (opts.detail) step.detail = opts.detail;
        if (opts.meta) step.meta = opts.meta;
        emitSteps();
        return;
      }
    }
  }

  function setError(err: Error, step: string): void {
    const message = getUserMessage(err);
    const code = err instanceof RagError ? err.code : RagErrorCode.UNKNOWN_ERROR;
    const retryable = isRetryable(err);

    // Circuit breaker: track consecutive provider failures
    if (code === RagErrorCode.PROVIDER_UNREACHABLE || code === RagErrorCode.INVALID_API_KEY) {
      consecutiveProviderFailures++;
      if (consecutiveProviderFailures >= MAX_CONSECUTIVE_FAILURES) {
        // Circuit breaker triggered - don't retry
        throw new Error(`Circuit breaker triggered after ${consecutiveProviderFailures} consecutive provider failures. Check your configuration.`);
      }
    } else {
      // Reset counter on non-provider errors
      consecutiveProviderFailures = 0;
    }

    state.error = {
      code,
      message,
      step,
      retryable,
    };

    callbacks.onError?.(err);
    emitSteps();
  }

  let emitScheduled = false;
  function emitSteps(): void {
    state.steps = [...steps];
    callbacks.onStepUpdate?.(state.steps);
    if (!emitScheduled) {
      emitScheduled = true;
      requestAnimationFrame(() => {
        emitScheduled = false;
        callbacks.onPartialUpdate?.({ ...state });
      });
    }
  }

  async function withRetry<T>(
    stepName: string,
    fn: () => Promise<T>,
    signal?: AbortSignal
  ): Promise<T> {
    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= MAX_STEP_RETRIES; attempt++) {
      if (signal?.aborted) {
        throw new GenerationFailedError('orchestrator', new Error('Aborted'), { retryable: false });
      }
      try {
        return await fn();
      } catch (e) {
        lastError = e instanceof Error ? e : new Error(String(e));

        // Don't retry on abort or non-retryable errors
        if (signal?.aborted || !isRetryable(lastError)) {
          if (lastError instanceof RagError) {
            throw lastError.withStep(stepName);
          }
          throw new GenerationFailedError('orchestrator', lastError, {
            retryable: false,
          }).withStep(stepName);
        }

        // Retry with exponential backoff
        if (attempt < MAX_STEP_RETRIES) {
          const delay = BASE_RETRY_DELAY_MS * Math.pow(2, attempt);
          // Record retry event in the currently-running step (last step with status 'running')
          for (let i = steps.length - 1; i >= 0; i--) {
            const step = steps[i];
            if (step.status === 'running') {
              if (!step.retries) step.retries = [];
              step.retries.push({ attempt: attempt + 1, error: lastError.message, delayMs: delay });
              emitSteps();
              break;
            }
          }
          await new Promise(r => setTimeout(r, delay));
          if (import.meta.env.DEV) {
            console.warn(`[orchestrator] Retrying ${stepName} (attempt ${attempt + 1}/${MAX_STEP_RETRIES}):`, lastError.message);
          }
        }
      }
    }
    if (lastError instanceof RagError) {
      throw lastError.withStep(stepName);
    }
    throw new GenerationFailedError('orchestrator', lastError!, { retryable: false }).withStep(stepName);
  }

  function buildContextForEval(): string {
    const parts: string[] = [];
    if (state.documents.length > 0) parts.push(state.documents.map(d => d.content).join('\n---\n'));
    if (state.webResults.length > 0) parts.push(state.webResults.map(w => w.content).join('\n---\n'));
    if (state.dataAnalysis && state.dataAnalysis.resultType !== 'error') {
      parts.push(JSON.stringify(state.dataAnalysis.result));
    }
    return parts.join('\n\n');
  }

  function buildCitations(): void {
    const citations: Citation[] = [];
    if (state.documents.length > 0) {
      for (const d of state.documents) {
        citations.push(formatHeadingCitation(d));
      }
    }
    if (state.webResults.length > 0) {
      for (const w of state.webResults) {
        citations.push({ source: w.title, excerpt: w.content.slice(0, 200), type: 'websearch' });
      }
    }
    if (state.dataAnalysis && state.dataAnalysis.resultType !== 'error') {
      citations.push({ source: 'Data Analysis', excerpt: state.dataAnalysis.explanation, type: 'python' });
    }
    state.citations = citations;
  }

  async function gradeDocRelevance(doc: Document): Promise<'relevant' | 'irrelevant' | 'keep-on-error'> {
    try {
      const resp = await deps.llm.invoke({
        system: DOC_GRADER_INSTRUCTIONS,
        messages: [
          {
            role: 'user',
            content: DOC_GRADER_PROMPT
              .replace('{document}', doc.content.slice(0, 1500))
              .replace('{question}', question),
          },
        ],
        jsonMode: true,
        temperature: EVAL_TEMPERATURE,
        model: deps.pickedModels.chat,
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

  async function runVectorstorePath(signal?: AbortSignal, questionOverride?: string): Promise<void> {
    if (signal?.aborted) return;
    beginStep('retrieve', NODE_LABELS.retrieve);

    const effectiveQuestion = questionOverride ?? question;

    try {
      const hydeResult: HyDEResult = await expandHyDE(effectiveQuestion, { llm: deps.llm, model: deps.pickedModels.chat });
      const initialK = deps.settings.vectorstoreInitialK ?? DEFAULT_VECTORSTORE_INITIAL_K;
      const rerankK = RERANK_K;

      const docs = await withRetry('vectorstore-similaritySearch', () =>
        parallelFanOut(
          (q, k) => deps.vectorstore.similaritySearch(q, k),
          effectiveQuestion,
          hydeResult.hypothetical,
          initialK,
          rerankK,
        ),
        signal
      );
      state.documents = docs;
      endStep('retrieve', { detail: `${docs.length} docs`, meta: { count: docs.length, tokensUsed: hydeResult.tokensUsed } });

      beginStep('grade_docs', NODE_LABELS.grade_docs);
      const filtered = await parallelGrade(docs, gradeDocRelevance, {
        earlyExitAt: GRADE_EARLY_EXIT_AT,
        signal,
      });
      state.documents = filtered;
      endStep('grade_docs', { detail: `${filtered.length}/${docs.length} relevant` });
    } catch (e) {
      endStep('retrieve', { detail: 'error' });
      endStep('grade_docs', { detail: 'error' });
      throw e;
    }
  }

  async function runPythonPath(signal?: AbortSignal, questionOverride?: string): Promise<void> {
    if (signal?.aborted) return;
    beginStep('analyze', NODE_LABELS.analyze);
    try {
      const reflections: string[] = [];
      let plan: string | undefined;
      const hooks: AnalyzerHooks = {
        onToolStart: (info) => {
          steps.push({
            id: `analyze:${info.tool}-${crypto.randomUUID()}`,
            node: `analyze:${info.tool}`,
            label: info.tool,
            status: 'running',
            startedAt: info.startedAt,
          });
          emitSteps();
        },
        onToolEnd: (info) => {
          for (let i = steps.length - 1; i >= 0; i--) {
            const step = steps[i]!;
            if (step.node === `analyze:${info.tool}` && step.status === 'running') {
              step.status = info.error ? 'error' : 'done';
              step.finishedAt = Date.now();
              step.durationMs = info.durationMs;
              if (info.error) step.detail = info.error;
              break;
            }
          }
          emitSteps();
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
          callbacks.onToken?.(token);
        },
      };
      const previousContext = deps.previousAnalysis
        ? formatPreviousContext(deps.previousAnalysis)
        : undefined;
      const effectiveQuestion = questionOverride ?? question;
      const result = await withRetry('analyzer-analyze', () => deps.analyzer.analyze(effectiveQuestion, signal, hooks, previousContext), signal);
      state.dataAnalysis = result;
      endStep('analyze', {
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
      endStep('analyze', { detail: 'error' });
      throw e;
    }
  }

  async function runWebSearchStep(signal?: AbortSignal, questionOverride?: string): Promise<void> {
    if (signal?.aborted) return;
    beginStep('web_search', NODE_LABELS.web_search);
    try {
      const effectiveQuestion = questionOverride ?? question;
      const results = await withRetry('webSearch-search', () => deps.webSearch.search(effectiveQuestion, WEB_SEARCH_RESULT_COUNT), signal);
      state.webResults = results;
      endStep('web_search', { detail: `${results.length} results` });
    } catch (e) {
      endStep('web_search', { detail: 'error' });
      throw e;
    }
  }

  function clearSourceData(source: SourceType): void {
    if (source === 'vectorstore') state.documents = [];
    else if (source === 'python') state.dataAnalysis = undefined;
    else if (source === 'websearch') state.webResults = [];
  }

  async function rewriteQuestionForSource(originalQuestion: string, source: SourceType, signal?: AbortSignal): Promise<string> {
    const prompt = `Rewrite this question to maximize retrieval effectiveness for ${source} search. Return JSON {question: string}. Keep the same information need.

Original question: ${originalQuestion}`;

    try {
      const resp = await withRetry('llm-invoke-rewrite', () =>
        deps.llm.invoke({
          system: 'You are a query rewriter that optimizes questions for specific retrieval sources. Return only valid JSON.',
          messages: [{ role: 'user', content: prompt }],
          jsonMode: true,
          temperature: 0,
          model: deps.pickedModels.chat,
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

  async function runPath(source: SourceType, signal?: AbortSignal, questionOverride?: string): Promise<void> {
    if (signal?.aborted) return;
    switch (source) {
      case 'vectorstore':
        await runVectorstorePath(signal, questionOverride);
        break;
      case 'python':
        await runPythonPath(signal, questionOverride);
        break;
      case 'websearch':
        await runWebSearchStep(signal, questionOverride);
        break;
    }
  }

  async function generate(): Promise<void> {
    beginStep('generate', NODE_LABELS.generate);
    const sections: string[] = [];

    if (state.documents.length > 0) {
      const docText = state.documents
        .map(d => `[${d.source}${d.page ? ' p.' + d.page : ''}]\n${d.content}`)
        .join('\n\n---\n\n');
      sections.push(`DOCUMENTS:\n${docText}`);
    }
    if (state.dataAnalysis && state.dataAnalysis.resultType !== 'error') {
      const da = state.dataAnalysis;
      let dataAnalysisSection = `DATA ANALYSIS:\n` +
        `Question: ${question}\n` +
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
    if (state.webResults.length > 0) {
      const webText = state.webResults
        .map(r => `[${r.title}]\n${r.content}`)
        .join('\n\n---\n\n');
      sections.push(`WEB SEARCH RESULTS:\n${webText}`);
    }
    if (sections.length === 0) {
      state.answer = "I couldn't find relevant information to answer your question.";
      endStep('generate', { detail: 'no context' });
      return;
    }

    const context = sections.join('\n\n============\n\n');
    const prompt = RAG_PROMPT.replace('{context}', context).replace('{question}', question);

    try {
      const resp = await withRetry('llm-stream', () =>
        deps.llm.stream(
          {
            system: 'You are a helpful assistant that answers questions based solely on the provided context. Cite sources inline using [1], [2], etc., and include a References: section at the end.',
            messages: [{ role: 'user', content: prompt }],
            temperature: deps.settings.temperature ?? 0,
            model: deps.pickedModels.chat,
          },
          callbacks.onToken ?? (() => {}),
        ),
      );
      state.answer = resp.content;
      buildCitations();
      endStep('generate', { detail: 'complete', meta: { tokensUsed: resp.usage?.totalTokens ?? 0 } });
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      state.answer = `Error generating answer: ${getUserMessage(err)}`;
      endStep('generate', { detail: 'error' });
      throw err;
    }
  }

  async function evaluate(): Promise<boolean> {
    beginStep('evaluate', NODE_LABELS.evaluate);
    const context = buildContextForEval();
    if (!context.trim()) {
      endStep('evaluate', { detail: 'no context' });
      return false;
    }

    let totalTokensUsed = 0;
    try {
      const halluc = await withRetry('llm-invoke-hallucination', () =>
        deps.llm.invoke({
          system: HALLUCINATION_INSTRUCTIONS,
          messages: [
            {
              role: 'user',
              content: HALLUCINATION_PROMPT.replace('{documents}', context).replace('{generation}', state.answer || ''),
            },
          ],
          jsonMode: true,
          temperature: EVAL_TEMPERATURE,
          model: deps.pickedModels.chat,
        }),
      );
      totalTokensUsed += halluc.usage?.totalTokens ?? 0;
      const hallucParsed = JSON.parse(halluc.content || '{}');
      if ((hallucParsed.binary_score || '').toLowerCase() !== 'yes') {
        endStep('evaluate', { detail: 'hallucination', meta: { tokensUsed: totalTokensUsed } });
        return false;
      }

      const ansResp = await withRetry('llm-invoke-answer', () =>
        deps.llm.invoke({
          system: ANSWER_INSTRUCTIONS,
          messages: [
            {
              role: 'user',
              content: ANSWER_PROMPT.replace('{question}', question).replace('{generation}', state.answer || ''),
            },
          ],
          jsonMode: true,
          temperature: EVAL_TEMPERATURE,
          model: deps.pickedModels.chat,
        }),
      );
      totalTokensUsed += ansResp.usage?.totalTokens ?? 0;
      const ansParsed = JSON.parse(ansResp.content || '{}');
      const useful = (ansParsed.binary_score || '').toLowerCase() === 'yes';
      endStep('evaluate', { detail: useful ? 'useful' : 'not useful', meta: { tokensUsed: totalTokensUsed } });
      return useful;
    } catch (e) {
      if (import.meta.env.DEV) {
        console.warn('[orchestrator] evaluate failed (treating as useful to avoid infinite retry):', e);
      }
      endStep('evaluate', { detail: 'eval-error', meta: { tokensUsed: totalTokensUsed } });
      return true;
    }
  }

  async function run(signal?: AbortSignal): Promise<WorkflowState> {
    steps = [];
    emitSteps();

    try {
      beginStep('route', NODE_LABELS.route);
      const routeResp = await withRetry('llm-invoke-route', () =>
        deps.llm.invoke({
          system: ROUTER_INSTRUCTIONS,
          messages: [{ role: 'user', content: question }],
          jsonMode: true,
          temperature: EVAL_TEMPERATURE,
          model: deps.pickedModels.chat,
        }),
        signal
      );
      let source: SourceType;
      let confidence = 1.0;
      try {
        const parsed = JSON.parse(routeResp.content || '{}');
        const raw = (parsed.datasource as SourceType) || 'vectorstore';
        source = VALID_SOURCE_TYPES.includes(raw) ? raw : 'vectorstore';
        const parsedConfidence = Number(parsed.confidence);
        if (!Number.isNaN(parsedConfidence) && parsedConfidence >= 0 && parsedConfidence <= 1) {
          confidence = parsedConfidence;
        }
      } catch (e) {
        if (import.meta.env.DEV) {
          console.warn('[orchestrator] route JSON parse failed (defaulting to vectorstore):', e);
        }
        source = 'vectorstore';
      }
      state.routing = source;

      const isLowConfidence = confidence < ROUTER_CONFIDENCE_THRESHOLD;
      const routeDetail = isLowConfidence
        ? `-> ${source} (confidence: ${confidence.toFixed(2)}, low confidence -> multi-source)`
        : `-> ${source} (confidence: ${confidence.toFixed(2)})`;
      endStep('route', { detail: routeDetail, meta: { tokensUsed: routeResp.usage?.totalTokens ?? 0, confidence } });

      if (isLowConfidence) {
        await Promise.all([
          runPath('vectorstore', signal),
          runPath('websearch', signal),
        ]);
      } else {
        await runPath(source, signal);
      }

      let useful = false;
      const maxRetries = deps.settings.maxRetries ?? 3;
      while (!useful && state.retryCount < maxRetries) {
        if (signal?.aborted) {
          setError(new Error('Aborted'), 'generate');
          break;
        }
        await generate();
        useful = await evaluate();
        if (!useful && state.retryCount < maxRetries) {
          state.retryCount++;
          beginStep('decide', 'Re-routing');
          const previousSource: SourceType = state.routing ?? 'vectorstore';
          const fallback: SourceType = previousSource === 'vectorstore' ? 'websearch' : 'vectorstore';
          state.routing = fallback;
          const rewrittenQuestion = await rewriteQuestionForSource(question, fallback, signal);
          endStep('decide', { detail: `-> ${fallback} (rewritten: ${rewrittenQuestion.slice(0, 100)}${rewrittenQuestion.length > 100 ? '…' : ''})` });
          clearSourceData(previousSource);
          await runPath(fallback, signal, rewrittenQuestion);
        }
      }

      if (signal?.aborted) {
        beginStep('end', NODE_LABELS.end);
        state.finishedAt = Date.now();
        endStep('end');
        emitSteps();
        callbacks.onPartialUpdate?.(state);
        return state;
      }

      beginStep('end', NODE_LABELS.end);
      state.finishedAt = Date.now();
      endStep('end');
      emitSteps();
      callbacks.onPartialUpdate?.(state);
      return state;
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      const stepContext = err instanceof RagError ? (err.step ?? 'run') : 'run';
      setError(err, stepContext);
      state.finishedAt = Date.now();
      emitSteps();
      return state;
    }
  }

  return { run };
}

const ROUTER_INSTRUCTIONS = `You are an expert router that decides where to send a user question.

The vectorstore contains user-uploaded documents (PDFs, markdown, text files).

The Python data API contains user-uploaded CSV datasets as Arquero tables.

Web search is for current/factual general knowledge questions.

Return JSON with keys "datasource" (one of "vectorstore", "python", or "websearch") and "confidence" (number 0-1 indicating confidence in the routing decision).`;

const DOC_GRADER_INSTRUCTIONS = `You assess document relevance to a question. Return JSON with "binary_score": "yes" if relevant, "no" otherwise.`;

const DOC_GRADER_PROMPT = `Document: {document}\n\nQuestion: {question}\n\nIs this document relevant? Return JSON {"binary_score": "yes|no"}.`;

const RAG_PROMPT = `You are a careful assistant that answers based ONLY on the context below.

Context:
{context}

Question: {question}

Instructions:
- Base your answer ONLY on the provided context
- Be concise and direct
- Cite sources inline with [1], [2] etc.
- Include a "References:" section at the end
- If you can't answer, say so

Answer:`;

const HALLUCINATION_INSTRUCTIONS = `You check if a student answer is grounded in provided facts. Return JSON {"binary_score": "yes|no", "explanation": "..."}.`;

const HALLUCINATION_PROMPT = `FACTS: {documents}\n\nSTUDENT ANSWER: {generation}\n\nIs the answer fully grounded in the FACTS? Return JSON {"binary_score": "yes|no", "explanation": "..."}.`;

const ANSWER_INSTRUCTIONS = `You grade whether an answer addresses the question. Return JSON {"binary_score": "yes|no", "explanation": "..."}.`;

const ANSWER_PROMPT = `QUESTION: {question}\n\nSTUDENT ANSWER: {generation}\n\nDoes the answer address the question? Return JSON {"binary_score": "yes|no", "explanation": "..."}.`;