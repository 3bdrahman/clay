import type {
  DataAnalysisResult,
  Settings,
  SourceType,
  StepTrace,
  WorkflowState,
} from '../lib/types';
import type { LLMClient } from '../lib/llm';
import type { PickedModels } from '../lib/models';
import type { VectorStore } from '../lib/vectorstore';
import type { WebSearchClient } from '../lib/websearch';
import type { DataAnalyzer } from './analyzer';
import { ROUTER_INSTRUCTIONS } from './orchestratorPrompts';
import { RagError, RagErrorCode, isRetryable, getUserMessage, GenerationFailedError } from '../lib/errors';
import {
  NODE_LABELS,
  EVAL_TEMPERATURE,
  type OrchestratorStepContext,
  rewriteQuestionForSource,
  runPath,
  generate,
  evaluate,
} from './orchestratorSteps';

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

const VALID_SOURCE_TYPES: SourceType[] = ['vectorstore', 'python', 'websearch'];

const ROUTER_CONFIDENCE_THRESHOLD = 0.6;
const MAX_STEP_RETRIES = 2;
const BASE_RETRY_DELAY_MS = 1000;

function webSearchAvailability(settings: Settings): string {
  if (settings.webSearchProvider === 'none') return 'disabled by the user — never route here';
  if (settings.webSearchProvider === 'serper') {
    return settings.serperApiKey
      ? 'available (Serper API key configured)'
      : 'selected but no Serper API key configured — requests would fail';
  }
  return import.meta.env.DEV
    ? 'available (DuckDuckGo via the dev proxy)'
    : 'DEGRADED: DuckDuckGo cannot be reached from browser deployments in production (no CORS headers) — route here only for general-knowledge questions the sources above cannot answer';
}

function buildRouterContext(deps: OrchestratorDeps): string {
  const lines: string[] = [];

  const datasets = deps.analyzer.listDatasets();
  if (datasets.length > 0) {
    const dsLines = datasets.slice(0, 8).map(d => {
      const cols = (d.columns ?? []).slice(0, 12).join(', ');
      return `- ${d.name} (${d.rowCount} rows): [${cols}]`;
    });
    lines.push(`Datasets available for data analysis (source "python"):\n${dsLines.join('\n')}`);
  } else {
    lines.push('Datasets: none loaded — data analysis cannot answer anything');
  }

  const sources = deps.vectorstore.listSources();
  if (sources.length > 0) {
    const docLines = sources.slice(0, 8).map(s => `- ${s.source} (${s.entryCount} chunks)`);
    lines.push(`Documents available for vector search (source "vectorstore"):\n${docLines.join('\n')}`);
  } else {
    lines.push('Documents: none uploaded — vector search cannot answer anything');
  }

  lines.push(`Web search (source "websearch"): ${webSearchAvailability(deps.settings)}`);

  return lines.join('\n\n');
}

function nextUntriedSource(tried: SourceType[]): SourceType {
  // Prefer the richest untried source: the real data beats documents beats
  // the open web, and web search is CORS-degraded in production deployments.
  const richness: SourceType[] = ['python', 'vectorstore', 'websearch'];
  for (const s of richness) {
    if (!tried.includes(s)) return s;
  }
  return 'vectorstore';
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

  function endStep(node: string, opts: { status?: 'done' | 'error'; detail?: string; meta?: Record<string, unknown> } = {}): void {
    for (let i = steps.length - 1; i >= 0; i--) {
      const step = steps[i];
      if (step && step.node === node && step.status === 'running') {
        step.status = opts.status ?? 'done';
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

    state.error = {
      code,
      message,
      step,
      retryable,
    };

    callbacks.onError?.(err);
    emitSteps();
  }

  function endAllRunningSteps(detail: string): void {
    const now = Date.now();
    for (const step of steps) {
      if (step.status === 'running') {
        step.status = 'skipped';
        step.finishedAt = now;
        step.durationMs = now - (step.startedAt || now);
        step.detail = detail;
      }
    }
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

  async function run(signal?: AbortSignal): Promise<WorkflowState> {
    steps = [];
    emitSteps();

    try {
      const ctx: OrchestratorStepContext = {
        question,
        deps,
        callbacks,
        state,
        steps,
        beginStep,
        endStep,
        emitSteps,
        withRetry,
      };

      beginStep('route', NODE_LABELS.route);
      const routeResp = await withRetry('llm-invoke-route', () =>
        deps.llm.invoke({
          system: ROUTER_INSTRUCTIONS,
          messages: [{
            role: 'user',
            content: `${question}\n\n---\n\nROUTING CONTEXT (current sandbox contents):\n\n${buildRouterContext(deps)}`,
          }],
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
      // Eval measures the FIRST routing decision; routing is mutated by the
      // retry loop when it re-routes, so the initial value is kept separate.
      state.initialRouting = source;

      const isLowConfidence = confidence < ROUTER_CONFIDENCE_THRESHOLD;
      const routeDetail = isLowConfidence
        ? `-> ${source} (confidence: ${confidence.toFixed(2)}, low confidence -> multi-source)`
        : `-> ${source} (confidence: ${confidence.toFixed(2)})`;
      endStep('route', { detail: routeDetail, meta: { tokensUsed: routeResp.usage?.totalTokens ?? 0, confidence } });

      const triedSources: SourceType[] = [];
      if (isLowConfidence) {
        triedSources.push('vectorstore', 'websearch', 'python');
        const outcomes = await Promise.allSettled([
          runPath(ctx, 'vectorstore', signal),
          runPath(ctx, 'websearch', signal),
          runPath(ctx, 'python', signal),
        ]);
        const rejected = outcomes.filter((o): o is PromiseRejectedResult => o.status === 'rejected');
        if (rejected.length === outcomes.length) {
          const first = rejected[0];
          throw first
            ? first.reason
            : new GenerationFailedError('orchestrator', new Error('All retrieval paths failed'), { retryable: false });
        }
      } else {
        triedSources.push(source);
        await runPath(ctx, source, signal);
      }

      let useful = false;
      const maxRetries = deps.settings.maxRetries ?? 3;
      while (!useful && state.retryCount < maxRetries) {
        if (signal?.aborted) break;
        await generate(ctx);
        useful = await evaluate(ctx);
        if (!useful && state.retryCount < maxRetries) {
          state.retryCount++;
          beginStep('decide', 'Re-routing');
          const fallback: SourceType = nextUntriedSource(triedSources);
          state.routing = fallback;
          const rewrittenQuestion = await rewriteQuestionForSource(ctx, question, fallback, signal);
          endStep('decide', { detail: `-> ${fallback} (rewritten: ${rewrittenQuestion.slice(0, 100)}${rewrittenQuestion.length > 100 ? '…' : ''})` });
          triedSources.push(fallback);
          const fallbackOutcome = await Promise.allSettled([runPath(ctx, fallback, signal, rewrittenQuestion)]);
          if (fallbackOutcome[0]?.status === 'rejected') {
            // The fallback source failed after its own retries — keep the last
            // generated answer and stop retrying; the failed step trace carries the reason.
            break;
          }
        }
      }

      if (signal?.aborted) {
        // User-intentional abort: not a workflow error. Close in-flight steps
        // and return the partial state so the streamed answer is kept.
        endAllRunningSteps('aborted');
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
      if (signal?.aborted || (err instanceof RagError && err.code === RagErrorCode.STREAM_INTERRUPTED)) {
        // In-flight LLM calls reject with StreamInterruptedError when the user
        // aborts mid-step — same graceful path as the explicit abort check.
        endAllRunningSteps('aborted');
        beginStep('end', NODE_LABELS.end);
        state.finishedAt = Date.now();
        endStep('end');
        emitSteps();
        callbacks.onPartialUpdate?.(state);
        return state;
      }
      const stepContext = err instanceof RagError ? (err.step ?? 'run') : 'run';
      setError(err, stepContext);
      state.finishedAt = Date.now();
      emitSteps();
      return state;
    }
  }

  return { run };
}
