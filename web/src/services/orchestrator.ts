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
import { RagError, RagErrorCode, GenerationFailedError } from '../lib/errors';
import {
  NODE_LABELS,
  EVAL_TEMPERATURE,
  type OrchestratorStepContext,
  runPath,
  generate,
  evaluate,
  rewriteQuestionForSource,
} from './orchestratorSteps';
import {
  VALID_SOURCE_TYPES,
  ROUTER_CONFIDENCE_THRESHOLD,
  buildRouterContext,
  nextUntriedSource,
} from './orchestratorRouter';
import { withRetry } from './orchestratorRetry';
import { createStepManager } from './orchestratorStepManager';

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

  const stepManager = createStepManager(state, callbacks);
  const { steps, beginStep, endStep, setError, endAllRunningSteps, emitSteps } = stepManager;

  const retryCtx = { steps, emitSteps };

  async function run(signal?: AbortSignal): Promise<WorkflowState> {
    steps.length = 0;
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
        withRetry: <T>(stepName: string, fn: () => Promise<T>, sig?: AbortSignal) =>
          withRetry<T>(stepName, fn, retryCtx, sig),
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
        retryCtx,
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