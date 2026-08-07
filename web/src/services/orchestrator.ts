// Workflow orchestrator — proper state machine for the Clay assistant

import type {
  Citation,
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
import type { DataAnalyzer } from './analyzer';
import {
  expandHyDE,
  parallelFanOut,
  parallelGrade,
  formatHeadingCitation,
} from './orchestratorHelpers';

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
      id: `${node}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
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

  function emitSteps(): void {
    state.steps = [...steps];
    callbacks.onStepUpdate?.(state.steps);
    callbacks.onPartialUpdate?.({ ...state });
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
        temperature: 0,
        model: deps.pickedModels.eval,
      });
      try {
        const parsed = JSON.parse(resp.content || '{}');
        return (parsed.binary_score || '').toLowerCase() === 'yes' ? 'relevant' : 'irrelevant';
      } catch {
        return 'irrelevant';
      }
    } catch {
      return 'keep-on-error';
    }
  }

  async function runVectorstorePath(signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) return;
    beginStep('retrieve', NODE_LABELS.retrieve);

    const hypothetical = await expandHyDE(question, { llm: deps.llm, model: deps.pickedModels.eval });
    const initialK = deps.settings.maxRetries ?? 8;
    const rerankK = 4;

    const docs = await parallelFanOut(
      (q, k) => deps.vectorstore.similaritySearch(q, k),
      question,
      hypothetical,
      initialK,
      rerankK,
    );
    state.documents = docs;
    endStep('retrieve', { detail: `${docs.length} docs`, meta: { count: docs.length } });

    beginStep('grade_docs', NODE_LABELS.grade_docs);
    const filtered = await parallelGrade(docs, gradeDocRelevance, {
      earlyExitAt: 4,
      signal,
    });
    state.documents = filtered;
    endStep('grade_docs', { detail: `${filtered.length}/${docs.length} relevant` });
  }

  async function runPythonPath(signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) return;
    beginStep('analyze', NODE_LABELS.analyze);
    try {
      const result = await deps.analyzer.analyze(question, signal);
      state.dataAnalysis = result;
      endStep('analyze', {
        detail: result.resultType === 'error' ? 'error' : 'complete',
        meta: { attempts: result.attempts, durationMs: Math.round(result.durationMs) },
      });
    } catch (e) {
      endStep('analyze', { detail: 'failed' });
      throw e;
    }
  }

  async function runWebSearchStep(signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) return;
    beginStep('web_search', NODE_LABELS.web_search);
    const results = await deps.webSearch.search(question, 4);
    state.webResults = results;
    endStep('web_search', { detail: `${results.length} results` });
  }

  function clearSourceData(source: SourceType): void {
    if (source === 'vectorstore') state.documents = [];
    else if (source === 'python') state.dataAnalysis = undefined;
    else if (source === 'websearch') state.webResults = [];
  }

  async function runPath(source: SourceType, signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) return;
    switch (source) {
      case 'vectorstore':
        await runVectorstorePath(signal);
        break;
      case 'python':
        await runPythonPath(signal);
        break;
      case 'websearch':
        await runWebSearchStep(signal);
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
      sections.push(
        `DATA ANALYSIS:\n` +
          `Question: ${question}\n` +
          `Code:\n\`\`\`js\n${da.code}\n\`\`\`\n` +
          `Result: ${JSON.stringify(da.result, null, 2)}\n` +
          `Explanation: ${da.explanation}`
      );
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
      const resp = await deps.llm.stream(
        {
          system: 'You are a helpful assistant that answers questions based solely on the provided context. Cite sources inline using [1], [2], etc., and include a References: section at the end.',
          messages: [{ role: 'user', content: prompt }],
          temperature: deps.settings.temperature ?? 0,
          model: deps.pickedModels.answer,
        },
        callbacks.onToken ?? (() => {}),
      );
      state.answer = resp.content;
      buildCitations();
      endStep('generate', { detail: 'complete' });
    } catch (e) {
      state.answer = `Error generating answer: ${e instanceof Error ? e.message : String(e)}`;
      endStep('generate', { detail: 'error' });
    }
  }

  async function evaluate(): Promise<boolean> {
    beginStep('evaluate', NODE_LABELS.evaluate);
    const context = buildContextForEval();
    if (!context.trim()) {
      endStep('evaluate', { detail: 'no context' });
      return false;
    }

    try {
      const halluc = await deps.llm.invoke({
        system: HALLUCINATION_INSTRUCTIONS,
        messages: [
          {
            role: 'user',
            content: HALLUCINATION_PROMPT.replace('{documents}', context).replace('{generation}', state.answer || ''),
          },
        ],
        jsonMode: true,
        temperature: 0,
        model: deps.pickedModels.eval,
      });
      const hallucParsed = JSON.parse(halluc.content || '{}');
      if ((hallucParsed.binary_score || '').toLowerCase() !== 'yes') {
        endStep('evaluate', { detail: 'hallucination' });
        return false;
      }

      const ansResp = await deps.llm.invoke({
        system: ANSWER_INSTRUCTIONS,
        messages: [
          {
            role: 'user',
            content: ANSWER_PROMPT.replace('{question}', question).replace('{generation}', state.answer || ''),
          },
        ],
        jsonMode: true,
        temperature: 0,
        model: deps.pickedModels.eval,
      });
      const ansParsed = JSON.parse(ansResp.content || '{}');
      const useful = (ansParsed.binary_score || '').toLowerCase() === 'yes';
      endStep('evaluate', { detail: useful ? 'useful' : 'not useful' });
      return useful;
    } catch {
      endStep('evaluate', { detail: 'eval-error' });
      return true;
    }
  }

  async function run(signal?: AbortSignal): Promise<WorkflowState> {
    steps = [];
    emitSteps();

    try {
      beginStep('route', NODE_LABELS.route);
      const routeResp = await deps.llm.invoke({
        system: ROUTER_INSTRUCTIONS,
        messages: [{ role: 'user', content: question }],
        jsonMode: true,
        temperature: 0,
        model: deps.pickedModels.routing,
      });
      let source: SourceType;
      try {
        const parsed = JSON.parse(routeResp.content || '{}');
        source = (parsed.datasource as SourceType) || 'vectorstore';
      } catch {
        source = 'vectorstore';
      }
      state.routing = source;
      endStep('route', { detail: `-> ${source}` });

      await runPath(source, signal);

      let useful = false;
      const maxRetries = deps.settings.maxRetries ?? 3;
      while (!useful && state.retryCount < maxRetries) {
        if (signal?.aborted) {
          state.error = 'Aborted';
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
          endStep('decide', { detail: `-> ${fallback}` });
          clearSourceData(previousSource);
          await runPath(fallback, signal);
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
      state.error = err.message;
      callbacks.onError?.(err);
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

Return JSON with a single key "datasource": one of "vectorstore", "python", or "websearch".`;

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