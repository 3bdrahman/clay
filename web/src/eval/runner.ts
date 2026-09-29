// Eval runner — executes golden test set and computes metrics.
// Usage: import in a test file or run via `npm run eval`

import { loadSampleDatasets } from '../services/datasets';
import type { LLMClient } from '../lib/llm';
import type { EmbeddingsClient } from '../lib/embeddings';
import type { WebSearchClient } from '../lib/websearch';
import type { VectorStore } from '../lib/vectorstore';
import { createClayServiceBundle } from '../services/clayServices';
import type { DataAnalyzer, DatasetMeta } from '../services/analyzer';
import { createWorkflowOrchestrator } from '../services/orchestrator';
import { listModels, listLocalCatalog, type PickedModels } from '../lib/models';
import { resolveProviderEndpoint } from '../lib/providers';
import type { Settings } from '../lib/types';
import type { DatasetSummary, DocumentSummary } from '../lib/exampleQueries';
import { generateEvalQuestions } from './dynamicQuestions';
import type { EvalQuestion, EvalResult, EvalSummary } from './types';
import { computeLexicalOverlap, computeRecallAtK, gradeRouting } from './metrics';
import { scoreWithJudge } from './grading';

// Re-export types
export type { EvalQuestion, EvalResult, EvalSummary } from './types';

// Re-export metrics functions
export { computeLexicalOverlap, computeRecallAtK, gradeRouting } from './metrics';

// Re-export grading function
export { scoreWithJudge } from './grading';

// Re-export report formatting
export { formatReport } from './report';

async function createServices(settings: Settings): Promise<{
  llm: LLMClient;
  embeddings: EmbeddingsClient;
  vectorstore: VectorStore;
  webSearch: WebSearchClient;
  analyzer: DataAnalyzer;
  pickedModels: PickedModels;
  datasetMetadata: DatasetMeta;
}> {
  const endpoint = resolveProviderEndpoint(settings);
  const catalog =
    settings.provider === 'local'
      ? await listLocalCatalog(endpoint.baseUrl, '')
      : await listModels(settings.provider, endpoint.apiKey);

  const { tables, metadata } = await loadSampleDatasets();
  const bundle = createClayServiceBundle({
    settings,
    catalog,
    analyzerTables: tables,
    analyzerMetadata: metadata,
  });

  await bundle.vectorstore.load();

  return {
    llm: bundle.llm,
    embeddings: bundle.embeddings,
    vectorstore: bundle.vectorstore,
    webSearch: bundle.webSearch,
    analyzer: bundle.analyzer,
    pickedModels: bundle.pickedModels,
    datasetMetadata: metadata,
  };
}

/**
 * Generate evaluation questions dynamically from actual loaded data.
 * If questions array is provided, use it (backwards compatibility).
 * Otherwise, generate questions based on the datasets and documents.
 */
async function getEvalQuestions(
  _settings: Settings,
  providedQuestions: EvalQuestion[] | undefined,
  datasets: DatasetSummary[],
  documents: DocumentSummary[]
): Promise<EvalQuestion[]> {
  if (providedQuestions && providedQuestions.length > 0) {
    return providedQuestions;
  }
  return generateEvalQuestions(datasets, documents);
}

export async function runEval(
  settings: Settings,
  questions: EvalQuestion[] | undefined,
  onProgress?: (done: number, total: number, current: EvalQuestion) => void,
): Promise<EvalSummary> {
  const services = await createServices(settings);

  // Get datasets and documents for dynamic question generation
  const metadata = services.datasetMetadata;
  const datasets: DatasetSummary[] = Object.entries(metadata).map(([name, meta]) => ({
    name,
    fileName: name + '.csv',
    columns: meta.columns,
    rowCount: meta.rowCount,
  }));
  const documents: DocumentSummary[] = services.vectorstore.listSources().map(s => ({
    fileName: s.source,
  }));

  const evalQuestions = await getEvalQuestions(settings, questions, datasets, documents);
  const results: EvalResult[] = [];

  for (let i = 0; i < evalQuestions.length; i++) {
    const q = evalQuestions[i];
    onProgress?.(i, evalQuestions.length, q);

    const start = Date.now();
    let actualSource: string | undefined;
    let retrievedChunks = 0;
    let relevantChunks = 0;
    let answer = '';
    let error: string | undefined;

    try {
      const orchestrator = createWorkflowOrchestrator(
        q.question,
        {
          llm: services.llm,
          vectorstore: services.vectorstore,
          webSearch: services.webSearch,
          analyzer: services.analyzer,
          settings,
          pickedModels: services.pickedModels,
        },
        {},
      );

      const workflow = await orchestrator.run();
      answer = workflow.answer || '';
      actualSource = workflow.initialRouting ?? workflow.routing;
      retrievedChunks = workflow.documents.length;
      relevantChunks = workflow.documents.filter(d => (d.score ?? 0) > 0.3).length;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }

    const latencyMs = Date.now() - start;
    const routingCorrect = gradeRouting({ actualSource, expectedSource: q.expectedSource } as EvalResult);
    const recallAtK = computeRecallAtK(relevantChunks, q.minRelevantChunks);

    // Compute lexical overlap score
    const answerScore = computeLexicalOverlap(answer, q.goldenAnswer);

    // Compute LLM-as-judge score if LLM is available
    let judgeScore: number | undefined;
    const judgeModel = services.pickedModels.chat;
    if (!error && answer.trim() && judgeModel) {
      const judgeResult = await scoreWithJudge(services.llm, judgeModel, q.question, answer, q.goldenAnswer);
      judgeScore = judgeResult.score;
    }

    results.push({
      questionId: q.id,
      question: q.question,
      category: q.category,
      expectedSource: q.expectedSource,
      actualSource,
      routingCorrect,
      retrievedChunks,
      relevantChunks,
      recallAtK,
      answer,
      latencyMs,
      error,
      answerScore,
      judgeScore,
    });
  }

  const total = results.length;
  const _passed = results.filter(r => !r.error && r.routingCorrect && r.recallAtK >= 0.5).length;
  const failed = total - _passed;
  const routingAccuracy = results.filter(r => r.routingCorrect).length / total;
  const avgRecallAtK = results.reduce((sum, r) => sum + r.recallAtK, 0) / total;
  const avgLatencyMs = results.reduce((sum, r) => sum + r.latencyMs, 0) / total;
  const avgAnswerScore = results.reduce((sum, r) => sum + (r.answerScore ?? 0), 0) / total;
  // Judge failures score 0 as a per-result artifact; the average excludes
  // them so a failed judge call doesn't drag the aggregate down.
  const judged = results.filter(r => r.judgeScore !== undefined);
  const avgJudgeScore = judged.length > 0
    ? judged.reduce((sum, r) => sum + (r.judgeScore ?? 0), 0) / judged.length
    : 0;

  const byCategory: Record<string, { total: number; passed: number; routingAccuracy: number }> = {};
  for (const r of results) {
    if (!byCategory[r.category]) byCategory[r.category] = { total: 0, passed: 0, routingAccuracy: 0 };
    byCategory[r.category].total++;
    if (!r.error && r.routingCorrect && r.recallAtK >= 0.5) byCategory[r.category].passed++;
  }
  for (const cat of Object.keys(byCategory)) {
    byCategory[cat].routingAccuracy =
      results.filter(r => r.category === cat && r.routingCorrect).length / byCategory[cat].total;
  }

  return {
    total,
    passed: _passed,
    failed,
    routingAccuracy,
    avgRecallAtK,
    avgLatencyMs,
    avgAnswerScore,
    avgJudgeScore,
    byCategory,
    results,
  };
}

export function gradeQuestionSet(
  questions: EvalQuestion[],
  results: EvalResult[],
): EvalSummary {
  const total = results.length;
  const passed = results.filter(
    (r) => !r.error && r.routingCorrect && r.recallAtK >= 0.5,
  ).length;
  const failed = total - passed;
  const routingAccuracy =
    results.filter((r) => r.routingCorrect).length / (total || 1);
  const avgRecallAtK =
    results.reduce((sum, r) => sum + r.recallAtK, 0) / (total || 1);
  const avgLatencyMs =
    results.reduce((sum, r) => sum + r.latencyMs, 0) / (total || 1);
  const avgAnswerScore =
    results.reduce((sum, r) => sum + (r.answerScore ?? 0), 0) / (total || 1);
  const judged = results.filter((r) => r.judgeScore !== undefined);
  const avgJudgeScore = judged.length > 0
    ? judged.reduce((sum, r) => sum + (r.judgeScore ?? 0), 0) / judged.length
    : 0;

  const byCategory: Record<
    string,
    { total: number; passed: number; routingAccuracy: number }
  > = {};
  for (const r of results) {
    if (!byCategory[r.category]) {
      byCategory[r.category] = { total: 0, passed: 0, routingAccuracy: 0 };
    }
    byCategory[r.category].total++;
    if (!r.error && r.routingCorrect && r.recallAtK >= 0.5) {
      byCategory[r.category].passed++;
    }
  }
  for (const cat of Object.keys(byCategory)) {
    const entry = byCategory[cat]!;
    const matching = results.filter((r) => r.category === cat);
    entry.routingAccuracy =
      matching.filter((r) => r.routingCorrect).length /
      (entry.total || 1);
  }

  void questions;

  return {
    total,
    passed,
    failed,
    routingAccuracy,
    avgRecallAtK,
    avgLatencyMs,
    avgAnswerScore,
    avgJudgeScore,
    byCategory,
    results,
  };
}