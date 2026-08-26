// Eval runner — executes golden test set and computes metrics.
// Usage: import in a test file or run via `npm run eval`

import { loadSampleDatasets } from '../services/datasets';
import { createLLMClient, type LLMClient } from '../lib/llm';
import { createEmbeddingsClient, type EmbeddingsClient } from '../lib/embeddings';
import { createWebSearchClient, type WebSearchClient } from '../lib/websearch';
import { createVectorStore, type VectorStore } from '../lib/vectorstore';
import { createDataAnalyzer, type DataAnalyzer } from '../services/analyzer';
import { createWorkflowOrchestrator } from '../services/orchestrator';
import {
  listNimModels,
  listLocalCatalog,
  resolveModels,
  pickLocalModels,
  type PickedModels,
} from '../lib/models';
import { resolveProviderEndpoint } from '../lib/providers';
import type { Settings } from '../lib/types';
import type { DatasetSummary, DocumentSummary } from '../lib/exampleQueries';
import { generateEvalQuestions } from './dynamicQuestions';

export interface EvalQuestion {
  id: string;
  question: string;
  category: 'data_analysis' | 'documents' | 'web_search';
  expectedSource: 'python' | 'vectorstore' | 'websearch';
  expectedColumnIntent?: string[];
  goldenAnswer: string;
  minRelevantChunks: number;
}

export interface EvalResult {
  questionId: string;
  question: string;
  category: string;
  expectedSource: string;
  actualSource: string | undefined;
  routingCorrect: boolean;
  retrievedChunks: number;
  relevantChunks: number;
  recallAtK: number;
  answer: string;
  latencyMs: number;
  error?: string;
}

export interface EvalSummary {
  total: number;
  passed: number;
  failed: number;
  routingAccuracy: number;
  avgRecallAtK: number;
  avgLatencyMs: number;
  byCategory: Record<string, { total: number; passed: number; routingAccuracy: number }>;
  results: EvalResult[];
}

async function createServices(settings: Settings): Promise<{
  llm: LLMClient;
  embeddings: EmbeddingsClient;
  vectorstore: VectorStore;
  webSearch: WebSearchClient;
  analyzer: DataAnalyzer;
  pickedModels: PickedModels;
}> {
  const endpoint = resolveProviderEndpoint(settings);

  const catalog =
    settings.provider === 'local'
      ? await listLocalCatalog(endpoint.baseUrl, '')
      : await listNimModels(endpoint.apiKey);
  const picked: PickedModels =
    settings.provider === 'local'
      ? pickLocalModels(settings.localModels)
      : resolveModels(settings, catalog).picked;

  const embeddingKey =
    settings.provider === 'local' ? '' : (settings.embeddingApiKey || settings.apiKey);
  const embeddings = createEmbeddingsClient({
    baseUrl: endpoint.baseUrl,
    apiKey: embeddingKey,
    embeddingModel: picked.embedding ?? '',
    providerLabel: endpoint.providerLabel,
  });
  const vectorstore = createVectorStore(embeddings);
  const webSearch = createWebSearchClient(settings);

  const llm = createLLMClient({
    baseUrl: endpoint.baseUrl,
    apiKey: endpoint.apiKey,
    temperature: settings.temperature,
    providerLabel: endpoint.providerLabel,
  });

  const { tables, metadata } = await loadSampleDatasets();
  const analyzer = createDataAnalyzer({
    llm,
    embeddings,
    datasets: tables,
    metadata,
    codeGenModel: picked.codeGen,
  });

  await vectorstore.load();

  return { llm, embeddings, vectorstore, webSearch, analyzer, pickedModels: picked };
}

function gradeRouting(result: EvalResult): boolean {
  return result.actualSource === result.expectedSource;
}

function computeRecallAtK(relevant: number, total: number): number {
  if (total === 0) return 1;
  return Math.min(1, relevant / total);
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
  const { metadata } = await loadSampleDatasets();
  const datasets: DatasetSummary[] = Object.entries(metadata).map(([name, meta]) => ({
    name,
    fileName: name + '.csv',
    columns: meta.columns,
    rowCount: meta.rowCount,
  }));
  const documents: DocumentSummary[] = []; // Would be populated from vectorstore in real use

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
      actualSource = workflow.routing;
      retrievedChunks = workflow.documents.length;
      relevantChunks = workflow.documents.filter(d => (d.score ?? 0) > 0.3).length;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }

    const latencyMs = Date.now() - start;
    const routingCorrect = gradeRouting({ actualSource, expectedSource: q.expectedSource } as EvalResult);
    const recallAtK = computeRecallAtK(relevantChunks, q.minRelevantChunks);

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
    });
  }

  const total = results.length;
  const _passed = results.filter(r => !r.error && r.routingCorrect && r.recallAtK >= 0.5).length;
  const failed = total - _passed;
  const routingAccuracy = results.filter(r => r.routingCorrect).length / total;
  const avgRecallAtK = results.reduce((sum, r) => sum + r.recallAtK, 0) / total;
  const avgLatencyMs = results.reduce((sum, r) => sum + r.latencyMs, 0) / total;

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
    byCategory,
    results,
  };
}

export function formatReport(summary: EvalSummary): string {
  const lines: string[] = [];
  lines.push('# Clay Eval Report');
  lines.push('');
  lines.push(`**Total Questions:** ${summary.total}`);
  lines.push(`**Passed:** ${summary.passed} / ${summary.total} (${((summary.passed / summary.total) * 100).toFixed(1)}%)`);
  lines.push(`**Routing Accuracy:** ${(summary.routingAccuracy * 100).toFixed(1)}%`);
  lines.push(`**Avg Recall@K:** ${(summary.avgRecallAtK * 100).toFixed(1)}%`);
  lines.push(`**Avg Latency:** ${summary.avgLatencyMs.toFixed(0)}ms`);
  lines.push('');

  lines.push('## By Category');
  lines.push('');
  for (const [cat, stats] of Object.entries(summary.byCategory)) {
    lines.push(`- **${cat}**: ${stats.passed}/${stats.total} passed, routing ${(stats.routingAccuracy * 100).toFixed(1)}%`);
  }
  lines.push('');

  lines.push('## Per-Question Results');
  lines.push('');
  for (const r of summary.results) {
    const status = r.error ? '❌ ERROR' : (r.routingCorrect && r.recallAtK >= 0.5 ? '✅ PASS' : '❌ FAIL');
    lines.push(`### ${r.questionId} ${status}`);
    lines.push(`- **Question**: ${r.question}`);
    lines.push(`- **Category**: ${r.category}`);
    lines.push(`- **Expected Source**: ${r.expectedSource}`);
    lines.push(`- **Actual Source**: ${r.actualSource ?? '—'}`);
    lines.push(`- **Routing**: ${r.routingCorrect ? '✅' : '❌'}`);
    lines.push(`- **Retrieved Chunks**: ${r.retrievedChunks}`);
    lines.push(`- **Relevant Chunks**: ${r.relevantChunks}`);
    lines.push(`- **Recall@K**: ${(r.recallAtK * 100).toFixed(1)}%`);
    lines.push(`- **Latency**: ${r.latencyMs}ms`);
    if (r.error) lines.push(`- **Error**: ${r.error}`);
    lines.push(`- **Answer Preview**: ${r.answer.slice(0, 200)}...`);
    lines.push('');
  }

  return lines.join('\n');
}