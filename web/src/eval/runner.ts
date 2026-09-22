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
  listModels,
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
  answerScore?: number; // Lexical overlap score 0-1
  judgeScore?: number; // LLM-as-judge score 0-1
}

export interface EvalSummary {
  total: number;
  passed: number;
  failed: number;
  routingAccuracy: number;
  avgRecallAtK: number;
  avgLatencyMs: number;
  avgAnswerScore: number;
  avgJudgeScore: number;
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
      : await listModels(settings.provider, endpoint.apiKey);
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
    datasets: tables,
    metadata,
    codeGenModel: picked.chat,
    maxToolLoopTokens: settings.maxToolLoopTokens,
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
 * Tokenize text into lowercase alphanumeric tokens.
 */
function tokenize(text: string): string[] {
  return text.toLowerCase().split(/\W+/).filter(t => t.length > 0);
}

/**
 * Compute lexical overlap score between two texts.
 * Uses weighted Jaccard similarity: intersection weight / union weight.
 * Weight of each token is 1 / (1 + log(total_frequency)) to downweight common tokens.
 * Returns a score between 0 and 1.
 */
export function computeLexicalOverlap(generated: string, golden: string): number {
  const genTokens = tokenize(generated);
  const goldTokens = tokenize(golden);

  if (genTokens.length === 0 && goldTokens.length === 0) return 1;
  if (genTokens.length === 0 || goldTokens.length === 0) return 0;

  // Count frequencies
  const genFreq = new Map<string, number>();
  const goldFreq = new Map<string, number>();

  for (const t of genTokens) genFreq.set(t, (genFreq.get(t) ?? 0) + 1);
  for (const t of goldTokens) goldFreq.set(t, (goldFreq.get(t) ?? 0) + 1);

  // Compute weighted intersection and union
  let intersectionWeight = 0;
  let unionWeight = 0;

  const allTokens = new Set([...genFreq.keys(), ...goldFreq.keys()]);
  for (const token of allTokens) {
    const genCount = genFreq.get(token) ?? 0;
    const goldCount = goldFreq.get(token) ?? 0;
    const totalCount = genCount + goldCount;
    const weight = 1 / (1 + Math.log(totalCount));

    if (genCount > 0 && goldCount > 0) {
      intersectionWeight += weight * Math.min(genCount, goldCount);
    }
    unionWeight += weight * Math.max(genCount, goldCount);
  }

  return unionWeight === 0 ? 0 : intersectionWeight / unionWeight;
}

/**
 * Score an answer against a golden answer using LLM-as-judge.
 * Returns a score 0-1 and a one-line rationale.
 */
export async function scoreWithJudge(
  llm: { invoke: (req: { system?: string; messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>; jsonMode?: boolean; temperature?: number; model?: string }) => Promise<{ content: string; usage?: { totalTokens?: number } }> },
  model: string,
  question: string,
  generated: string,
  golden: string
): Promise<{ score: number; rationale: string }> {
  const prompt = `Question: ${question}

Golden Answer: ${golden}

Generated Answer: ${generated}

Score the generated answer on factual coverage of the golden answer (0.0 to 1.0).
Consider: Does the generated answer contain the key facts from the golden answer? Are there hallucinations or missing critical information?
Return JSON: {"score": 0.0-1.0, "rationale": "one-line explanation"}`;

  try {
    const resp = await llm.invoke({
      system: 'You are an expert evaluator. Score factual coverage accurately and concisely.',
      messages: [{ role: 'user', content: prompt }],
      jsonMode: true,
      temperature: 0,
      model,
    });

    const parsed = JSON.parse(resp.content || '{}');
    const score = Math.max(0, Math.min(1, Number.isFinite(parsed.score) ? parsed.score : 0));
    const rationale = String(parsed.rationale ?? '').slice(0, 200);
    return { score, rationale };
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[eval] Judge scoring failed:', e);
    }
    return { score: 0, rationale: 'Judge scoring failed' };
  }
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
  const avgJudgeScore = results.reduce((sum, r) => sum + (r.judgeScore ?? 0), 0) / total;

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
  const avgJudgeScore =
    results.reduce((sum, r) => sum + (r.judgeScore ?? 0), 0) / (total || 1);

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

export function formatReport(summary: EvalSummary): string {
  const lines: string[] = [];
  lines.push('# Clay Eval Report');
  lines.push('');
  lines.push(`**Total Questions:** ${summary.total}`);
  lines.push(`**Passed:** ${summary.passed} / ${summary.total} (${((summary.passed / summary.total) * 100).toFixed(1)}%)`);
  lines.push(`**Routing Accuracy:** ${(summary.routingAccuracy * 100).toFixed(1)}%`);
  lines.push(`**Avg Recall@K:** ${(summary.avgRecallAtK * 100).toFixed(1)}%`);
  lines.push(`**Avg Answer Score:** ${(summary.avgAnswerScore * 100).toFixed(1)}%`);
  lines.push(`**Avg Judge Score:** ${(summary.avgJudgeScore * 100).toFixed(1)}%`);
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
    lines.push(`- **Answer Score**: ${((r.answerScore ?? 0) * 100).toFixed(1)}%`);
    if (r.judgeScore !== undefined) lines.push(`- **Judge Score**: ${(r.judgeScore * 100).toFixed(1)}%`);
    lines.push(`- **Latency**: ${r.latencyMs}ms`);
    if (r.error) lines.push(`- **Error**: ${r.error}`);
    lines.push(`- **Answer Preview**: ${r.answer.slice(0, 200)}...`);
    lines.push('');
  }

  return lines.join('\n');
}