import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createWorkflowOrchestrator } from '../services/orchestrator';
import type { LLMClient } from '../lib/llm';
import type { VectorStore } from '../lib/vectorstore';
import type { WebSearchClient } from '../lib/websearch';
import type { DataAnalyzer } from '../services/analyzer';
import type { Settings, WorkflowState, Document, WebResult, DataAnalysisResult } from '../lib/types';
import type { PickedModels } from '../lib/models';

const mockLLM: LLMClient = {
  invoke: vi.fn<Promise<{ content: string; usage?: { promptTokens?: number; completionTokens?: number; totalTokens?: number }; model?: string }>, [Parameters<LLMClient['invoke']>[0]]>(),
  stream: vi.fn<Promise<{ content: string; usage?: { promptTokens?: number; completionTokens?: number; totalTokens?: number }; model?: string }>, [Parameters<LLMClient['stream']>[0], (token: string) => void, AbortSignal?]>(),
};

const mockVectorstore: VectorStore = {
  load: vi.fn<Promise<void>, []>(),
  similaritySearch: vi.fn<Promise<Document[]>, [string, number?]>(),
  addEntries: vi.fn<void, [Array<{ id: string; text: string; source: string; page?: number; embedding: number[] }>]>(),
  removeBySource: vi.fn<number, [string]>(),
  clear: vi.fn<void, []>(),
  stats: { entries: 0 },
};

const mockWebSearch: WebSearchClient = {
  search: vi.fn<Promise<WebResult[]>, [string, number?]>(),
};

const mockAnalyzer: DataAnalyzer = {
  analyze: vi.fn<Promise<DataAnalysisResult>, [string, AbortSignal?]>(),
  listDatasets: vi.fn<DatasetSummary[], []>(),
  getDatasetSummary: vi.fn<DatasetSummary | undefined, [string]>(),
};

const testSettings: Settings = {
  apiKey: 'test-key',
  embeddingApiKey: '',
  webSearchProvider: 'duckduckgo',
  serperApiKey: '',
  temperature: 0,
  maxRetries: 3,
  theme: 'system',
};

const testPickedModels: PickedModels = {
  routing: 'routing-model',
  codeGen: 'codegen-model',
  answer: 'answer-model',
  eval: 'eval-model',
  embedding: 'embedding-model',
};

describe('createWorkflowOrchestrator', () => {
  let orchestrator: ReturnType<typeof createWorkflowOrchestrator>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([]);
    orchestrator = createWorkflowOrchestrator(
      'test question',
      {
        llm: mockLLM,
        vectorstore: mockVectorstore,
        webSearch: mockWebSearch,
        analyzer: mockAnalyzer,
        settings: testSettings,
        pickedModels: testPickedModels,
      },
      {},
    );
  });

  it('routes to vectorstore for document questions', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
      { id: '1', content: 'doc content', source: 'test.pdf', score: 0.9 },
    ]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer based on doc',
      usage: undefined,
      model: 'answer-model',
    });

    const state = await orchestrator.run();

    expect(state.routing).toBe('vectorstore');
    expect(state.documents).toHaveLength(1);
    expect(state.answer).toBe('Answer based on doc');
  });

  it('routes to python for data analysis questions', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'python' }),
    });
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockResolvedValue({
      type: 'data_analysis',
      question: 'test',
      code: 'result = employees.count()',
      explanation: 'Count',
      resultType: 'scalar',
      result: 10,
      attempts: 1,
      durationMs: 100,
      timestamp: Date.now(),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'There are 10 employees',
      usage: undefined,
      model: 'answer-model',
    });

    const state = await orchestrator.run();

    expect(state.routing).toBe('python');
    expect(state.dataAnalysis).toBeDefined();
    expect(state.dataAnalysis?.result).toBe(10);
  });

  it('routes to websearch for general knowledge questions', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'websearch' }),
    });
    (mockWebSearch.search as ReturnType<typeof vi.fn>).mockResolvedValue([
      { type: 'web_search', title: 'Result', content: 'Web content', url: 'http://example.com' },
    ]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Web answer',
      usage: undefined,
      model: 'answer-model',
    });

    const state = await orchestrator.run();

    expect(state.routing).toBe('websearch');
    expect(state.webResults).toHaveLength(1);
  });

  it('grades documents and filters irrelevant ones', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
      { id: '1', content: 'relevant doc', source: 'a.pdf', score: 0.9 },
      { id: '2', content: 'irrelevant doc', source: 'b.pdf', score: 0.8 },
    ]);
    // First doc: yes, second doc: no
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }) })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'no' }) });

    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });

    const state = await orchestrator.run();

    expect(state.documents).toHaveLength(1);
    expect(state.documents[0].id).toBe('1');
  });

  it('retries with different source on evaluation failure', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Bad answer',
      usage: undefined,
      model: 'answer-model',
    });
    // Hallucination check fails
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ binary_score: 'no', explanation: 'Hallucinated' }),
    });
    // Fallback to websearch — returns results so generate() uses them
    (mockWebSearch.search as ReturnType<typeof vi.fn>).mockResolvedValue([
      { type: 'web_search', title: 'Web', content: 'Web content', url: 'http://x.com' },
    ]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Good answer',
      usage: undefined,
      model: 'answer-model',
    });
    // Evaluation passes (2 calls: hallucination + answer-usefulness)
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }) })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }) });

    const state = await orchestrator.run();

    expect(state.retryCount).toBeGreaterThanOrEqual(1);
    expect(state.answer).toBe('Good answer');
  });

  it('respects maxRetries limit', async () => {
    let _streamCallCount = 0;
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      _streamCallCount++;
      return { content: 'Bad answer', usage: undefined, model: 'answer-model' };
    });
    // All evaluations fail
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'no', explanation: 'Failed' }),
    });

    const state = await orchestrator.run();

    expect(state.retryCount).toBeLessThanOrEqual(3); // maxRetries = 3
  });

  it('handles abort signal', async () => {
    const controller = new AbortController();
    controller.abort();

    const state = await orchestrator.run(controller.signal);
    expect(state.error).toBe('Aborted');
  });

  it('emits step updates via callback', async () => {
    const steps: ReturnType<typeof createWorkflowOrchestrator> extends { run(signal?: AbortSignal): Promise<WorkflowState> } ? never : unknown[] = [];
    const callbacks = {
      onStepUpdate: (s: any[]) => steps.push(...s),
    };

    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    const o = createWorkflowOrchestrator(
      'test',
      {
        llm: mockLLM,
        vectorstore: mockVectorstore,
        webSearch: mockWebSearch,
        analyzer: mockAnalyzer,
        settings: testSettings,
        pickedModels: testPickedModels,
      },
      callbacks,
    );

    await o.run();

    expect(steps.length).toBeGreaterThan(0);
    expect(steps.some(s => s.node === 'route')).toBe(true);
    expect(steps.some(s => s.node === 'retrieve')).toBe(true);
    expect(steps.some(s => s.node === 'generate')).toBe(true);
  });

  it('builds citations from documents', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
      { id: '1', content: 'doc content', source: 'test.pdf', page: 5, score: 0.9 },
    ]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer [1]',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    const state = await orchestrator.run();

    expect(state.citations).toHaveLength(1);
    expect(state.citations[0].source).toBe('test.pdf');
    expect(state.citations[0].page).toBe(5);
    expect(state.citations[0].type).toBe('vectorstore');
  });

  it('includes web results in citations', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'websearch' }),
    });
    (mockWebSearch.search as ReturnType<typeof vi.fn>).mockResolvedValue([
      { type: 'web_search', title: 'Web Page', content: 'Web content', url: 'http://example.com' },
    ]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Web answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    const state = await orchestrator.run();

    expect(state.citations.some(c => c.type === 'websearch')).toBe(true);
  });

it('handles empty context gracefully', async () => {
    (mockWebSearch.search as ReturnType<typeof vi.fn>).mockResolvedValue([]);

    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: "I couldn't find relevant information",
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    const state = await orchestrator.run();

    expect(state.answer).toContain('couldn\'t find');
    expect(state.citations).toHaveLength(0);
  });
});