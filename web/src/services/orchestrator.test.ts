import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createWorkflowOrchestrator } from '../services/orchestrator';
import type { LLMClient } from '../lib/llm';
import type { VectorStore } from '../lib/vectorstore';
import type { WebSearchClient } from '../lib/websearch';
import type { DataAnalyzer, AnalyzerHooks } from '../services/analyzer';
import type { Settings, Document, WebResult, DataAnalysisResult } from '../lib/types';
import type { PickedModels } from '../lib/models';
import { AnalysisBudgetExceededError, ProviderUnreachableError } from '../lib/errors';

const mockLLM: LLMClient = {
  invoke: vi.fn<
    Promise<{ content: string; usage?: { promptTokens?: number; completionTokens?: number; totalTokens?: number }; model?: string }>,
    [Parameters<LLMClient['invoke']>[0]]
  >(),
  stream: vi.fn<
    Promise<{ content: string; usage?: { promptTokens?: number; completionTokens?: number; totalTokens?: number }; model?: string }>,
    [Parameters<LLMClient['stream']>[0], (token: string) => void, AbortSignal?]
  >(),
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
  analyze: vi.fn<Promise<DataAnalysisResult>, [string, AbortSignal?, AnalyzerHooks?]>(),
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
  chat: 'user-selected-chat',
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
    // HyDE expansion call
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage for testing.',
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
    for (const [request] of vi.mocked(mockLLM.invoke).mock.calls) {
      expect(request.model).toBe('user-selected-chat');
    }
    for (const [request] of vi.mocked(mockLLM.stream).mock.calls) {
      expect(request.model).toBe('user-selected-chat');
    }
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
    // HyDE expansion call
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage for testing.',
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
      { id: '1', content: 'relevant doc', source: 'a.pdf', score: 0.9 },
      { id: '2', content: 'irrelevant doc', source: 'b.pdf', score: 0.8 },
    ]);
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }) })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'no' }) });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer based on relevant',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    const state = await orchestrator.run();

    expect(state.documents).toHaveLength(1);
    expect(state.documents[0].id).toBe('1');
  });

  it('retries with fallback source on evaluation failure', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    // HyDE expansion call
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage for testing.',
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
      { id: '1', content: 'doc', source: 'a.pdf', score: 0.9 },
    ]);
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }) })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }) })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'no' }) })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }) });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });

    const state = await orchestrator.run();

    expect(state.retryCount).toBe(1);
    expect(state.routing).toBe('websearch');
  });

  it('includes data analysis in citations', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'python' }),
    });
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockResolvedValue({
      type: 'data_analysis',
      question: 'test',
      code: 'result = 1',
      explanation: 'One',
      resultType: 'scalar',
      result: 1,
      attempts: 1,
      durationMs: 10,
      timestamp: Date.now(),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    const state = await orchestrator.run();

    expect(state.citations.some(c => c.type === 'python')).toBe(true);
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

    expect(state.answer).toContain("couldn't find");
    expect(state.citations).toHaveLength(0);
  });

  // ===== New T9 tests =====

  describe('HyDE query expansion', () => {
    it('runs HyDE and uses hypothetical passage for retrieval', async () => {
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: JSON.stringify({ datasource: 'vectorstore' }),
      });
      (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
        { id: '1', content: 'hypothetical passage content', source: 'test.pdf', score: 0.9 },
        { id: '2', content: 'other content', source: 'test.pdf', score: 0.5 },
      ]);
      (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: 'Answer based on doc',
        usage: undefined,
        model: 'answer-model',
      });
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: JSON.stringify({ binary_score: 'yes' }),
      });

      const state = await orchestrator.run();
      expect(state.documents).toHaveLength(2);
      expect(state.answer).toBe('Answer based on doc');
    });

    it('falls back to original question when HyDE fails', async () => {
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: JSON.stringify({ datasource: 'vectorstore' }),
      });
      (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
        { id: '1', content: 'doc', source: 'test.pdf', score: 0.9 },
      ]);
      (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: 'Answer',
        usage: undefined,
        model: 'answer-model',
      });
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: JSON.stringify({ binary_score: 'yes' }),
      });

      const state = await orchestrator.run();
      expect(state.documents).toHaveLength(1);
    });
  });

  describe('Parallel fan-out with rerank', () => {
    it('runs both primary and hypothetical fetches and reranks union to 4', async () => {
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: JSON.stringify({ datasource: 'vectorstore' }),
      });
      (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce([
          { id: '1', content: 'primary match', source: 'a.pdf', score: 0.8 },
          { id: '2', content: 'secondary', source: 'a.pdf', score: 0.7 },
          { id: '3', content: 'tertiary', source: 'a.pdf', score: 0.6 },
          { id: '4', content: 'fourth', source: 'a.pdf', score: 0.5 },
          { id: '5', content: 'fifth', source: 'a.pdf', score: 0.4 },
        ])
        .mockResolvedValueOnce([
          { id: '1', content: 'primary match', source: 'a.pdf', score: 0.8 },
          { id: '6', content: 'hypo unique', source: 'a.pdf', score: 0.9 },
          { id: '7', content: 'another hypo', source: 'a.pdf', score: 0.85 },
          { id: '8', content: 'third hypo', source: 'a.pdf', score: 0.75 },
          { id: '9', content: 'fourth hypo', source: 'a.pdf', score: 0.7 },
        ]);
      (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: 'Answer',
        usage: undefined,
        model: 'answer-model',
      });
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: JSON.stringify({ binary_score: 'yes' }),
      });

      const state = await orchestrator.run();
      expect(state.documents.length).toBeLessThanOrEqual(4);
    });

    it('deduplicates by id across both fetches', async () => {
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: JSON.stringify({ datasource: 'vectorstore' }),
      });
      const common = { id: 'dup', content: 'shared content', source: 'a.pdf', score: 0.9 };
      (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce([common, { id: 'a', content: 'a', source: 'a.pdf', score: 0.8 }])
        .mockResolvedValueOnce([common, { id: 'b', content: 'b', source: 'a.pdf', score: 0.7 }]);
      (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: 'Answer',
        usage: undefined,
        model: 'answer-model',
      });
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: JSON.stringify({ binary_score: 'yes' }),
      });

      const state = await orchestrator.run();
      const dupCount = state.documents.filter(d => d.id === 'dup').length;
      expect(dupCount).toBe(1);
    });
  });

  describe('Parallel grading with early-exit', () => {
    it('grades all docs in parallel and keeps relevant ones', async () => {
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: JSON.stringify({ datasource: 'vectorstore' }),
      });
      // HyDE expansion call
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: 'Hypothetical passage for testing.',
      });
      (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
        { id: '1', content: 'relevant doc', source: 'a.pdf', score: 0.9 },
        { id: '2', content: 'irrelevant doc', source: 'a.pdf', score: 0.8 },
        { id: '3', content: 'also relevant', source: 'a.pdf', score: 0.7 },
      ]);
      (mockLLM.invoke as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }) })
        .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'no' }) })
        .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }) });
      (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: 'Answer',
        usage: undefined,
        model: 'answer-model',
      });
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: JSON.stringify({ binary_score: 'yes' }),
      });

      const state = await orchestrator.run();
      expect(state.documents).toHaveLength(2);
      expect(state.documents.map(d => d.id).sort()).toEqual(['1', '3']);
    });

    it('early-exits after finding 4 relevant docs', async () => {
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: JSON.stringify({ datasource: 'vectorstore' }),
      });
      // HyDE expansion call
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: 'Hypothetical passage for testing.',
      });
      const many = Array.from({ length: 10 }, (_, i) => ({
        id: String(i),
        content: 'doc',
        source: 'a.pdf',
        score: 0.9 - i * 0.01,
      }));
      (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue(many);
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({ content: JSON.stringify({ binary_score: 'yes' }) });
      (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: 'Answer',
        usage: undefined,
        model: 'answer-model',
      });
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: JSON.stringify({ binary_score: 'yes' }),
      });

      const state = await orchestrator.run();
      expect(state.documents.length).toBeLessThanOrEqual(4);
    });

    it('keeps docs on grader error (keep-on-error)', async () => {
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: JSON.stringify({ datasource: 'vectorstore' }),
      });
      // HyDE expansion call
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: 'Hypothetical passage for testing.',
      });
      (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
        { id: '1', content: 'doc', source: 'a.pdf', score: 0.9 },
      ]);
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('grader down'));
      (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: 'Answer',
        usage: undefined,
        model: 'answer-model',
      });
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: JSON.stringify({ binary_score: 'yes' }),
      });

      const state = await orchestrator.run();
      expect(state.documents).toHaveLength(1);
    });
  });

  describe('Heading-aware citations', () => {
    it('includes heading prefix in citation when metadata.heading present', async () => {
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: JSON.stringify({ datasource: 'vectorstore' }),
      });
      // HyDE expansion call
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: 'Hypothetical passage for testing.',
      });
      (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
        {
          id: '1',
          content: 'section body',
          source: 'a.pdf',
          page: 3,
          score: 0.9,
          metadata: { heading: 'Introduction' },
        },
      ]);
      (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: 'Answer',
        usage: undefined,
        model: 'answer-model',
      });
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: JSON.stringify({ binary_score: 'yes' }),
      });

      const state = await orchestrator.run();
      const vecCitation = state.citations.find(c => c.type === 'vectorstore');
      expect(vecCitation).toBeDefined();
      expect(vecCitation?.excerpt).toContain('[Introduction]');
    });

    it('omits heading prefix when no heading metadata', async () => {
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: JSON.stringify({ datasource: 'vectorstore' }),
      });
      // HyDE expansion call
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        content: 'Hypothetical passage for testing.',
      });
      (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
        { id: '1', content: 'section body', source: 'a.pdf', score: 0.9 },
      ]);
      (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: 'Answer',
        usage: undefined,
        model: 'answer-model',
      });
      (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
        content: JSON.stringify({ binary_score: 'yes' }),
      });

      const state = await orchestrator.run();
      const vecCitation = state.citations.find(c => c.type === 'vectorstore');
      expect(vecCitation?.excerpt).not.toContain('[');
    });

  it('truncates long excerpts to 200 chars', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    // HyDE expansion call
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage for testing.',
    });
    const longContent = 'x'.repeat(500);
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
      { id: '1', content: longContent, source: 'a.pdf', score: 0.9 },
    ]);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    const state = await orchestrator.run();
    const vecCitation = state.citations.find(c => c.type === 'vectorstore');
    expect(vecCitation?.excerpt.length).toBeLessThanOrEqual(200);
  });
});

describe('createWorkflowOrchestrator — retrieval K is independent of maxRetries (issue #1)', () => {
  function makeOrchestrator(settings: Settings) {
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockReset();
    mockVectorstore.similaritySearch.mockResolvedValue([]);
    return createWorkflowOrchestrator(
      'test question',
      {
        llm: mockLLM,
        vectorstore: mockVectorstore,
        webSearch: mockWebSearch,
        analyzer: mockAnalyzer,
        settings,
        pickedModels: testPickedModels,
      },
      {},
    );
  }

  function stubLLMForVectorstorePath() {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockReset();
    // Route decision → vectorstore
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    // HyDE expansion
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage for testing.',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockReset();
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
  }

  it('retrieves with the configured vectorstoreInitialK (not maxRetries) when maxRetries=1', async () => {
    const settings: Settings = {
      ...testSettings,
      maxRetries: 1,
      vectorstoreInitialK: 8,
    };
    stubLLMForVectorstorePath();
    const orch = makeOrchestrator(settings);

    await orch.run();

    const calls = (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls.length).toBeGreaterThan(0);
    for (const call of calls) {
      const k = call[1] as number | undefined;
      expect(k).toBe(8);
    }
  });

  it('retrieves with a sane default K when vectorstoreInitialK is unset, regardless of maxRetries', async () => {
    const settings: Settings = {
      ...testSettings,
      maxRetries: 1,
    };
    stubLLMForVectorstorePath();
    const orch = makeOrchestrator(settings);

    await orch.run();

    const calls = (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls.length).toBeGreaterThan(0);
    for (const call of calls) {
      const k = call[1] as number | undefined;
      expect(k).toBeGreaterThanOrEqual(4);
    }
  });

  it('respects an explicit vectorstoreInitialK=12 override independent of maxRetries=5', async () => {
    const settings: Settings = {
      ...testSettings,
      maxRetries: 5,
      vectorstoreInitialK: 12,
    };
    stubLLMForVectorstorePath();
    const orch = makeOrchestrator(settings);

    await orch.run();

    const calls = (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mock.calls;
    expect(calls.length).toBeGreaterThan(0);
    for (const call of calls) {
      const k = call[1] as number | undefined;
      expect(k).toBe(12);
    }
  });
});

describe('createWorkflowOrchestrator — error step context via RagError (issue #9)', () => {
  it('attaches the failing step name to RagError.step on retry exhaustion', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockReset();
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage.',
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockReset();
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    mockVectorstore.similaritySearch.mockReset();
    mockVectorstore.similaritySearch.mockRejectedValue(new Error('vectorstore down'));

    const orch = createWorkflowOrchestrator(
      'test',
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

    const state = await orch.run();
    expect(state.error).toBeDefined();
    expect(state.error?.step).toBe('vectorstore-similaritySearch');
  });
});

describe('createWorkflowOrchestrator — analyzer tool-call sub-steps and insights', () => {
  let orchestrator: ReturnType<typeof createWorkflowOrchestrator>;
  let stepUpdates: StepTrace[][] = [];

  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([]);
    stepUpdates = [];

    // Mock analyzer to capture hooks and allow test control
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockImplementation(
      async (question: string, _signal?: AbortSignal, _hooks?: AnalyzerHooks) => {
        return {
          type: 'data_analysis',
          question,
          code: '',
          explanation: 'Analysis complete',
          resultType: 'scalar',
          result: 'Analysis result',
          attempts: 1,
          durationMs: 100,
          timestamp: Date.now(),
          mode: 'tools',
          toolTrace: [{ tool: 'profile_column', calls: 1, durationMs: 5 }],
          insights: [{ finding: 'Revenue is up', evidence: 'sum 5000', confidence: 'high' }],
        };
      }
    );

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
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
      }
    );

    // Route to python path
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'python' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
  });

  it('emits tool-call sub-steps', async () => {
    // Override analyzer to fire hooks
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockImplementation(
      async (question: string, signal?: AbortSignal, hooks?: AnalyzerHooks) => {
        const startedAt = Date.now();
        hooks?.onToolStart?.({ tool: 'profile_column', argsSummary: '{}', startedAt });
        hooks?.onToolEnd?.({ tool: 'profile_column', durationMs: 5 });
        return {
          type: 'data_analysis',
          question,
          code: '',
          explanation: 'Analysis complete',
          resultType: 'scalar',
          result: 'Analysis result',
          attempts: 1,
          durationMs: 100,
          timestamp: Date.now(),
          mode: 'tools',
          toolTrace: [{ tool: 'profile_column', calls: 1, durationMs: 5 }],
        };
      }
    );

    await orchestrator.run();

    // Find the analyze sub-step
    const allSteps = stepUpdates.flat();
    const subStep = allSteps.find(s => s.node === 'analyze:profile_column');
    expect(subStep).toBeDefined();
    expect(subStep?.label).toBe('profile_column');
    expect(subStep?.status).toBe('done');
    expect(subStep?.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('tool error marks the sub-step error', async () => {
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockImplementation(
      async (question: string, signal?: AbortSignal, hooks?: AnalyzerHooks) => {
        const startedAt = Date.now();
        hooks?.onToolStart?.({ tool: 'profile_column', argsSummary: '{}', startedAt });
        hooks?.onToolEnd?.({ tool: 'profile_column', durationMs: 5, error: 'tool blew up' });
        return {
          type: 'data_analysis',
          question,
          code: '',
          explanation: 'Analysis complete',
          resultType: 'scalar',
          result: 'Analysis result',
          attempts: 1,
          durationMs: 100,
          timestamp: Date.now(),
          mode: 'tools',
          toolTrace: [{ tool: 'profile_column', calls: 1, durationMs: 5 }],
        };
      }
    );

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const subStep = allSteps.find(s => s.node === 'analyze:profile_column');
    expect(subStep).toBeDefined();
    expect(subStep?.status).toBe('error');
    expect(subStep?.detail).toBe('tool blew up');
  });

  it('records mode in the analyze step meta', async () => {
    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const analyzeStep = allSteps.find(s => s.node === 'analyze' && s.status === 'done');
    expect(analyzeStep).toBeDefined();
    expect(analyzeStep?.meta?.mode).toBe('tools');
    expect(analyzeStep?.meta?.toolCount).toBe(1);
    expect(analyzeStep?.meta?.fallbackReason).toBeUndefined();
  });

  it('insights appear in the generate context', async () => {
    await orchestrator.run();

    // Check the prompt sent to the LLM stream call
    const streamCalls = vi.mocked(mockLLM.stream).mock.calls;
    expect(streamCalls.length).toBeGreaterThan(0);
    const [request] = streamCalls[0]!;
    const prompt = request.messages[0]?.content || '';
    expect(prompt).toContain('Insights:');
    expect(prompt).toContain('Revenue is up');
    expect(prompt).toContain('sum 5000');
    expect(prompt).toContain('confidence: high');
  });
});

describe('createWorkflowOrchestrator — route datasource validation (FIX 1)', () => {
  it('an invalid route datasource falls back to vectorstore', async () => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([
      { id: '1', content: 'doc content', source: 'test.pdf', score: 0.9 },
    ]);

    // Route returns invalid datasource
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'data' }),
    });
    // HyDE expansion call
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage for testing.',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer based on doc',
      usage: undefined,
      model: 'answer-model',
    });

    const orchestrator = createWorkflowOrchestrator(
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

    const state = await orchestrator.run();

    expect(state.routing).toBe('vectorstore');
    expect(state.documents).toHaveLength(1);
    expect(state.answer).toBe('Answer based on doc');
  });
});

describe('createWorkflowOrchestrator — withRetry RagError preservation (FIX 2)', () => {
  it('a typed RagError from the analyzer keeps its message', async () => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([]);

    // Route to python path
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'python' }),
    });

    // Analyzer throws AnalysisBudgetExceededError
    const budgetError = new AnalysisBudgetExceededError({
      iterations: 9,
      elapsedMs: 1000,
      tokensUsed: 500,
      tripped: 'iterations',
      limit: 8,
    });
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockRejectedValue(budgetError);

    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    const orchestrator = createWorkflowOrchestrator(
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

    const state = await orchestrator.run();

    expect(state.error).toBeDefined();
    expect(state.error?.message).toContain('Analysis budget exceeded');
    expect(state.error?.message).not.toContain('Failed to generate response');
    expect(state.error?.code).toBe('ANALYSIS_BUDGET_EXCEEDED');
  });
});

describe('createWorkflowOrchestrator — fallback reason visibility (FIX 3)', () => {
  it('the fallback reason appears in the analyze step detail', async () => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([]);

    // Route to python path
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'python' }),
    });

    // Analyzer returns result with fallbackReason
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockResolvedValue({
      type: 'data_analysis',
      question: 'test',
      code: 'result = 1',
      explanation: 'One',
      resultType: 'scalar',
      result: 1,
      attempts: 1,
      durationMs: 100,
      timestamp: Date.now(),
      mode: 'fallback',
      fallbackReason: 'no-first-tool-call',
      toolTrace: [],
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    const stepUpdates: StepTrace[][] = [];
    const orchestrator = createWorkflowOrchestrator(
      'test question',
      {
        llm: mockLLM,
        vectorstore: mockVectorstore,
        webSearch: mockWebSearch,
        analyzer: mockAnalyzer,
        settings: testSettings,
        pickedModels: testPickedModels,
      },
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
      },
    );

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const analyzeStep = allSteps.find(s => s.node === 'analyze' && s.status === 'done');
    expect(analyzeStep).toBeDefined();
    expect(analyzeStep?.detail).toContain('fallback: no-first-tool-call');
  });
});

describe('createWorkflowOrchestrator — rAF coalescing of onPartialUpdate', () => {
  let orchestrator: ReturnType<typeof createWorkflowOrchestrator>;
  let stepUpdates: StepTrace[][] = [];
  let partialUpdates: WorkflowState[] = [];

  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([]);
    stepUpdates = [];
    partialUpdates = [];

    // Mock analyzer to return quickly
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockResolvedValue({
      type: 'data_analysis',
      question: 'test',
      code: '',
      explanation: 'Analysis complete',
      resultType: 'scalar',
      result: 'Analysis result',
      attempts: 1,
      durationMs: 10,
      timestamp: Date.now(),
      mode: 'tools',
      toolTrace: [],
      insights: [],
    });

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
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
        onPartialUpdate: (state) => partialUpdates.push(state),
      }
    );

    // Route to python path
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'python' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
  });

  it('coalesces rapid onPartialUpdate calls within one frame', async () => {
    // The orchestrator emits: route -> analyze -> generate -> evaluate -> end
    // That's 5 step updates. onPartialUpdate should fire fewer times due to rAF batching.
    await orchestrator.run();

    // onStepUpdate fires for each step (at least 5: route, analyze, generate, evaluate, end)
    expect(stepUpdates.length).toBeGreaterThanOrEqual(5);

    // onPartialUpdate should fire fewer times due to rAF coalescing
    // (exact count depends on timing, but should be less than stepUpdates)
    expect(partialUpdates.length).toBeLessThan(stepUpdates.length);

    // Final state should still arrive with all steps completed
    const finalState = partialUpdates[partialUpdates.length - 1];
    expect(finalState).toBeDefined();
    expect(finalState.steps.length).toBeGreaterThanOrEqual(5);
    expect(finalState.finishedAt).toBeDefined();
  });
});

describe('createWorkflowOrchestrator — step-level observability (retries + tokens)', () => {
  let orchestrator: ReturnType<typeof createWorkflowOrchestrator>;
  let stepUpdates: StepTrace[][] = [];

  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([]);
    stepUpdates = [];

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
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
      }
    );
  });

  it('records retry events in step retries array with attempt, error, delayMs', async () => {
    // Route to vectorstore
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    // HyDE expansion succeeds
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage for testing.',
      usage: { totalTokens: 50 },
    });
    // vectorstore-similaritySearch fails on first call with a retryable error, succeeds on subsequent calls
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce(new ProviderUnreachableError('test-provider', new Error('rate limited')))
      .mockResolvedValue([
        { id: '1', content: 'doc', source: 'a.pdf', score: 0.9 },
      ]);
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: { totalTokens: 100 },
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    // The retry happens on the vectorstore-similaritySearch call which is part of the retrieve step
    const retrieveStep = allSteps.find(s => s.node === 'retrieve');
    expect(retrieveStep).toBeDefined();
    expect(retrieveStep?.retries).toBeDefined();
    expect(retrieveStep?.retries?.length).toBe(1);
    expect(retrieveStep?.retries?.[0].attempt).toBe(1);
    expect(retrieveStep?.retries?.[0].error).toContain('Cannot reach test-provider');
    expect(retrieveStep?.retries?.[0].delayMs).toBe(1000); // BASE_RETRY_DELAY_MS * 2^0
  });

  it('records tokensUsed in step meta for route step', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
      usage: { totalTokens: 25 },
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage.',
      usage: { totalTokens: 30 },
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
      { id: '1', content: 'doc', source: 'a.pdf', score: 0.9 },
    ]);
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: { totalTokens: 100 },
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const routeStep = allSteps.find(s => s.node === 'route');
    expect(routeStep).toBeDefined();
    expect(routeStep?.meta?.tokensUsed).toBe(25);
  });

  it('records tokensUsed in step meta for retrieve step (includes HyDE tokens)', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage.',
      usage: { totalTokens: 30 },
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
      { id: '1', content: 'doc', source: 'a.pdf', score: 0.9 },
    ]);
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: { totalTokens: 100 },
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const retrieveStep = allSteps.find(s => s.node === 'retrieve');
    expect(retrieveStep).toBeDefined();
    expect(retrieveStep?.meta?.tokensUsed).toBe(30);
  });

  it('records tokensUsed in step meta for generate step', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage.',
    });
    (mockVectorstore.similaritySearch as ReturnType<typeof vi.fn>).mockResolvedValue([
      { id: '1', content: 'doc', source: 'a.pdf', score: 0.9 },
    ]);
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: { totalTokens: 150 },
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const generateStep = allSteps.find(s => s.node === 'generate');
    expect(generateStep).toBeDefined();
    expect(generateStep?.meta?.tokensUsed).toBe(150);
  });

  it('records tokensUsed in step meta for evaluate step (sum of both invokes)', async () => {
    // Reset all mocks for this test
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([
      { id: '1', content: 'doc', source: 'a.pdf', score: 0.9 },
    ]);
    stepUpdates = [];

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
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
      }
    );

    // Set up mocks in the exact order they will be called:
    // 1. Route decision
    // 2. HyDE expansion
    // 3. Grade docs (multiple calls, one per doc)
    // 4. Generate (stream)
    // 5. Evaluate hallucination check
    // 6. Evaluate answer check
    let invokeCallCount = 0;
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      invokeCallCount++;
      if (invokeCallCount === 1) {
        // Route
        return { content: JSON.stringify({ datasource: 'vectorstore' }) };
      }
      if (invokeCallCount === 2) {
        // HyDE
        return { content: 'Hypothetical passage.' };
      }
      if (invokeCallCount === 3) {
        // Grade doc (only 1 doc)
        return { content: JSON.stringify({ binary_score: 'yes' }) };
      }
      if (invokeCallCount === 4) {
        // Evaluate hallucination
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 20 } };
      }
      if (invokeCallCount === 5) {
        // Evaluate answer
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 30 } };
      }
      // Fallback
      return { content: JSON.stringify({ binary_score: 'yes' }) };
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: { totalTokens: 100 },
      model: 'answer-model',
    });

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const evaluateStep = allSteps.find(s => s.node === 'evaluate');
    expect(evaluateStep).toBeDefined();
    expect(evaluateStep?.meta?.tokensUsed).toBe(50); // 20 + 30
  });

  it('threads previousAnalysis into the analyzer call for cross-turn memory', async () => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    stepUpdates = [];

    const previousAnalysis: DataAnalysisResult = {
      type: 'data_analysis',
      question: 'previous question',
      code: 'result = 1',
      explanation: 'prior explanation',
      resultType: 'scalar',
      result: null,
      attempts: 1,
      durationMs: 5,
      timestamp: Date.now(),
      insights: [{ finding: 'prior finding', evidence: 'prior evidence', confidence: 'high' }],
    };

    orchestrator = createWorkflowOrchestrator(
      'test question',
      {
        llm: mockLLM,
        vectorstore: mockVectorstore,
        webSearch: mockWebSearch,
        analyzer: mockAnalyzer,
        settings: testSettings,
        pickedModels: testPickedModels,
        previousAnalysis,
      },
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
      }
    );

    let invokeCallCount = 0;
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      invokeCallCount++;
      if (invokeCallCount === 1) {
        // Route -> python
        return { content: JSON.stringify({ datasource: 'python' }) };
      }
      if (invokeCallCount === 2) {
        // Evaluate hallucination check
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      // Evaluate answer check
      return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
    });
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockResolvedValue(previousAnalysis);
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      model: 'answer-model',
    });

    await orchestrator.run();

    expect(mockAnalyzer.analyze).toHaveBeenCalledWith(
      'test question',
      undefined,
      expect.anything(),
      expect.stringContaining('prior finding')
    );
  });
});

describe('createWorkflowOrchestrator — query rewriting on eval failure (Unit 1)', () => {
  let orchestrator: ReturnType<typeof createWorkflowOrchestrator>;
  let stepUpdates: StepTrace[][] = [];

  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([
      { id: '1', content: 'doc content', source: 'test.pdf', score: 0.9 },
    ]);
    stepUpdates = [];

    orchestrator = createWorkflowOrchestrator(
      'original question about data',
      {
        llm: mockLLM,
        vectorstore: mockVectorstore,
        webSearch: mockWebSearch,
        analyzer: mockAnalyzer,
        settings: testSettings,
        pickedModels: testPickedModels,
      },
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
      }
    );
  });

  it('rewrites the question on eval failure and uses rewritten question for retrieval', async () => {
    let invokeCallCount = 0;
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      invokeCallCount++;
      if (invokeCallCount === 1) {
        return { content: JSON.stringify({ datasource: 'vectorstore' }) };
      }
      if (invokeCallCount === 2) {
        return { content: 'Hypothetical passage.' };
      }
      if (invokeCallCount === 3) {
        return { content: JSON.stringify({ binary_score: 'yes' }) };
      }
      if (invokeCallCount === 4) {
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      if (invokeCallCount === 5) {
        return { content: JSON.stringify({ binary_score: 'no' }), usage: { totalTokens: 10 } };
      }
      if (invokeCallCount === 6) {
        return { content: JSON.stringify({ question: 'rewritten question for web search' }), usage: { totalTokens: 20 } };
      }
      if (invokeCallCount === 7) {
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      if (invokeCallCount === 8) {
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      return { content: JSON.stringify({ binary_score: 'yes' }) };
    });

    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });

    (mockWebSearch.search as ReturnType<typeof vi.fn>).mockResolvedValue([
      { type: 'web_search', title: 'Web Result', content: 'Web content for rewritten question', url: 'http://example.com' },
    ]);

    const state = await orchestrator.run();

    expect(state.retryCount).toBe(1);
    expect(state.routing).toBe('websearch');
    expect(state.webResults).toHaveLength(1);
    expect(state.webResults[0].content).toBe('Web content for rewritten question');

    const rewriteCall = vi.mocked(mockLLM.invoke).mock.calls.find(
      (call) => call[0]?.messages?.[0]?.content?.includes('Rewrite this question to maximize retrieval effectiveness for websearch')
    );
    expect(rewriteCall).toBeDefined();

    const webSearchCalls = vi.mocked(mockWebSearch.search).mock.calls;
    expect(webSearchCalls.length).toBe(1);
    expect(webSearchCalls[0][0]).toBe('rewritten question for web search');
  });

  it('falls back to original question when rewrite parse fails', async () => {
    let invokeCallCount = 0;
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      invokeCallCount++;
      if (invokeCallCount === 1) {
        return { content: JSON.stringify({ datasource: 'vectorstore' }) };
      }
      if (invokeCallCount === 2) {
        return { content: 'Hypothetical passage.' };
      }
      if (invokeCallCount === 3) {
        return { content: JSON.stringify({ binary_score: 'yes' }) };
      }
      if (invokeCallCount === 4) {
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      if (invokeCallCount === 5) {
        return { content: JSON.stringify({ binary_score: 'no' }), usage: { totalTokens: 10 } };
      }
      if (invokeCallCount === 6) {
        return { content: 'not valid json', usage: { totalTokens: 20 } };
      }
      if (invokeCallCount === 7) {
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      if (invokeCallCount === 8) {
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      return { content: JSON.stringify({ binary_score: 'yes' }) };
    });

    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });

    (mockWebSearch.search as ReturnType<typeof vi.fn>).mockResolvedValue([
      { type: 'web_search', title: 'Web Result', content: 'Web content for original', url: 'http://example.com' },
    ]);

    const state = await orchestrator.run();

    expect(state.retryCount).toBe(1);
    expect(state.routing).toBe('websearch');
    const webSearchCalls = vi.mocked(mockWebSearch.search).mock.calls;
    expect(webSearchCalls[0][0]).toBe('original question about data');
  });

  it('generate step still uses original question in prompt even after rewrite', async () => {
    let invokeCallCount = 0;
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockImplementation(async () => {
      invokeCallCount++;
      if (invokeCallCount === 1) {
        return { content: JSON.stringify({ datasource: 'vectorstore' }) };
      }
      if (invokeCallCount === 2) {
        return { content: 'Hypothetical passage.' };
      }
      if (invokeCallCount === 3) {
        return { content: JSON.stringify({ binary_score: 'yes' }) };
      }
      if (invokeCallCount === 4) {
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      if (invokeCallCount === 5) {
        return { content: JSON.stringify({ binary_score: 'no' }), usage: { totalTokens: 10 } };
      }
      if (invokeCallCount === 6) {
        return { content: JSON.stringify({ question: 'rewritten for websearch' }), usage: { totalTokens: 20 } };
      }
      if (invokeCallCount === 7) {
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      if (invokeCallCount === 8) {
        return { content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } };
      }
      return { content: JSON.stringify({ binary_score: 'yes' }) };
    });

    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });

    (mockWebSearch.search as ReturnType<typeof vi.fn>).mockResolvedValue([
      { type: 'web_search', title: 'Web', content: 'Web content', url: 'http://example.com' },
    ]);

    await orchestrator.run();

    const streamCalls = vi.mocked(mockLLM.stream).mock.calls;
    const retryGenerateCall = streamCalls[1];
    expect(retryGenerateCall).toBeDefined();
    const prompt = retryGenerateCall?.[0]?.messages?.[0]?.content || '';
    expect(prompt).toContain('original question about data');
    expect(prompt).not.toContain('rewritten for websearch');
  });
});

describe('createWorkflowOrchestrator — plan on first iteration (Unit 2)', () => {
  let orchestrator: ReturnType<typeof createWorkflowOrchestrator>;
  let stepUpdates: StepTrace[][] = [];

  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([]);
    stepUpdates = [];

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
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
      }
    );
  });

  it('analyze step meta carries plan from first iteration and reflections from subsequent ones', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'python' }),
    });

    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockImplementation(
      async (question: string, _signal?: AbortSignal, hooks?: AnalyzerHooks) => {
        if (hooks?.onIteration) {
          hooks.onIteration({ iteration: 1, reflection: 'PLAN: Will list datasets then profile salary', tokensUsed: 100 });
          hooks.onIteration({ iteration: 2, reflection: 'REFLECTION: Found datasets, will profile salary', tokensUsed: 150 });
          hooks.onIteration({ iteration: 3, reflection: 'REFLECTION: Profiled salary, computing average', tokensUsed: 200 });
        }
        return {
          type: 'data_analysis',
          question,
          code: '',
          explanation: 'Analysis complete',
          resultType: 'scalar',
          result: 'Analysis result',
          attempts: 3,
          durationMs: 100,
          timestamp: Date.now(),
          mode: 'tools',
          toolTrace: [{ tool: 'profile_column', calls: 1, durationMs: 5 }],
          insights: [{ finding: 'Revenue is up', evidence: 'sum 5000', confidence: 'high' }],
        };
      }
    );

    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const analyzeStep = allSteps.find(s => s.node === 'analyze' && s.status === 'done');
    expect(analyzeStep).toBeDefined();
    expect(analyzeStep?.meta?.plan).toBe('Will list datasets then profile salary');
    expect(analyzeStep?.meta?.reflections).toEqual([
      'Found datasets, will profile salary',
      'Profiled salary, computing average',
    ]);
  });

  it('handles backward compatibility when reflection has no prefix', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'python' }),
    });

    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockImplementation(
      async (question: string, _signal?: AbortSignal, hooks?: AnalyzerHooks) => {
        if (hooks?.onIteration) {
          hooks.onIteration({ iteration: 1, reflection: 'Legacy reflection without prefix', tokensUsed: 100 });
        }
        return {
          type: 'data_analysis',
          question,
          code: '',
          explanation: 'Analysis complete',
          resultType: 'scalar',
          result: 'Analysis result',
          attempts: 1,
          durationMs: 100,
          timestamp: Date.now(),
          mode: 'tools',
          toolTrace: [],
          insights: [],
        };
      }
    );

    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const analyzeStep = allSteps.find(s => s.node === 'analyze' && s.status === 'done');
    expect(analyzeStep).toBeDefined();
    expect(analyzeStep?.meta?.plan).toBeUndefined();
    expect(analyzeStep?.meta?.reflections).toEqual(['Legacy reflection without prefix']);
  });
});

describe('createWorkflowOrchestrator — router confidence + multi-source fallback (Unit 3)', () => {
  let orchestrator: ReturnType<typeof createWorkflowOrchestrator>;
  let stepUpdates: StepTrace[][] = [];

  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([
      { id: '1', content: 'doc content', source: 'test.pdf', score: 0.9 },
    ]);
    stepUpdates = [];

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
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
      }
    );
  });

  it('high confidence route runs single path only', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore', confidence: 0.9 }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage.',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } });

    const state = await orchestrator.run();

    expect(state.routing).toBe('vectorstore');
    expect(state.documents).toHaveLength(1);
    expect(state.webResults).toHaveLength(0);

    const allSteps = stepUpdates.flat();
    const routeStep = allSteps.find(s => s.node === 'route');
    expect(routeStep?.detail).toContain('confidence: 0.90');
    expect(routeStep?.detail).not.toContain('multi-source');
    expect(routeStep?.meta?.confidence).toBe(0.9);
  });

  it('low confidence route runs both vectorstore and websearch concurrently', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore', confidence: 0.4 }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage.',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } });

    (mockWebSearch.search as ReturnType<typeof vi.fn>).mockResolvedValue([
      { type: 'web_search', title: 'Web Result', content: 'Web content', url: 'http://example.com' },
    ]);

    const state = await orchestrator.run();

    expect(state.routing).toBe('vectorstore');
    expect(state.documents).toHaveLength(1);
    expect(state.webResults).toHaveLength(1);

    const allSteps = stepUpdates.flat();
    const routeStep = allSteps.find(s => s.node === 'route');
    expect(routeStep?.detail).toContain('confidence: 0.40');
    expect(routeStep?.detail).toContain('multi-source');
    expect(routeStep?.meta?.confidence).toBe(0.4);

    expect(mockVectorstore.similaritySearch).toHaveBeenCalled();
    expect(mockWebSearch.search).toHaveBeenCalled();
  });

  it('missing confidence defaults to high confidence (single path)', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore' }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage.',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } });

    const state = await orchestrator.run();

    expect(state.routing).toBe('vectorstore');
    expect(state.documents).toHaveLength(1);
    expect(state.webResults).toHaveLength(0);

    const allSteps = stepUpdates.flat();
    const routeStep = allSteps.find(s => s.node === 'route');
    expect(routeStep?.meta?.confidence).toBe(1.0);
  });

  it('invalid confidence defaults to high confidence (single path)', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore', confidence: 'high' }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage.',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } });

    const state = await orchestrator.run();

    expect(state.routing).toBe('vectorstore');
    expect(state.webResults).toHaveLength(0);

    const allSteps = stepUpdates.flat();
    const routeStep = allSteps.find(s => s.node === 'route');
    expect(routeStep?.meta?.confidence).toBe(1.0);
  });

  it('confidence out of range defaults to high confidence', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'vectorstore', confidence: 1.5 }),
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: 'Hypothetical passage.',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });
    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } })
      .mockResolvedValueOnce({ content: JSON.stringify({ binary_score: 'yes' }), usage: { totalTokens: 10 } });

    await orchestrator.run();

    const allSteps = stepUpdates.flat();
    const routeStep = allSteps.find(s => s.node === 'route');
    expect(routeStep?.meta?.confidence).toBe(1.0);
  });
});

describe('createWorkflowOrchestrator — onSynthesisToken wiring (Unit 4)', () => {
  let _orchestrator: ReturnType<typeof createWorkflowOrchestrator>;
  let stepUpdates: StepTrace[][] = [];

  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorstore.load.mockResolvedValue(undefined);
    mockVectorstore.stats = { entries: 0 };
    mockVectorstore.similaritySearch.mockResolvedValue([]);
    stepUpdates = [];

    _orchestrator = createWorkflowOrchestrator(
      'test question',
      {
        llm: mockLLM,
        vectorstore: mockVectorstore,
        webSearch: mockWebSearch,
        analyzer: mockAnalyzer,
        settings: testSettings,
        pickedModels: testPickedModels,
      },
      {
        onStepUpdate: (steps) => stepUpdates.push(steps),
      }
    );
  });

  it('onSynthesisToken tokens from analyzer salvage reach the onToken callback', async () => {
    const receivedTokens: string[] = [];

    // Route to python path
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      content: JSON.stringify({ datasource: 'python' }),
    });

    // Mock analyzer to throw AnalysisBudgetExceededError so real analyzer salvage path runs
    // We use the REAL analyzer via createDataAnalyzer with a mocked LLM
    // But since the orchestrator tests use a mock analyzer, we simulate the salvage by
    // having the mock analyzer's analyze invoke the hooks.onSynthesisToken it receives
    (mockAnalyzer.analyze as ReturnType<typeof vi.fn>).mockImplementation(
      async (question: string, _signal?: AbortSignal, hooks?: AnalyzerHooks) => {
        // Simulate salvage synthesis streaming tokens via onSynthesisToken
        hooks?.onSynthesisToken?.('Salvaged ');
        hooks?.onSynthesisToken?.('answer ');
        hooks?.onSynthesisToken?.('streamed.');

        return {
          type: 'data_analysis',
          question,
          code: '',
          explanation: 'Salvaged answer streamed.',
          resultType: 'scalar',
          result: 'Salvaged answer streamed.',
          attempts: 3,
          durationMs: 100,
          timestamp: Date.now(),
          mode: 'fallback',
          fallbackReason: 'salvaged-tokens',
          partial: true,
          toolTrace: [{ tool: 'list_datasets', calls: 1, durationMs: 5, tokensUsed: 50 }],
          insights: [],
        };
      }
    );

    (mockLLM.stream as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: 'Answer',
      usage: undefined,
      model: 'answer-model',
    });
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({ binary_score: 'yes' }),
    });

    // Create orchestrator with onToken callback
    const tokenOrchestrator = createWorkflowOrchestrator(
      'test question',
      {
        llm: mockLLM,
        vectorstore: mockVectorstore,
        webSearch: mockWebSearch,
        analyzer: mockAnalyzer,
        settings: testSettings,
        pickedModels: testPickedModels,
      },
      {
        onToken: (token) => receivedTokens.push(token),
      }
    );

    await tokenOrchestrator.run();

    // Assert the tokens arrive at the onToken callback
    expect(receivedTokens).toEqual(['Salvaged ', 'answer ', 'streamed.']);
  });
});
});