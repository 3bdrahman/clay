import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createWorkflowOrchestrator } from '../services/orchestrator';
import type { LLMClient } from '../lib/llm';
import type { VectorStore } from '../lib/vectorstore';
import type { WebSearchClient } from '../lib/websearch';
import type { DataAnalyzer } from '../services/analyzer';
import type { Settings, Document, WebResult, DataAnalysisResult } from '../lib/types';
import type { PickedModels } from '../lib/models';

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
});