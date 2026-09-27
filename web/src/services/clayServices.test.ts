import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as aq from 'arquero';
import { createClayServiceBundle } from './clayServices';
import { resolveProviderEndpoint } from '../lib/providers';
import { ModelNotFoundError } from '../lib/errors';
import { EMBEDDING_MODEL_ID } from '../lib/embeddingModel';
import type { Settings } from '../lib/types';

const createEmbeddingsClientMock = vi.fn();
const createVectorStoreMock = vi.fn();
const createWebSearchClientMock = vi.fn();
const createLLMClientMock = vi.fn();
const createDataAnalyzerMock = vi.fn();

vi.mock('../lib/embeddings', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../lib/embeddings')>()),
  createEmbeddingsClient: (...args: unknown[]) => createEmbeddingsClientMock(...args),
}));
vi.mock('../lib/vectorstore', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../lib/vectorstore')>()),
  createVectorStore: (...args: unknown[]) => createVectorStoreMock(...args),
}));
vi.mock('../lib/websearch', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../lib/websearch')>()),
  createWebSearchClient: (...args: unknown[]) => createWebSearchClientMock(...args),
}));
vi.mock('../lib/llm', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../lib/llm')>()),
  createLLMClient: (...args: unknown[]) => createLLMClientMock(...args),
}));
vi.mock('./analyzer', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./analyzer')>()),
  createDataAnalyzer: (...args: unknown[]) => createDataAnalyzerMock(...args),
}));

const embeddingsStub = { embed: vi.fn() };
const vectorStoreStub = {
  load: vi.fn(),
  stats: { entries: 0 },
  persistenceAvailable: true,
};
const webSearchStub = {};
const llmStub = { invoke: vi.fn() };
const analyzerStub = {};

function baseSettings(overrides: Partial<Settings>): Settings {
  return {
    provider: 'openrouter',
    openrouterApiKey: '',
    groqApiKey: '',
    apiKey: '',
    webSearchProvider: 'duckduckgo',
    serperApiKey: '',
    temperature: 0.2,
    maxRetries: 2,
    theme: 'system',
    localServerUrl: '',
    localModels: { chat: '' },
    localCatalog: [],
    localCatalogFetchedAt: 0,
    pickedModelsOverride: { chatModel: '' },
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  createEmbeddingsClientMock.mockReturnValue(embeddingsStub);
  createVectorStoreMock.mockReturnValue(vectorStoreStub);
  createWebSearchClientMock.mockReturnValue(webSearchStub);
  createLLMClientMock.mockReturnValue(llmStub);
  createDataAnalyzerMock.mockReturnValue(analyzerStub);
});

describe('createClayServiceBundle', () => {
  const catalog = [
    { id: 'meta-llama/llama-3.3-70b-instruct', ownedBy: 'meta', created: 1 },
    { id: 'text-embedding-3-small', ownedBy: 'openai', created: 2 },
  ];

  it('embeddings are local: created with no provider config, vectorstore pinned to the fixed model', () => {
    const settings = baseSettings({
      apiKey: 'k1',
      pickedModelsOverride: { chatModel: 'meta-llama/llama-3.3-70b-instruct' },
    });
    const endpoint = resolveProviderEndpoint(settings);

    const bundle = createClayServiceBundle({
      settings,
      catalog,
      analyzerTables: new Map([['employees', aq.from([{ a: 1 }])]]),
      analyzerMetadata: { employees: { columns: ['a'], rowCount: 1 } },
    });

    expect(createEmbeddingsClientMock).toHaveBeenCalledWith();
    expect(createVectorStoreMock).toHaveBeenCalledWith(embeddingsStub, {
      embeddingModel: EMBEDDING_MODEL_ID,
    });
    expect(createLLMClientMock).toHaveBeenCalledWith({
      baseUrl: endpoint.baseUrl,
      apiKey: 'k1',
      temperature: 0.2,
      providerLabel: endpoint.providerLabel,
    });
    expect(bundle.pickedModels).toEqual({ chat: 'meta-llama/llama-3.3-70b-instruct' });
    expect(bundle.vectorstore).toBe(vectorStoreStub);
    expect(bundle.llm).toBe(llmStub);
  });

  it("adds the 'aq' namespace to the analyzer datasets alongside the adapter's tables", () => {
    const settings = baseSettings({ apiKey: 'k1' });

    createClayServiceBundle({
      settings,
      catalog,
      analyzerTables: new Map([['employees', aq.from([{ a: 1 }])]]),
      analyzerMetadata: { employees: { columns: ['a'], rowCount: 1 } },
    });

    const analyzerArgs = createDataAnalyzerMock.mock.calls[0][0] as {
      datasets: Map<string, unknown>;
      metadata: { employees: { columns: string[]; rowCount: number } };
      codeGenModel: string | undefined;
    };
    expect(analyzerArgs.datasets.get('aq')).toBe(aq);
    expect(analyzerArgs.datasets.get('employees')).toBeDefined();
    expect(analyzerArgs.metadata).toEqual({ employees: { columns: ['a'], rowCount: 1 } });
  });

  it('passes maxToolLoopTokens through to the analyzer (undefined when unset)', () => {
    const settings = baseSettings({ apiKey: 'k1', maxToolLoopTokens: 123_456 });
    createClayServiceBundle({
      settings,
      catalog,
      analyzerTables: new Map(),
      analyzerMetadata: {},
    });
    const analyzerArgs = createDataAnalyzerMock.mock.calls[0][0] as { maxToolLoopTokens: number | undefined };
    expect(analyzerArgs.maxToolLoopTokens).toBe(123_456);

    const unsetSettings = baseSettings({ apiKey: 'k1' });
    createClayServiceBundle({
      settings: unsetSettings,
      catalog,
      analyzerTables: new Map(),
      analyzerMetadata: {},
    });
    const unsetArgs = createDataAnalyzerMock.mock.calls[1][0] as { maxToolLoopTokens: number | undefined };
    expect(unsetArgs.maxToolLoopTokens).toBeUndefined();
  });

  it('local provider: user chat pick validated against the catalog', () => {
    const settings = baseSettings({
      provider: 'local',
      localServerUrl: 'http://localhost:11434/v1',
      localModels: { chat: 'm-chat' },
    });
    const endpoint = resolveProviderEndpoint(settings);
    const localCatalog = [
      { id: 'm-chat', ownedBy: '', created: 1 },
      { id: 'm-embed', ownedBy: '', created: 2 },
    ];

    const bundle = createClayServiceBundle({
      settings,
      catalog: localCatalog,
      analyzerTables: new Map(),
      analyzerMetadata: {},
    });

    expect(createEmbeddingsClientMock).toHaveBeenCalledWith();
    expect(createVectorStoreMock).toHaveBeenCalledWith(embeddingsStub, {
      embeddingModel: EMBEDDING_MODEL_ID,
    });
    expect(bundle.pickedModels).toEqual({ chat: 'm-chat' });
    expect(endpoint.baseUrl).toBe('http://localhost:11434/v1');
  });

  it('local provider: throws ModelNotFoundError when a pick is missing from the catalog', () => {
    const settings = baseSettings({
      provider: 'local',
      localServerUrl: 'http://localhost:11434/v1',
      localModels: { chat: 'gone' },
    });
    const localCatalog = [{ id: 'm-chat', ownedBy: '', created: 1 }];

    expect(() =>
      createClayServiceBundle({
        settings,
        catalog: localCatalog,
        analyzerTables: new Map(),
        analyzerMetadata: {},
      }),
    ).toThrow(ModelNotFoundError);
  });
});
