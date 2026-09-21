import { describe, expect, it, vi, beforeEach, afterAll } from 'vitest';
import {
  modelClass,
  pickBestEmbedding,
  pickLocalModels,
  listModels,
  listLocalCatalog,
  resolveModels,
  ModelNotFoundError,
  InvalidApiKeyError,
  ProviderUnreachableError,
  RateLimitError,
  ModelCatalogEmptyError,
  type ModelInfo,
} from './models';
import type { Settings, LocalModelPicks } from './types';
import { LOCAL_DEFAULT_BASE_URL } from './providers';

const fakeModels: ModelInfo[] = [
  // Embeddings
  { id: 'nvidia/nv-embedqa-e5-v5', ownedBy: 'nvidia', created: 0 },
  { id: 'nvidia/nv-embedqa-mistral-7b-v2', ownedBy: 'nvidia', created: 0 },
  // Code specialists
  { id: 'mistralai/codestral-22b-instruct-v0.1', ownedBy: 'mistralai', created: 0 },
  { id: 'meta/codellama-70b-instruct', ownedBy: 'meta', created: 0 },
  // Tiny chat
  { id: 'meta/llama-3.1-8b-instruct', ownedBy: 'meta', created: 0 },
  { id: 'mistralai/mistral-7b-instruct-v0.3', ownedBy: 'mistralai', created: 0 },
  // Huge chat
  { id: 'nvidia/nemotron-3-ultra-550b-a55b', ownedBy: 'nvidia', created: 0 },
  { id: 'nvidia/nemotron-4-340b-instruct', ownedBy: 'nvidia', created: 0 },
  // Vision (should be excluded)
  { id: 'meta/llama-3.2-11b-vision-instruct', ownedBy: 'meta', created: 0 },
  // Safety (should be excluded)
  { id: 'meta/llama-guard-3-8b', ownedBy: 'meta', created: 0 },
];

describe('modelClass', () => {
  it('classifies tiny models', () => {
    expect(modelClass('meta/llama-3.2-1b-instruct')).toBe('tiny');
    expect(modelClass('mistralai/mistral-mini-3b-instruct')).toBe('tiny');
  });

  it('classifies small models', () => {
    expect(modelClass('meta/llama-3.1-8b-instruct')).toBe('small');
    expect(modelClass('mistralai/mistral-7b-instruct')).toBe('small');
  });

  it('classifies medium models', () => {
    expect(modelClass('mistralai/codestral-22b-instruct')).toBe('medium');
    expect(modelClass('meta/llama-3.1-13b-instruct')).toBe('medium');
  });

  it('classifies large models', () => {
    expect(modelClass('meta/llama-3.1-70b-instruct')).toBe('large');
  });

  it('classifies huge models', () => {
    expect(modelClass('nvidia/nemotron-3-ultra-550b-a55b')).toBe('huge');
    expect(modelClass('nvidia/nemotron-4-340b-instruct')).toBe('huge');
  });

  it('defaults unknown models to medium', () => {
    expect(modelClass('foo/bar')).toBe('medium');
  });
});

describe('pickBestEmbedding', () => {
  it('picks the highest-scoring embedding model', () => {
    expect(pickBestEmbedding(fakeModels)).toBe('nvidia/nv-embedqa-e5-v5');
  });

  it('ignores non-embedding models when scoring', () => {
    expect(pickBestEmbedding([
      { id: 'meta/llama-3.1-8b-instruct', ownedBy: 'meta', created: 0 },
      { id: 'snowflake/arctic-embed-l', ownedBy: 'snowflake', created: 0 },
    ])).toBe('snowflake/arctic-embed-l');
  });

  it('returns undefined when the catalog has no embeddings', () => {
    expect(pickBestEmbedding([
      { id: 'meta/llama-3.1-8b-instruct', ownedBy: 'meta', created: 0 },
      { id: 'meta/llama-guard-3-8b', ownedBy: 'meta', created: 0 },
    ])).toBeUndefined();
  });

  it('returns undefined for an empty catalog', () => {
    expect(pickBestEmbedding([])).toBeUndefined();
  });
});

describe('listModels', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
    globalThis.fetch = vi.fn();
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
  });

  it('throws InvalidApiKeyError when API key is missing', async () => {
    await expect(listModels('openrouter', '')).rejects.toThrow(InvalidApiKeyError);
    await expect(listModels('openrouter', '')).rejects.toThrow('Invalid API key');
  });

  it('throws InvalidApiKeyError on 401 response', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('Unauthorized', { status: 401 })
    );
    try {
      await expect(listModels('openrouter', 'test-key')).rejects.toThrow(InvalidApiKeyError);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('throws ProviderUnreachableError on 500 response', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('Server Error', { status: 500 })
    );
    try {
      await expect(listModels('openrouter', 'test-key')).rejects.toThrow(ProviderUnreachableError);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('throws RateLimitError on 429 response', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('Rate Limited', { status: 429 })
    );
    try {
      await expect(listModels('openrouter', 'test-key')).rejects.toThrow(RateLimitError);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('throws ModelCatalogEmptyError on empty catalog', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ data: [] }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    );
    try {
      await expect(listModels('openrouter', 'test-key')).rejects.toThrow(ModelCatalogEmptyError);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('parses the catalog response', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [
            { id: 'meta/llama-3.1-8b-instruct', owned_by: 'meta', created: 123 },
            { id: 'nvidia/nv-embedqa-e5-v5', owned_by: 'nvidia', created: 456 },
          ],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    );
    try {
      const models = await listModels('openrouter', 'test-key');
      expect(models).toHaveLength(2);
      expect(models[0]).toEqual({ id: 'meta/llama-3.1-8b-instruct', ownedBy: 'meta', created: 123 });
      expect(models[1]).toEqual({ id: 'nvidia/nv-embedqa-e5-v5', ownedBy: 'nvidia', created: 456 });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('handles missing optional fields', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ data: [{ id: 'test/model' }] }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    );
    try {
      const models = await listModels('openrouter', 'test-key');
      expect(models[0]).toEqual({ id: 'test/model', ownedBy: '', created: 0 });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

describe('listLocalCatalog', () => {
  const mockFetch = vi.fn();
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
    globalThis.fetch = mockFetch;
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
  });

  it('throws ProviderUnreachableError when baseUrl is empty', async () => {
    await expect(listLocalCatalog('', '')).rejects.toThrow(ProviderUnreachableError);
  });

  it('throws InvalidApiKeyError on 401 response', async () => {
    mockFetch.mockResolvedValue(new Response('Unauthorized', { status: 401 }));
    await expect(listLocalCatalog('http://localhost:11434/v1', '')).rejects.toThrow(
      InvalidApiKeyError,
    );
  });

  it('throws ProviderUnreachableError on 502 response', async () => {
    mockFetch.mockResolvedValue(new Response('boom', { status: 502 }));
    await expect(listLocalCatalog('http://localhost:11434/v1', '')).rejects.toThrow(
      ProviderUnreachableError,
    );
  });

  it('parses the local OpenAI-compatible /models response', async () => {
    mockFetch.mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [
            { id: 'llama3.1:8b-instruct-q5_K_M', object: 'model' },
            { id: 'nomic-embed-text', object: 'model' },
          ],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      ),
    );

    const out = await listLocalCatalog('http://localhost:11434/v1/', '');
    expect(out).toHaveLength(2);
    expect(out[0]).toEqual({ id: 'llama3.1:8b-instruct-q5_K_M', ownedBy: '', created: 0 });
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:11434/v1/models',
      expect.objectContaining({ headers: expect.not.objectContaining({ Authorization: expect.anything() }) }),
    );
  });

  it('strips trailing slashes before appending /models', async () => {
    mockFetch.mockResolvedValue(
      new Response(JSON.stringify({ data: [{ id: 'test-model', object: 'model' }] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    await listLocalCatalog('http://localhost:1234/v1///', '');
    expect(mockFetch.mock.calls[0][0]).toBe('http://localhost:1234/v1/models');
  });

  it('does not send Authorization for local provider', async () => {
    mockFetch.mockResolvedValue(
      new Response(JSON.stringify({ data: [{ id: 'test-model', object: 'model' }] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    await listLocalCatalog('http://localhost:8000/v1', 'lm-studio-key');
    const headers = (mockFetch.mock.calls[0][1] as { headers: Record<string, string> }).headers;
    expect(headers.Authorization).toBeUndefined();
  });

  it('throws ModelCatalogEmptyError on empty catalog', async () => {
    mockFetch.mockResolvedValue(
      new Response(JSON.stringify({ data: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    await expect(listLocalCatalog('http://localhost:11434/v1', '')).rejects.toThrow(
      ModelCatalogEmptyError,
    );
  });
});

describe('pickLocalModels', () => {
  it('returns single chat model and embeddings separate', () => {
    const picks: LocalModelPicks = {
      chat: 'llama3.1:8b',
      embeddings: 'nomic-embed-text',
    };
    expect(pickLocalModels(picks)).toEqual({
      chat: 'llama3.1:8b',
      embedding: 'nomic-embed-text',
    });
  });

  it('returns undefined for empty / whitespace chat and embeddings', () => {
    const picks: LocalModelPicks = {
      chat: '   ',
      embeddings: '',
    };
    const out = pickLocalModels(picks);
    expect(out.chat).toBeUndefined();
    expect(out.embedding).toBeUndefined();
  });
});

describe('resolveModels', () => {
  const baseSettings: Settings = {
    provider: 'openrouter',
    openrouterApiKey: 'k',
    groqApiKey: '',
    togetherApiKey: '',
    apiKey: '',
    embeddingApiKey: '',
    webSearchProvider: 'duckduckgo',
    serperApiKey: '',
    temperature: 0,
    maxRetries: 3,
    theme: 'system',
    localServerUrl: LOCAL_DEFAULT_BASE_URL,
    localModels: { chat: '', embeddings: '' },
    localCatalog: [],
    localCatalogFetchedAt: 0,
    pickedModelsOverride: {
      chatModel: '',
      embedding: '',
    },
  };

  it('preserves an explicit chat model and leaves chat unset when that choice is cleared', () => {
    const selectedSettings: Settings = {
      ...baseSettings,
      pickedModelsOverride: { chatModel: 'user/chosen-model', embedding: '' },
    };
    expect(resolveModels(selectedSettings, fakeModels).picked.chat).toBe('user/chosen-model');

    const clearedSettings: Settings = {
      ...selectedSettings,
      pickedModelsOverride: { ...selectedSettings.pickedModelsOverride, chatModel: '' },
    };
    expect(resolveModels(clearedSettings, fakeModels).picked.chat).toBeUndefined();
  });

  it('keeps the embedding catalog default without automatically choosing a chat model', () => {
    const out = resolveModels(baseSettings, fakeModels);
    expect(out.picked.chat).toBeUndefined();
    expect(out.picked.embedding).toBe('nvidia/nv-embedqa-e5-v5');
    expect(out.catalog).toBe(fakeModels);
  });

  it('uses pickLocalModels and the local catalog when provider=local', () => {
    const localCatalog: ModelInfo[] = [
      { id: 'llama3.1:8b', ownedBy: 'ollama', created: 0 },
      { id: 'nomic-embed-text', ownedBy: 'ollama', created: 0 },
    ];
    const out = resolveModels(
      {
        ...baseSettings,
        provider: 'local',
        localModels: {
          chat: 'llama3.1:8b',
          embeddings: 'nomic-embed-text',
        },
        localCatalog,
      },
      fakeModels,
    );
    expect(out.picked.chat).toBe('llama3.1:8b');
    expect(out.picked.embedding).toBe('nomic-embed-text');
    expect(out.catalog).toBe(localCatalog);
    expect(out.warnings).toEqual([]);
  });

  it('warns when local catalog is empty', () => {
    const out = resolveModels(
      { ...baseSettings, provider: 'local', localCatalog: [] },
      fakeModels,
    );
    expect(out.warnings.some(w => w.includes('Local catalog is empty'))).toBe(true);
  });

  it('throws ModelNotFoundError when chat model is not in the catalog', () => {
    expect(() =>
      resolveModels(
        {
          ...baseSettings,
          provider: 'local',
          localModels: {
            chat: 'nonexistent-chat-model',
            embeddings: 'nomic-embed-text',
          },
          localCatalog: [
            { id: 'llama3.1:8b', ownedBy: 'ollama', created: 0 },
            { id: 'nomic-embed-text', ownedBy: 'ollama', created: 0 },
          ],
        },
        fakeModels,
      ),
    ).toThrow(ModelNotFoundError);
  });

  it('throws ModelNotFoundError when embeddings model is not in the catalog, separately from chat', () => {
    expect(() =>
      resolveModels(
        {
          ...baseSettings,
          provider: 'local',
          localModels: {
            chat: 'llama3.1:8b',
            embeddings: 'nonexistent-embed-model',
          },
          localCatalog: [{ id: 'llama3.1:8b', ownedBy: 'ollama', created: 0 }],
        },
        fakeModels,
      ),
    ).toThrow(ModelNotFoundError);
  });
});