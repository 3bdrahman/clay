import { describe, expect, it, vi, beforeEach, afterAll } from 'vitest';
import {
  modelClass,
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

const approvedOpenRouterCatalog: ModelInfo[] = [
  {
    id: 'nvidia/nemotron-3-super-120b-a12b:free',
    ownedBy: 'nvidia',
    created: 0,
    pricing: { prompt: '0', completion: '0', request: '0' },
    supportedParameters: ['tools', 'tool_choice', 'response_format'],
  },
  {
    id: 'qwen/qwen3.8-27b:free',
    ownedBy: 'qwen',
    created: 0,
    pricing: { prompt: '0', completion: '0', request: '0' },
    supportedParameters: ['tools', 'tool_choice', 'structured_outputs'],
  },
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

  it('keeps OpenRouter pricing and supported parameter metadata', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          data: [
            {
              id: 'nvidia/nemotron-3-super-120b-a12b:free',
              owned_by: 'nvidia',
              created: 123,
              pricing: { prompt: '0', completion: '0', request: 0, internal_reasoning: null },
              supported_parameters: ['tools', 'tool_choice', 'response_format'],
            },
          ],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    try {
      const models = await listModels('openrouter', 'test-key');
      expect(models[0]).toEqual({
        id: 'nvidia/nemotron-3-super-120b-a12b:free',
        ownedBy: 'nvidia',
        created: 123,
        pricing: { prompt: '0', completion: '0', request: '0' },
        supportedParameters: ['tools', 'tool_choice', 'response_format'],
      });
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
  it('returns the chat pick; embeddings are local and never picked', () => {
    const picks: LocalModelPicks = {
      chat: 'llama3.1:8b',
    };
    expect(pickLocalModels(picks)).toEqual({
      chat: 'llama3.1:8b',
    });
  });

  it('returns undefined for empty / whitespace chat', () => {
    const picks: LocalModelPicks = {
      chat: '   ',
    };
    expect(pickLocalModels(picks).chat).toBeUndefined();
  });
});

describe('resolveModels', () => {
  const baseSettings: Settings = {
    provider: 'openrouter',
    openrouterApiKey: 'k',
    apiKey: '',
    webSearchProvider: 'duckduckgo',
    serperApiKey: '',
    temperature: 0,
    maxRetries: 3,
    theme: 'system',
    localServerUrl: LOCAL_DEFAULT_BASE_URL,
    localModels: { chat: '' },
    localCatalog: [],
    localCatalogFetchedAt: 0,
    pickedModelsOverride: {
      chatModel: '',
    },
  };

  it('preserves an approved explicit cloud model and auto-picks the first approved model when cleared', () => {
    const selectedSettings: Settings = {
      ...baseSettings,
      pickedModelsOverride: { chatModel: 'qwen/qwen3.8-27b:free' },
    };
    expect(resolveModels(selectedSettings, approvedOpenRouterCatalog).picked.chat).toBe(
      'qwen/qwen3.8-27b:free',
    );

    const clearedSettings: Settings = {
      ...selectedSettings,
      pickedModelsOverride: { chatModel: '' },
    };
    expect(resolveModels(clearedSettings, approvedOpenRouterCatalog).picked.chat).toBe(
      'nvidia/nemotron-3-super-120b-a12b:free',
    );
  });

  it('uses the second approved cloud model when the first is unavailable and selection is blank', () => {
    const out = resolveModels(baseSettings, [approvedOpenRouterCatalog[1]]);
    expect(out.picked.chat).toBe('qwen/qwen3.8-27b:free');
    expect(out.catalog.map((model) => model.id)).toEqual(['qwen/qwen3.8-27b:free']);
    expect(out.warnings).toEqual([]);
  });

  it('does not silently override an explicit invalid or malicious cloud model', () => {
    const out = resolveModels(
      {
        ...baseSettings,
        pickedModelsOverride: { chatModel: 'attacker/paid-model' },
      },
      approvedOpenRouterCatalog,
    );
    expect(out.picked.chat).toBeUndefined();
    expect(out.catalog.map((model) => model.id)).toEqual([
      'nvidia/nemotron-3-super-120b-a12b:free',
      'qwen/qwen3.8-27b:free',
    ]);
    expect(out.warnings[0]).toContain('attacker/paid-model');
  });

  it('rejects approved OpenRouter ids when catalog metadata shows paid or missing eligibility', () => {
    const out = resolveModels(baseSettings, [
      { ...approvedOpenRouterCatalog[0], pricing: { prompt: '0', completion: '0.01', request: '0' } },
      { ...approvedOpenRouterCatalog[1], supportedParameters: ['tools', 'tool_choice'] },
    ]);
    expect(out.picked.chat).toBeUndefined();
    expect(out.catalog).toEqual([]);
    expect(out.warnings[0]).toContain('No approved openrouter models');
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
        },
        localCatalog,
      },
      fakeModels,
    );
    expect(out.picked.chat).toBe('llama3.1:8b');
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
});
