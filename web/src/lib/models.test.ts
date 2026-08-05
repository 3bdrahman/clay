import { describe, expect, it, vi, beforeEach, afterAll } from 'vitest';
import {
  modelClass,
  pickBestModels,
  pickLocalModels,
  listNimModels,
  listLocalCatalog,
  resolveModels,
  ModelsFetchError,
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

describe('pickBestModels', () => {
  it('returns one model for every role', () => {
    const picked = pickBestModels(fakeModels);
    expect(picked.routing).toBeTypeOf('string');
    expect(picked.codeGen).toBeTypeOf('string');
    expect(picked.answer).toBeTypeOf('string');
    expect(picked.eval).toBeTypeOf('string');
    expect(picked.embedding).toBeTypeOf('string');
  });

  it('picks the highest-scoring embedding model', () => {
    const picked = pickBestModels(fakeModels);
    expect(picked.embedding).toBe('nvidia/nv-embedqa-e5-v5');
  });

  it('picks Codestral 22B for code generation', () => {
    const picked = pickBestModels(fakeModels);
    expect(picked.codeGen).toBe('mistralai/codestral-22b-instruct-v0.1');
  });

  it('picks Nemotron-3 Ultra for answer', () => {
    const picked = pickBestModels(fakeModels);
    expect(picked.answer).toBe('nvidia/nemotron-3-ultra-550b-a55b');
  });

  it('picks a small chat model for routing', () => {
    const picked = pickBestModels(fakeModels);
    expect(['meta/llama-3.1-8b-instruct', 'mistralai/mistral-7b-instruct-v0.3'])
      .toContain(picked.routing);
  });

  it('picks a different small chat model for eval', () => {
    const picked = pickBestModels(fakeModels);
    expect(picked.eval).toBeDefined();
    expect(picked.eval).not.toBe(picked.routing);
  });

  it('excludes vision models from chat picks', () => {
    const picked = pickBestModels(fakeModels);
    expect(picked.routing).not.toContain('vision');
    expect(picked.answer).not.toContain('vision');
    expect(picked.eval).not.toContain('vision');
  });

  it('excludes safety/guard models from chat picks', () => {
    const picked = pickBestModels(fakeModels);
    expect(picked.routing).not.toContain('guard');
    expect(picked.answer).not.toContain('guard');
  });

  it('handles empty model list gracefully', () => {
    const picked = pickBestModels([]);
    expect(picked.routing).toBeUndefined();
    expect(picked.codeGen).toBeUndefined();
    expect(picked.answer).toBeUndefined();
    expect(picked.eval).toBeUndefined();
    expect(picked.embedding).toBeUndefined();
  });

  it('falls back to first chat model when no small models exist', () => {
    const noSmall: ModelInfo[] = [
      { id: 'mistralai/codestral-22b-instruct-v0.1', ownedBy: 'mistralai', created: 0 },
      { id: 'nvidia/nemotron-3-ultra-550b-a55b', ownedBy: 'nvidia', created: 0 },
      { id: 'nvidia/nv-embedqa-e5-v5', ownedBy: 'nvidia', created: 0 },
    ];
    const picked = pickBestModels(noSmall);
    expect(picked.codeGen).toBe('mistralai/codestral-22b-instruct-v0.1');
    expect(picked.answer).toBe('nvidia/nemotron-3-ultra-550b-a55b');
    expect(picked.embedding).toBe('nvidia/nv-embedqa-e5-v5');
  });
});

describe('listNimModels', () => {
  it('throws ModelsFetchError when API key is missing', async () => {
    await expect(listNimModels('')).rejects.toThrow(ModelsFetchError);
    await expect(listNimModels('')).rejects.toThrow('API key required');
  });

  it('throws ModelsFetchError on non-OK response', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (() =>
      Promise.resolve(new Response('Unauthorized', { status: 401 }))) as typeof fetch;
    try {
      await expect(listNimModels('test-key')).rejects.toThrow(ModelsFetchError);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('parses the catalog response', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (() =>
      Promise.resolve(
        new Response(
          JSON.stringify({
            data: [
              { id: 'meta/llama-3.1-8b-instruct', owned_by: 'meta', created: 123 },
              { id: 'nvidia/nv-embedqa-e5-v5', owned_by: 'nvidia', created: 456 },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      )) as typeof fetch;
    try {
      const models = await listNimModels('test-key');
      expect(models).toHaveLength(2);
      expect(models[0]).toEqual({ id: 'meta/llama-3.1-8b-instruct', ownedBy: 'meta', created: 123 });
      expect(models[1]).toEqual({ id: 'nvidia/nv-embedqa-e5-v5', ownedBy: 'nvidia', created: 456 });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('handles missing optional fields', async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (() =>
      Promise.resolve(
        new Response(
          JSON.stringify({ data: [{ id: 'test/model' }] }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      )) as typeof fetch;
    try {
      const models = await listNimModels('test-key');
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

  it('throws when baseUrl is empty', async () => {
    await expect(listLocalCatalog('', '')).rejects.toThrow(ModelsFetchError);
  });

  it('throws ModelsFetchError on non-OK response', async () => {
    mockFetch.mockResolvedValue(new Response('boom', { status: 502 }));
    await expect(listLocalCatalog('http://localhost:11434/v1', '')).rejects.toThrow(
      ModelsFetchError,
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
        { status: 200, headers: { 'Content-Type': 'application/json' } },
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
      new Response(JSON.stringify({ data: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    await listLocalCatalog('http://localhost:1234/v1///', '');
    expect(mockFetch.mock.calls[0][0]).toBe('http://localhost:1234/v1/models');
  });

  it('sends Authorization when apiKey is provided', async () => {
    mockFetch.mockResolvedValue(
      new Response(JSON.stringify({ data: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    await listLocalCatalog('http://localhost:8000/v1', 'lm-studio-key');
    const headers = (mockFetch.mock.calls[0][1] as { headers: Record<string, string> }).headers;
    expect(headers.Authorization).toBe('Bearer lm-studio-key');
  });
});

describe('pickLocalModels', () => {
  it('maps Settings.localModels into PickedModels', () => {
    const picks: LocalModelPicks = {
      routing: 'llama3.1:8b',
      codeGen: 'qwen2.5-coder:7b',
      answer: 'llama3.1:70b',
      eval: 'mistral:7b',
      embedding: 'nomic-embed-text',
    };
    expect(pickLocalModels(picks)).toEqual({
      routing: 'llama3.1:8b',
      codeGen: 'qwen2.5-coder:7b',
      answer: 'llama3.1:70b',
      eval: 'mistral:7b',
      embedding: 'nomic-embed-text',
    });
  });

  it('returns undefined for empty / whitespace entries', () => {
    const picks: LocalModelPicks = {
      routing: '   ',
      codeGen: '',
      answer: 'llama3.1:8b',
      eval: '',
      embedding: '  ',
    };
    const out = pickLocalModels(picks);
    expect(out.routing).toBeUndefined();
    expect(out.codeGen).toBeUndefined();
    expect(out.answer).toBe('llama3.1:8b');
    expect(out.eval).toBeUndefined();
    expect(out.embedding).toBeUndefined();
  });
});

describe('resolveModels', () => {
  const baseSettings: Settings = {
    provider: 'nim',
    apiKey: 'k',
    embeddingApiKey: '',
    webSearchProvider: 'duckduckgo',
    serperApiKey: '',
    temperature: 0,
    maxRetries: 3,
    theme: 'system',
    localServerUrl: LOCAL_DEFAULT_BASE_URL,
    localModels: { routing: '', codeGen: '', answer: '', eval: '', embedding: '' },
    localCatalog: [],
    localCatalogFetchedAt: 0,
  };

  it('uses pickBestModels for the NIM provider', () => {
    const out = resolveModels(baseSettings, fakeModels);
    expect(out.picked.answer).toBe('nvidia/nemotron-3-ultra-550b-a55b');
    expect(out.catalog).toBe(fakeModels);
  });

  it('uses pickLocalModels and the local catalog when provider=local', () => {
    const localCatalog: ModelInfo[] = [
      { id: 'llama3.1:8b', ownedBy: 'ollama', created: 0 },
    ];
    const out = resolveModels(
      {
        ...baseSettings,
        provider: 'local',
        localModels: {
          routing: 'llama3.1:8b',
          codeGen: '',
          answer: 'llama3.1:8b',
          eval: 'llama3.1:8b',
          embedding: '',
        },
        localCatalog,
      },
      fakeModels,
    );
    expect(out.picked.routing).toBe('llama3.1:8b');
    expect(out.picked.embedding).toBeUndefined();
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

  it('warns when a picked model is not in the catalog', () => {
    const out = resolveModels(
      {
        ...baseSettings,
        provider: 'local',
        localModels: {
          routing: 'llama3.1:8b',
          codeGen: 'nonexistent-model',
          answer: 'llama3.1:8b',
          eval: '',
          embedding: '',
        },
        localCatalog: [{ id: 'llama3.1:8b', ownedBy: 'ollama', created: 0 }],
      },
      fakeModels,
    );
    expect(out.warnings.some(w => w.includes('codeGen') && w.includes('nonexistent-model'))).toBe(true);
  });
});
