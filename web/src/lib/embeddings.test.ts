import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest';
import { createEmbeddingsClient, EmbeddingsConfigError } from './embeddings';
import { NIM_BASE_URL } from './providers';

describe('createEmbeddingsClient', () => {
  const mockFetch = vi.fn();
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
    globalThis.fetch = mockFetch;
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
  });

  it('throws EmbeddingsConfigError when embedding model is empty', async () => {
    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: '',
    });
    await expect(client.embed('test')).rejects.toThrow(EmbeddingsConfigError);
  });

  it('throws EmbeddingsConfigError when base URL is empty', async () => {
    const client = createEmbeddingsClient({
      baseUrl: '',
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed('test')).rejects.toThrow(EmbeddingsConfigError);
  });

  it('uses the supplied api key', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1, 0.2] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'my-key',
      embeddingModel: 'test-model',
    });
    await client.embed('test');

    expect(mockFetch).toHaveBeenCalledWith(
      `${NIM_BASE_URL}/embeddings`,
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: 'Bearer my-key' }),
      }),
    );
  });

  it('omits Authorization header when apiKey is empty (local server, no auth)', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: 'http://localhost:11434/v1',
      apiKey: '',
      embeddingModel: 'nomic-embed-text',
    });
    await client.embed('test');

    const callHeaders = (mockFetch.mock.calls[0][1] as { headers: Record<string, string> }).headers;
    expect(callHeaders.Authorization).toBeUndefined();
  });

  it('targets the configured baseUrl (not NIM)', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: 'http://localhost:11434/v1',
      apiKey: '',
      embeddingModel: 'nomic-embed-text',
    });
    await client.embed('test');

    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:11434/v1/embeddings',
      expect.objectContaining({ method: 'POST' }),
    );
  });

  it('strips trailing slashes from baseUrl', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: 'http://localhost:11434/v1///',
      apiKey: '',
      embeddingModel: 'nomic-embed-text',
    });
    await client.embed('test');

    expect(mockFetch.mock.calls[0][0]).toBe('http://localhost:11434/v1/embeddings');
  });

  it('embeds single string', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1, 0.2, 0.3] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const result = await client.embed('hello');

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual([0.1, 0.2, 0.3]);
  });

  it('embeds array of strings', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1] }, { embedding: [0.2] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const result = await client.embed(['a', 'b']);

    expect(result).toHaveLength(2);
    expect(result[0]).toEqual([0.1]);
    expect(result[1]).toEqual([0.2]);
  });

  it('throws on API error', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 401,
      text: async () => 'Unauthorized',
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed('test')).rejects.toThrow('Embedding API error 401');
  });

  it('sends correct model name', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'nvidia/nv-embedqa-e5-v5',
    });
    await client.embed('test');

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.model).toBe('nvidia/nv-embedqa-e5-v5');
  });

  // Regression: nv-embedqa-e5-v5 is asymmetric — NIM API requires input_type.
  // Without it, the API returns HTTP 400 "'input_type' parameter is required for asymmetric models".
  it('sends input_type=query for asymmetric models on query calls', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'nvidia/nv-embedqa-e5-v5',
    });
    await client.embed('a question', { inputType: 'query' });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.input_type).toBe('query');
  });

  it('sends input_type=passage for asymmetric models on document calls', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'nvidia/nv-embedqa-e5-v5',
    });
    await client.embed(['chunk a', 'chunk b'], { inputType: 'passage' });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.input_type).toBe('passage');
  });

  it('omits input_type for symmetric models', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1] }] }),
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'nomic-embed-text',
    });
    await client.embed('test', { inputType: 'query' });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.input_type).toBeUndefined();
  });
});
