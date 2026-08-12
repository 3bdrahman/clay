import { describe, it, expect, vi, beforeEach, afterEach, afterAll } from 'vitest';
import { createEmbeddingsClient } from './embeddings';
import { NIM_BASE_URL } from './providers';
import {
  InvalidApiKeyError,
  RateLimitError,
  ProviderUnreachableError,
  EmbeddingModelMissingError,
} from './errors';

describe('createEmbeddingsClient', () => {
  const mockFetch = vi.fn();
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
    globalThis.fetch = mockFetch;
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
  });

  it('throws EmbeddingModelMissingError when embedding model is empty', async () => {
    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: '',
    });
    await expect(client.embed('test')).rejects.toThrow(EmbeddingModelMissingError);
  });

  it('throws ProviderUnreachableError when base URL is empty', async () => {
    const client = createEmbeddingsClient({
      baseUrl: '',
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed('test')).rejects.toThrow(ProviderUnreachableError);
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
    const norm = Math.sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3);
    expect(result[0]).toEqual([0.1 / norm, 0.2 / norm, 0.3 / norm]);
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
    expect(result[0]).toEqual([1]);
    expect(result[1]).toEqual([1]);
  });

  // --- Error classification tests ---

  it('throws InvalidApiKeyError on 401 response', async () => {
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
    await expect(client.embed('test')).rejects.toThrow(InvalidApiKeyError);
    await expect(client.embed('test')).rejects.toThrow('Invalid API key');
  });

  it('throws InvalidApiKeyError on 403 response', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 403,
      text: async () => 'Forbidden',
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed('test')).rejects.toThrow(InvalidApiKeyError);
  });

  it('throws RateLimitError on 429 response after exhausting retries', async () => {
    vi.useFakeTimers();
    mockFetch.mockImplementation(() =>
      Promise.resolve({
        ok: false,
        status: 429,
        headers: new Map([['retry-after', '60']]),
        text: async () => 'Rate limited',
        json: async () => ({}),
      } as Record<string, unknown>),
    );

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const pending = client.embed('test').catch((e: unknown) => e);
    // Drive through all 3 attempts (Retry-After=60s, capped at MAX_BACKOFF_MS=8000)
    await vi.advanceTimersByTimeAsync(30000);
    const error = await pending;

    expect(error).toBeInstanceOf(RateLimitError);
    expect(String(error)).toMatch(/rate limit/i);
    expect(mockFetch).toHaveBeenCalledTimes(3);
  }, 15000);

  it('throws ProviderUnreachableError on 500 response', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => 'Server error',
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed('test')).rejects.toThrow(ProviderUnreachableError);
  });

  it('throws ProviderUnreachableError on 503 response (retryable)', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 503,
      text: async () => 'Unavailable',
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed('test')).rejects.toThrow(ProviderUnreachableError);
    const error = await client.embed('test').catch(e => e);
    expect(error.retryable).toBe(true);
  });

  it('throws ProviderUnreachableError on network error', async () => {
    mockFetch.mockRejectedValue(new TypeError('Failed to fetch'));

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed('test')).rejects.toThrow(ProviderUnreachableError);
  });

  it('throws ProviderUnreachableError on dimension mismatch in batch', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Map<string, string>(),
      text: async () => '',
      json: async () => ({ data: [{ embedding: [1, 2, 3] }, { embedding: [1, 2] }] }),
    } as Record<string, unknown>);

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed(['a', 'b'])).rejects.toThrow(ProviderUnreachableError);
  });

  // --- T4 hardening: batch split, retry, length cap, dim check, normalize, cache ---

  function okResponse(data: Array<{ embedding: number[] }>): Record<string, unknown> {
    return {
      ok: true,
      status: 200,
      headers: new Map<string, string>(),
      text: async () => '',
      json: async () => ({ data }),
    };
  }

  it('splits 65 inputs into 64+1 batches (two fetch calls)', async () => {
    const inputs = Array.from({ length: 65 }, (_, i) => `chunk ${i}`);
    let call = 0;
    mockFetch.mockImplementation(() => {
      call += 1;
      const size = call === 1 ? 64 : 1;
      return Promise.resolve(
        okResponse(Array.from({ length: size }, () => ({ embedding: [0.1, 0.2] }))),
      );
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const result = await client.embed(inputs);

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result).toHaveLength(65);
  });

  it('splits 200 inputs into 4 batches (64+64+64+8)', async () => {
    const inputs = Array.from({ length: 200 }, (_, i) => `c${i}`);
    let call = 0;
    mockFetch.mockImplementation(() => {
      call += 1;
      const sizes = [64, 64, 64, 8];
      const size = sizes[call - 1] ?? 0;
      return Promise.resolve(
        okResponse(Array.from({ length: size }, () => ({ embedding: [0.1] }))),
      );
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const result = await client.embed(inputs);

    expect(mockFetch).toHaveBeenCalledTimes(4);
    expect(result).toHaveLength(200);
  });

  it('retries on 429 then succeeds', async () => {
    vi.useFakeTimers();
    let attempt = 0;
    mockFetch.mockImplementation(() => {
      attempt += 1;
      if (attempt === 1) {
        return Promise.resolve({
          ok: false,
          status: 429,
          headers: new Map([['retry-after', '1']]),
          text: async () => 'rate limited',
          json: async () => ({}),
        } as Record<string, unknown>);
      }
      return Promise.resolve(okResponse([{ embedding: [0.5, 0.5] }]) as Record<string, unknown>);
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const pending = client.embed('hello');
    await vi.advanceTimersByTimeAsync(2000);
    await vi.runAllTimersAsync();
    const result = await pending;

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result).toEqual([[0.7071067811865475, 0.7071067811865475]]);
  }, 15000);

  it('retries on 503 then succeeds', async () => {
    vi.useFakeTimers();
    let attempt = 0;
    mockFetch.mockImplementation(() => {
      attempt += 1;
      if (attempt === 1) {
        return Promise.resolve({
          ok: false,
          status: 503,
          headers: new Map([['retry-after', '1']]),
          text: async () => 'unavailable',
          json: async () => ({}),
        } as Record<string, unknown>);
      }
      return Promise.resolve(okResponse([{ embedding: [1] }]) as Record<string, unknown>);
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const pending = client.embed('hello');
    await vi.advanceTimersByTimeAsync(2000);
    const result = await pending;

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result).toEqual([[1]]);
  });

  it('exhausts retries after 3 attempts on persistent 429', async () => {
    vi.useFakeTimers();
    mockFetch.mockImplementation(() =>
      Promise.resolve({
        ok: false,
        status: 429,
        headers: new Map<string, string>(),
        text: async () => 'rate limited',
        json: async () => ({}),
      } as Record<string, unknown>),
    );

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const pending = client.embed('hello').catch((e: unknown) => e);
    await vi.advanceTimersByTimeAsync(30000);
    await expect(pending).resolves.toBeInstanceOf(RateLimitError);
    expect(String(await pending)).toMatch(/rate limit|RateLimitError/);
    expect(mockFetch).toHaveBeenCalledTimes(3);
  });

  it('does not retry on non-retryable status (400)', async () => {
    mockFetch.mockImplementation(() =>
      Promise.resolve({
        ok: false,
        status: 400,
        headers: new Map<string, string>(),
        text: async () => 'bad request',
        json: async () => ({}),
      } as Record<string, unknown>),
    );

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed('hello')).rejects.toThrow(ProviderUnreachableError);
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('throws ProviderUnreachableError on input exceeding 8192 token limit', async () => {
    // ~7000 words * 1.3 = ~9100 tokens > 8192
    const huge = Array.from({ length: 7000 }, () => 'word').join(' ');

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    await expect(client.embed(huge)).rejects.toThrow(ProviderUnreachableError);
    await expect(client.embed(huge)).rejects.toThrow(/token limit/);
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('L2 normalizes vectors (norm 5 → unit)', async () => {
    mockFetch.mockResolvedValue(
      okResponse([{ embedding: [3, 4] }]) as Record<string, unknown>,
    );

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const result = await client.embed('hello');

    expect(result).toEqual([[0.6, 0.8]]);
  });

  it('leaves zero-norm vectors unchanged (avoids div/0)', async () => {
    mockFetch.mockResolvedValue(
      okResponse([{ embedding: [0, 0] }]) as Record<string, unknown>,
    );

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const result = await client.embed('hello');

    expect(result).toEqual([[0, 0]]);
  });

  it('asymmetric model defaults input_type to passage', async () => {
    mockFetch.mockResolvedValue(
      okResponse([{ embedding: [0.1] }]) as Record<string, unknown>,
    );

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'nv-embedqa-e5-v5',
    });
    await client.embed('a doc');

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.input_type).toBe('passage');
  });

  it('asymmetric model honors opts.inputType=query', async () => {
    mockFetch.mockResolvedValue(
      okResponse([{ embedding: [0.1] }]) as Record<string, unknown>,
    );

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'nv-embedqa-e5-v5',
    });
    await client.embed('a question', { inputType: 'query' });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.input_type).toBe('query');
  });

  it('symmetric model has no input_type in body', async () => {
    mockFetch.mockResolvedValue(
      okResponse([{ embedding: [0.1] }]) as Record<string, unknown>,
    );

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'nomic-embed-text',
    });
    await client.embed('a doc');

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.input_type).toBeUndefined();
  });

  it('cache hit skips fetch for cached input', async () => {
    const cache = {
      get: vi.fn((_model: string, _hash: string) => [0.9, 0.1] as number[] | undefined),
      set: vi.fn(),
    };
    let fetchCalls = 0;
    mockFetch.mockImplementation(() => {
      fetchCalls += 1;
      return Promise.resolve(
        okResponse([{ embedding: [0.4, 0.3] }]) as Record<string, unknown>,
      );
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const result = await client.embed('cached', { cache });

    expect(fetchCalls).toBe(0);
    expect(result).toEqual([[0.9, 0.1]]);
  });

  it('cache: uncached input fetched, then written to cache', async () => {
    const cache = {
      get: vi.fn(() => undefined),
      set: vi.fn(),
    };
    mockFetch.mockResolvedValue(
      okResponse([{ embedding: [1, 0] }]) as Record<string, unknown>,
    );

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const result = await client.embed('fresh text', { cache });

    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(cache.set).toHaveBeenCalledTimes(1);
    expect(result).toEqual([[1, 0]]);
  });

  it('cache: mixed batch — cached skip, uncached fetched', async () => {
    const cache = {
      // Return a vector only when the hashed text maps to "cached".
      get: vi.fn((_m: string, _h: string): number[] | undefined => [0.9, 0.1]),
      set: vi.fn(),
    };
    let fetchCalls = 0;
    mockFetch.mockImplementation(() => {
      fetchCalls += 1;
      return Promise.resolve(
        okResponse([{ embedding: [0.3, 0.4] }]) as Record<string, unknown>,
      );
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const result = await client.embed(['cached-a', 'cached-b'], { cache });

    expect(fetchCalls).toBe(0);
    expect(result).toEqual([
      [0.9, 0.1],
      [0.9, 0.1],
    ]);
  });

  it('honor Retry-After header when present', async () => {
    vi.useFakeTimers();
    let attempt = 0;
    const retryAfter = new Map<string, string>([['retry-after', '5']]);
    mockFetch.mockImplementation(() => {
      attempt += 1;
      if (attempt === 1) {
        return Promise.resolve({
          ok: false,
          status: 429,
          headers: retryAfter,
          text: async () => 'rate limited',
          json: async () => ({}),
        } as Record<string, unknown>);
      }
      return Promise.resolve(okResponse([{ embedding: [1] }]) as Record<string, unknown>);
    });

    const client = createEmbeddingsClient({
      baseUrl: NIM_BASE_URL,
      apiKey: 'key',
      embeddingModel: 'test-model',
    });
    const pending = client.embed('hello');
    // First advance less than Retry-After — should still be waiting.
    await vi.advanceTimersByTimeAsync(4000);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    // Now pass the Retry-After window.
    await vi.advanceTimersByTimeAsync(2000);
    const result = await pending;

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(result).toEqual([[1]]);
  });
});