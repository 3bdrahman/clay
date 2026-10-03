import { afterEach, describe, expect, it, vi } from 'vitest';
import { createWebSearchClient, getWebSearchAvailability } from './websearch';
import { useAppStore } from '../store';
import { WebSearchProviderError } from './errors';

const settings = () => ({
  ...useAppStore.getState().settings,
  webSearchProvider: 'mwmbl' as const,
  openrouterApiKey: 'private-model-key',
  serperApiKey: 'private-search-key',
});

const result = {
  url: 'https://en.wikipedia.org/wiki/Retrieval-augmented_generation',
  title: 'Retrieval-augmented generation',
  content: 'RAG retrieves information for language models.',
  engine: 'wikipedia',
  score: 1,
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
  vi.unstubAllEnvs();
});

describe('keyless Mwmbl search', () => {
  it('is available in production without a proxy or search API key', () => {
    vi.stubEnv('DEV', false);
    expect(getWebSearchAvailability({ webSearchProvider: 'mwmbl', serperApiKey: '' })).toMatchObject({ available: true });
  });

  it('uses the supported JSON API and never forwards model or Serper credentials', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ results: [result] })));
    const results = await createWebSearchClient(settings()).search('RAG & privacy?', 3);
    expect(results).toEqual([{ type: 'web_search', url: result.url, title: result.title, content: result.content }]);
    const [url, init] = fetchMock.mock.calls[0];
    const requestUrl = new URL(String(url));
    expect(requestUrl.origin + requestUrl.pathname).toBe('https://mwmbl.org/api/v2/search/');
    expect(requestUrl.searchParams.get('q')).toBe('RAG & privacy?');
    expect(init?.method).toBe('GET');
    expect(init?.credentials).toBe('omit');
    const headers = new Headers(init?.headers);
    expect(headers.get('Authorization')).toBeNull();
    expect(headers.get('X-API-KEY')).toBeNull();
    expect(JSON.stringify(fetchMock.mock.calls)).not.toContain('private-');
  });

  it('keeps source text literal and filters unsafe and duplicate URLs before limiting results', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ results: [
      { ...result, url: 'javascript:alert(1)' },
      { ...result, url: 'https://user:password@example.com/private' },
      result,
      result,
      { ...result, url: 'https://example.com/other', title: '<b>Literal title</b>' },
      { ...result, url: 'https://example.com/extra' },
    ] })));
    const results = await createWebSearchClient(settings()).search('RAG', 2);
    expect(results.map(item => item.url)).toEqual([result.url, 'https://example.com/other']);
    expect(results[1].title).toBe('<b>Literal title</b>');
  });

  it('returns no invented citations for a genuine empty result list', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ results: [] })));
    await expect(createWebSearchClient(settings()).search('no matching documents')).resolves.toEqual([]);
  });

  it.each([
    '<html>Bot challenge</html>',
    JSON.stringify({ message: 'service failed' }),
    JSON.stringify({ results: [{ url: 'https://example.com', title: 42, content: 'bad' }] }),
  ])('reports malformed provider responses instead of presenting empty successful search', async body => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(body));
    await expect(createWebSearchClient(settings()).search('query')).rejects.toMatchObject({
      provider: 'mwmbl', code: 'WEB_SEARCH_PROVIDER_FAILED', retryable: false,
    });
  });

  it('surfaces provider failure without sending a query to Serper or another service', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('unavailable', { status: 503 }));
    await expect(createWebSearchClient(settings()).search('private query')).rejects.toMatchObject({
      provider: 'mwmbl', retryable: true, message: expect.stringContaining('503'),
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('reports a rate limit without immediately retrying a public keyless service', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('slow down', { status: 429 }));
    await expect(createWebSearchClient(settings()).search('query')).rejects.toMatchObject({
      provider: 'mwmbl', retryable: false, message: expect.stringContaining('429'),
    });
  });

  it('does not issue requests for empty queries or zero requested results', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');
    const client = createWebSearchClient(settings());
    await expect(client.search('   ')).resolves.toEqual([]);
    await expect(client.search('query', 0)).resolves.toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('aborts an in-flight fetch when the caller cancels', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation((_url, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    }));
    const controller = new AbortController();
    const pending = createWebSearchClient(settings()).search('query', 3, controller.signal);
    const rejected = expect(pending).rejects.toMatchObject({ provider: 'mwmbl', retryable: false });
    controller.abort();
    await rejected;
  });

  it('does not fetch when the caller already cancelled', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');
    const controller = new AbortController();
    controller.abort();
    await expect(createWebSearchClient(settings()).search('query', 3, controller.signal)).rejects.toBeInstanceOf(WebSearchProviderError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('times out a stalled response body as well as a stalled connection', async () => {
    vi.useFakeTimers();
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => ({
      ok: true,
      json: () => new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
      }),
    }) as Response);
    const pending = createWebSearchClient(settings()).search('query');
    const rejected = expect(pending).rejects.toMatchObject({ provider: 'mwmbl', message: expect.stringContaining('timed out') });
    await vi.advanceTimersByTimeAsync(20_000);
    await rejected;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('times out a connection that never returns response headers', async () => {
    vi.useFakeTimers();
    vi.spyOn(globalThis, 'fetch').mockImplementation((_url, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    }));
    const pending = createWebSearchClient(settings()).search('query');
    const rejected = expect(pending).rejects.toMatchObject({ provider: 'mwmbl', message: expect.stringContaining('timed out') });
    await vi.advanceTimersByTimeAsync(20_000);
    await rejected;
    expect(vi.getTimerCount()).toBe(0);
  });

  it('reports a network error with the correct provider and releases its timeout', async () => {
    vi.useFakeTimers();
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));
    await expect(createWebSearchClient(settings()).search('query')).rejects.toMatchObject({ provider: 'mwmbl', retryable: true });
    expect(vi.getTimerCount()).toBe(0);
  });
});
