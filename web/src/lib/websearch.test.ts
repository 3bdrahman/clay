import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { createWebSearchClient } from './websearch';
import {
  decodeHtmlEntities,
  extractRealUrl,
  parseDuckDuckGoHtml,
} from './websearch';
import type { Settings } from './types';
import { WebSearchProviderError } from './errors';

const baseSettings = (overrides: Partial<Settings> = {}): Settings =>
  ({
    provider: 'nim',
    apiKey: '',
    embeddingApiKey: '',
    webSearchProvider: 'duckduckgo',
    serperApiKey: '',
    temperature: 0,
    maxRetries: 3,
    vectorstoreInitialK: 8,
    theme: 'system',
    localServerUrl: '',
    localModels: { chat: '', embeddings: '' },
    localCatalog: [],
    localCatalogFetchedAt: 0,
    ...overrides,
  }) as Settings;

describe('decodeHtmlEntities', () => {
  it('decodes the five basic entities', () => {
    expect(decodeHtmlEntities('a & b < c > d " e &apos; f')).toBe(
      "a & b < c > d \" e ' f",
    );
  });

  it('does not double-decode ampersands inside entities', () => {
    expect(decodeHtmlEntities('Tom & Jerry & Friends')).toBe('Tom & Jerry & Friends');
  });

  it('passes through plain text untouched', () => {
    expect(decodeHtmlEntities('hello world')).toBe('hello world');
  });

  it('handles empty string', () => {
    expect(decodeHtmlEntities('')).toBe('');
  });

  it('decodes entities produced by previous decoding (no double-decode)', () => {
    expect(decodeHtmlEntities('<script>alert("x")</script>')).toBe(
      '<script>alert("x")</script>',
    );
  });
});

describe('extractRealUrl', () => {
  it('unwraps DuckDuckGo redirect urls to the real destination', () => {
    const uddg = encodeURIComponent('https://example.com/path?q=1');
    expect(extractRealUrl(`https://duckduckgo.com/l/?uddg=${uddg}`)).toBe(
      'https://example.com/path?q=1',
    );
  });

  it('returns the original url when there is no uddg param', () => {
    expect(extractRealUrl('https://example.com/page')).toBe('https://example.com/page');
  });

  it('returns the original url when it is not a valid url', () => {
    expect(extractRealUrl('not a url')).toBe('not a url');
  });
});

describe('parseDuckDuckGoHtml', () => {
  const lt = '<';
  const gt = '>';
  const slash = '/';
  const closeA = lt + slash + 'a' + gt;

  it('parses a result block with an html-encoded title and snippet', () => {
    const html =
      '<a class="result__a" href="https://example.com/">Foo & Bar' + closeA +
      '<a class="result__snippet">Hello <world> "hi"' + closeA;
    const [r] = parseDuckDuckGoHtml(html, 5);
    expect(r.title).toBe('Foo & Bar');
    expect(r.content).toBe('Hello  "hi"');
    expect(r.url).toBe('https://example.com/');
    expect(r.type).toBe('web_search');
  });

  it('strips inner tags from snippets', () => {
    const html =
      '<a class="result__a" href="https://example.com/">Title' + closeA +
      '<a class="result__snippet"><b>bold</b> and <i>italic</i>' + closeA;
    const [r] = parseDuckDuckGoHtml(html, 5);
    expect(r.content).toBe('bold and italic');
  });

  it('returns at most k results', () => {
    const blocks = Array.from({ length: 7 }, (_, i) =>
      '<a class="result__a" href="https://example.com/' + i + slash + '">Title ' + i + closeA +
      '<a class="result__snippet">Snippet ' + i + closeA,
    ).join('');
    expect(parseDuckDuckGoHtml(blocks, 3)).toHaveLength(3);
  });

  it('returns a stub result when no results are found', () => {
    const r = parseDuckDuckGoHtml(
      '<html><body>no results here</body' + closeA + 'html' + closeA,
      5,
    );
    expect(r).toHaveLength(1);
    expect(r[0].title).toMatch(/no web search results found/i);
    expect(r[0].url).toBe('https://serper.dev');
  });
});

describe('createWebSearchClient.search', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('returns empty array when provider is "none"', async () => {
    const client = createWebSearchClient(baseSettings({ webSearchProvider: 'none' }));
    expect(await client.search('anything')).toEqual([]);
  });

  it('calls serper with the API key when configured', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ organic: [{ title: 'Hi', snippet: 'There', link: 'https://x' }] }),
    });
    globalThis.fetch = fetchMock;

    const client = createWebSearchClient(
      baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' }),
    );
    const results = await client.search('hello', 3);

    expect(fetchMock).toHaveBeenCalledWith(
      'https://google.serper.dev/search',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'X-API-KEY': 'KEY' }),
      }),
    );
    expect(results[0]).toMatchObject({ title: 'Hi', content: 'There', url: 'https://x' });
  });

  it('throws WebSearchProviderError on serper 401', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      text: async () => 'Unauthorized',
    });

    const client = createWebSearchClient(
      baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' }),
    );
    await expect(client.search('hello')).rejects.toThrow(WebSearchProviderError);
  });

  it('throws WebSearchProviderError on serper 429', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 429,
      headers: new Map([['retry-after', '60']]),
      text: async () => 'Rate limited',
    });

    const client = createWebSearchClient(
      baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' }),
    );
    await expect(client.search('hello')).rejects.toThrow(WebSearchProviderError);
    const error = await client.search('hello').catch(e => e);
    expect(error.retryable).toBe(true);
  });

  it('falls back to DuckDuckGo when serper throws', async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 500, headers: new Map(), json: async () => ({}) }) // serper
      .mockResolvedValueOnce({
        ok: true,
        text: async () => `
          <a class="result__a" href="https://example.com/">DDG Title</a>
          <a class="result__snippet">DDG snippet</a>
        `,
      });

    const client = createWebSearchClient(
      baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' }),
    );
    const results = await client.search('hello');
    expect(results[0].title).toBe('DDG Title');
  });

  it('throws WebSearchProviderError on DuckDuckGo 503', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false, status: 503, text: async () => '' });

    const client = createWebSearchClient(baseSettings());
    await expect(client.search('hello')).rejects.toThrow(WebSearchProviderError);
  });

  it('throws WebSearchProviderError on DuckDuckGo network error', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('network'));

    const client = createWebSearchClient(baseSettings());
    await expect(client.search('hello')).rejects.toThrow(WebSearchProviderError);
  });

  it('throws WebSearchProviderError when both providers fail', async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 500, headers: new Map(), json: async () => ({}) }) // serper
      .mockResolvedValueOnce({ ok: false, status: 503, headers: new Map(), text: async () => '' }); // ddg

    const client = createWebSearchClient(
      baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' }),
    );
    await expect(client.search('hello')).rejects.toThrow(WebSearchProviderError);
    const error = await client.search('hello').catch(e => e);
    expect(error.message).toContain('Both Serper and DuckDuckGo failed');
  });
});