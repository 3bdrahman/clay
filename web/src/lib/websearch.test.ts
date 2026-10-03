import { describe, expect, it, vi, afterEach } from 'vitest';
import { createWebSearchClient, getWebSearchAvailability } from './websearch';
import { useAppStore } from '../store';
import type { Settings } from './types';

const baseSettings = (overrides: Partial<Settings> = {}): Settings => ({
  ...useAppStore.getState().settings,
  webSearchProvider: 'mwmbl',
  ...overrides,
});

afterEach(() => vi.restoreAllMocks());

describe('explicit search provider selection', () => {
  it('makes no request when search is disabled', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');
    const client = createWebSearchClient(baseSettings({ webSearchProvider: 'none' }));
    expect(await client.search('anything')).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('calls Serper with only its own key when explicitly selected', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      organic: [{ title: 'Hi', snippet: 'There', link: 'https://example.com/result' }],
    })));
    const client = createWebSearchClient(baseSettings({
      webSearchProvider: 'serper', serperApiKey: ' KEY ', openrouterApiKey: 'model-key',
    }));
    const results = await client.search('hello', 3);
    expect(fetchMock).toHaveBeenCalledWith('https://google.serper.dev/search', expect.objectContaining({
      method: 'POST',
      headers: { 'X-API-KEY': 'KEY', 'Content-Type': 'application/json' },
      body: JSON.stringify({ q: 'hello', num: 3 }),
      credentials: 'omit',
      redirect: 'error',
    }));
    expect(JSON.stringify(fetchMock.mock.calls)).not.toContain('model-key');
    expect(results).toEqual([{ type: 'web_search', title: 'Hi', content: 'There', url: 'https://example.com/result' }]);
  });

  it('requires a Serper key before making any request', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');
    const client = createWebSearchClient(baseSettings({ webSearchProvider: 'serper', serperApiKey: '' }));
    await expect(client.search('private query')).rejects.toMatchObject({
      message: expect.stringContaining('Serper API key'), retryable: false,
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([401, 403, 429, 503])('preserves Serper HTTP %s without sending the query elsewhere', async status => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('Failed', { status }));
    const client = createWebSearchClient(baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' }));
    await expect(client.search('private query')).rejects.toMatchObject({
      provider: 'serper', message: expect.stringContaining(String(status)), retryable: status >= 500 || status === 429,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith('https://google.serper.dev/search', expect.anything());
  });

  it('returns no citations for a genuine empty Serper result list', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ organic: [] })));
    const client = createWebSearchClient(baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' }));
    await expect(client.search('nothing indexed')).resolves.toEqual([]);
  });

  it('keeps a valid Serper source when its optional snippet is omitted', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      organic: [{ title: 'Source title', link: 'https://example.com/' }],
    })));
    const client = createWebSearchClient(baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' }));
    await expect(client.search('query')).resolves.toEqual([
      { type: 'web_search', title: 'Source title', content: '', url: 'https://example.com/' },
    ]);
  });

  it('treats a malformed Serper payload as a provider error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ message: 'not results' })));
    const client = createWebSearchClient(baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' }));
    await expect(client.search('query')).rejects.toMatchObject({ provider: 'serper', retryable: false });
  });
});

describe('web search availability', () => {
  it('agrees with the selected provider and credentials', () => {
    expect(getWebSearchAvailability(baseSettings({ webSearchProvider: 'none' })).available).toBe(false);
    expect(getWebSearchAvailability(baseSettings({ webSearchProvider: 'serper', serperApiKey: ' ' })).available).toBe(false);
    expect(getWebSearchAvailability(baseSettings({ webSearchProvider: 'serper', serperApiKey: 'KEY' })).available).toBe(true);
    expect(getWebSearchAvailability(baseSettings({ webSearchProvider: 'mwmbl', serperApiKey: '' })).available).toBe(true);
  });
});
