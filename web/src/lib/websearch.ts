// Web search client: keyless Mwmbl JSON API or explicitly configured Serper.
import type { Settings, WebResult } from './types';
import { WebSearchProviderError } from './errors';

type SearchProvider = Exclude<Settings['webSearchProvider'], 'none'>;
const MWMBL_SEARCH_URL = 'https://mwmbl.org/api/v2/search/';
const SERPER_SEARCH_URL = 'https://google.serper.dev/search';
const DEFAULT_WEB_SEARCH_K = 5;
const WEB_SEARCH_TIMEOUT_MS = 15_000;

export interface WebSearchClient {
  search(query: string, k?: number, signal?: AbortSignal): Promise<WebResult[]>;
}

export function getWebSearchAvailability(
  settings: Pick<Settings, 'webSearchProvider' | 'serperApiKey'>,
): { available: boolean; message: string } {
  if (settings.webSearchProvider === 'none') {
    return { available: false, message: 'Web search is disabled. Questions can use your loaded files.' };
  }
  if (settings.webSearchProvider === 'serper') {
    return settings.serperApiKey.trim()
      ? { available: true, message: 'Serper API key configured. Search queries are sent to Serper.' }
      : { available: false, message: 'Add a Serper API key in Settings to enable web search.' };
  }
  return {
    available: true,
    message: 'Keyless search uses Mwmbl’s public index. Queries are sent directly to Mwmbl; coverage and freshness vary.',
  };
}

function invalidResponse(provider: SearchProvider): WebSearchProviderError {
  return new WebSearchProviderError(provider, 'The search service returned an invalid result format', undefined, { retryable: false });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseResults(provider: SearchProvider, data: unknown, k: number): WebResult[] {
  if (!isRecord(data)) throw invalidResponse(provider);
  const items = provider === 'mwmbl' ? data.results : data.organic;
  if (!Array.isArray(items)) throw invalidResponse(provider);

  const results: WebResult[] = [];
  const urls = new Set<string>();
  for (const item of items) {
    if (!isRecord(item)) throw invalidResponse(provider);
    const title = item.title;
    const content = provider === 'mwmbl' ? item.content : (item.snippet ?? '');
    const rawUrl = provider === 'mwmbl' ? item.url : item.link;
    if (typeof title !== 'string' || !title.trim() || typeof content !== 'string' || typeof rawUrl !== 'string') {
      throw invalidResponse(provider);
    }
    if (!URL.canParse(rawUrl)) continue;
    const url = new URL(rawUrl);
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password || urls.has(url.href)) continue;
    urls.add(url.href);
    results.push({ type: 'web_search', title: title.trim(), content: content.trim(), url: url.href });
    if (results.length >= k) break;
  }
  if (items.length > 0 && results.length === 0) {
    throw new WebSearchProviderError(provider, 'The search service returned no usable source URLs', undefined, { retryable: false });
  }
  return results;
}

async function requestJson(
  provider: SearchProvider,
  url: string,
  init: RequestInit,
  signal?: AbortSignal,
): Promise<unknown> {
  const cancelled = () => new WebSearchProviderError(provider, 'Search was cancelled', undefined, { retryable: false });
  if (signal?.aborted) throw cancelled();

  const controller = new AbortController();
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort, { once: true });
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, WEB_SEARCH_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...init,
      credentials: 'omit',
      redirect: 'error',
      signal: controller.signal,
    });
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        const reason = provider === 'serper' ? 'Invalid API key' : 'The public search service refused the request';
        throw new WebSearchProviderError(provider, `${reason} (${response.status})`, undefined, { retryable: false });
      }
      if (response.status === 429) {
        // A public keyless service should not be hit by automatic rate-limit retries.
        throw new WebSearchProviderError(provider, 'Rate limited (429); try again later', undefined, { retryable: provider === 'serper' });
      }
      throw new WebSearchProviderError(provider, `HTTP ${response.status}: ${response.statusText}`, undefined, { retryable: response.status >= 500 });
    }
    // Keep the deadline active while reading the response, not just its headers.
    return await response.json();
  } catch (error) {
    if (signal?.aborted) throw cancelled();
    if (timedOut) {
      throw new WebSearchProviderError(provider, 'The search request timed out; try again', undefined, { retryable: true });
    }
    if (error instanceof WebSearchProviderError) throw error;
    if (isRecord(error) && error.name === 'SyntaxError') throw invalidResponse(provider);
    const cause = error instanceof Error ? error : new Error(String(error));
    throw new WebSearchProviderError(provider, 'Could not reach the search service', cause, { retryable: true });
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener('abort', onAbort);
  }
}

export function createWebSearchClient(settings: Settings): WebSearchClient {
  return {
    async search(query, k = DEFAULT_WEB_SEARCH_K, signal) {
      const provider = settings.webSearchProvider;
      const trimmedQuery = query.trim();
      if (provider === 'none' || !trimmedQuery || !Number.isFinite(k) || k < 1) return [];
      const availability = getWebSearchAvailability(settings);
      if (!availability.available) {
        throw new WebSearchProviderError(provider, availability.message, undefined, { retryable: false });
      }
      const limit = Math.floor(k);
      // Only the explicitly selected service receives this query. A failure
      // never triggers another provider or forwards unrelated credentials.
      if (provider === 'serper') {
        const data = await requestJson(provider, SERPER_SEARCH_URL, {
          method: 'POST',
          headers: { 'X-API-KEY': settings.serperApiKey.trim(), 'Content-Type': 'application/json' },
          body: JSON.stringify({ q: trimmedQuery, num: limit }),
        }, signal);
        return parseResults(provider, data, limit);
      }
      const url = new URL(MWMBL_SEARCH_URL);
      url.searchParams.set('q', trimmedQuery);
      const data = await requestJson(provider, url.href, {
        method: 'GET', headers: { Accept: 'application/json' },
      }, signal);
      return parseResults(provider, data, limit);
    },
  };
}
