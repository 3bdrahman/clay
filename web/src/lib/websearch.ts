// Web search client — Serper API (Google) and DuckDuckGo (no key)

import type { Settings, WebResult } from './types';
import { WebSearchProviderError } from './errors';

// DuckDuckGo HTML returns no CORS headers, so the dev server proxies /ddg
// to html.duckduckgo.com. In production, VITE_WEBSEARCH_BASE_URL should point at
// an edge proxy. Direct production requests are blocked by CORS.
function resolveWebSearchBaseUrl(): string {
  const envUrl = (import.meta.env.VITE_WEBSEARCH_BASE_URL as string | undefined)?.trim();
  if (envUrl) return envUrl.replace(/\/+$/, '');
  if (import.meta.env.DEV) return '/ddg';
  return '';
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
  return resolveWebSearchBaseUrl()
    ? { available: true, message: 'DuckDuckGo search is available through the configured proxy.' }
    : { available: false, message: 'DuckDuckGo is unavailable on this site. Choose Serper and add its API key in Settings, or use your loaded files.' };
}

/**
 * Default result count for web search when the caller omits `k`. Mirrors the
 * orchestrator's `WEB_SEARCH_RESULT_COUNT` — kept independent because the
 * websearch module is reusable from non-orchestrator call sites (e.g. evals).
 */
const DEFAULT_WEB_SEARCH_K = 5;

export interface WebSearchClient {
  search(query: string, k?: number): Promise<WebResult[]>;
}

/**
 * Decode common HTML entities to plain text.
 * Order matters: the ampersand entity must be decoded last so entity text
 * produced by an earlier decode is never double-decoded.
 */
export function decodeHtmlEntities(s: string): string {
  return s
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&');
}

/**
 * Extract the real destination URL from a DuckDuckGo redirect link.
 * If parsing fails, returns the original URL.
 */
export function extractRealUrl(ddgUrl: string): string {
  try {
    const u = new URL(ddgUrl);
    const uddg = u.searchParams.get('uddg');
    return uddg || ddgUrl;
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[websearch] extractRealUrl: malformed URL (returning input as-is):', e);
    }
    return ddgUrl;
  }
}

/**
 * Parse DuckDuckGo HTML search results into WebResult array.
 * An empty response yields no sources; configuration notices are not citations.
 * @param html - Raw HTML from DuckDuckGo
 * @param k - Maximum number of results to return
 * @returns Array of WebResult objects
 */
export function parseDuckDuckGoHtml(html: string, k: number): WebResult[] {
  const results: WebResult[] = [];
  const resultRegex =
    /<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)<\/a>[\s\S]*?<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;

  let m: RegExpExecArray | null;
  while ((m = resultRegex.exec(html)) !== null && results.length < k) {
    const url = m[1];
    const title = m[2];
    const snippetRaw = m[3];
    const snippet = snippetRaw.replace(/<[^>]*>/g, '').trim();
    results.push({
      type: 'web_search',
      title: decodeHtmlEntities(title.trim()),
      content: decodeHtmlEntities(snippet),
      url: extractRealUrl(url),
    });
  }
  return results;
}

/**
 * Create a web search client based on settings (Serper or DuckDuckGo).
 * @param settings - App settings with provider choice and API keys
 * @returns WebSearchClient with search(query, k?) method
 * @throws WebSearchProviderError on provider failures
 */
export function createWebSearchClient(settings: Settings): WebSearchClient {
  async function searchSerper(query: string, k: number): Promise<WebResult[]> {
    let resp: Response;
    try {
      resp = await fetch('https://google.serper.dev/search', {
        method: 'POST',
        headers: {
          'X-API-KEY': settings.serperApiKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ q: query, num: k }),
      });
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));
      throw new WebSearchProviderError('serper', `Network error: ${error.message}`, error, { retryable: true });
    }

    if (!resp.ok) {
      if (resp.status === 401 || resp.status === 403) {
        throw new WebSearchProviderError('serper', `Invalid API key (${resp.status})`, undefined, { retryable: false });
      }
      if (resp.status === 429) {
        throw new WebSearchProviderError('serper', `Rate limited (${resp.status})`, undefined, { retryable: true });
      }
      throw new WebSearchProviderError('serper', `HTTP ${resp.status}: ${resp.statusText}`, undefined, { retryable: resp.status >= 500 });
    }

    const data = await resp.json();
    const organic = data.organic || [];
    return organic.slice(0, k).map((r: { title?: string; snippet?: string; link?: string }) => ({
      type: 'web_search' as const,
      title: r.title || 'Untitled',
      content: r.snippet || '',
      url: r.link,
    }));
  }

  async function searchDuckDuckGo(query: string, k: number): Promise<WebResult[]> {
    const url = `${resolveWebSearchBaseUrl()}/html/?q=${encodeURIComponent(query)}`;
    let resp: Response;
    try {
      resp = await fetch(url, {
        method: 'GET',
        headers: { Accept: 'text/html' },
      });
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));
      throw new WebSearchProviderError('duckduckgo', `Could not reach the search proxy: ${error.message}`, error, {
        retryable: true,
      });
    }

    if (!resp.ok) {
      throw new WebSearchProviderError('duckduckgo', `HTTP ${resp.status}: ${resp.statusText}`, undefined, { retryable: resp.status >= 500 });
    }

    const html = await resp.text();
    return parseDuckDuckGoHtml(html, k);
  }

  async function search(query: string, k = DEFAULT_WEB_SEARCH_K): Promise<WebResult[]> {
    const provider = settings.webSearchProvider;
    if (provider === 'none') return [];
    const availability = getWebSearchAvailability(settings);
    if (!availability.available) {
      throw new WebSearchProviderError(provider, availability.message, undefined, { retryable: false });
    }
    // The selected provider owns this request, including its failures. Never
    // send the user's query to a second provider without their selection.
    return provider === 'serper' ? searchSerper(query, k) : searchDuckDuckGo(query, k);
  }

  return { search };
}
