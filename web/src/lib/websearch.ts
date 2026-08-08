// Web search client — Serper API (Google) and DuckDuckGo (no key)

import type { Settings, WebResult } from './types';
import {
  WebSearchProviderError,
  InvalidApiKeyError,
  RateLimitError,
  isRetryable,
} from './errors';

// DuckDuckGo HTML returns no CORS headers, so we route through the same
// dev proxy as NIM. In production, VITE_WEBSEARCH_BASE_URL should point at
// whatever edge proxy the deployment exposes (or stay empty to fall back to
// direct DuckDuckGo, which only works from origins it whitelists).
function resolveWebSearchBaseUrl(): string {
  const envUrl = (import.meta.env.VITE_WEBSEARCH_BASE_URL as string | undefined)?.trim();
  if (envUrl) return envUrl;
  if (import.meta.env.DEV) return '/ddg';
  return 'https://html.duckduckgo.com';
}

const DDG_BASE_URL = resolveWebSearchBaseUrl();

export interface WebSearchClient {
  search(query: string, k?: number): Promise<WebResult[]>;
}

// Order matters: named entities (other than &) must be decoded first so
// that a literal '&' produced by decoding one entity is not double-decoded
// when & runs last.
/**
 * Decode common HTML entities to plain text.
 * Order matters: named entities (other than &) must be decoded first.
 */
export function decodeHtmlEntities(s: string): string {
  return s
    .replace(/</g, '<')
    .replace(/>/g, '>')
    .replace(/"/g, '"')
    .replace(/'/g, "'")
    .replace(/&apos;/g, "'")
    .replace(/&/g, '&');
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
  } catch {
    return ddgUrl;
  }
}

/**
 * Parse DuckDuckGo HTML search results into WebResult array.
 * Falls back to notice if no results found.
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
  if (results.length === 0) {
    return [
      {
        type: 'web_search',
        title: 'No web search results found',
        content:
          'Configure a Serper API key in Settings for live Google results, or try a different query with DuckDuckGo.',
        url: 'https://serper.dev',
      },
    ];
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
        resp.headers.get('retry-after');
        throw new WebSearchProviderError('serper', `Rate limited (${resp.status})`, undefined, { retryable: true });
      }
      // 500 errors are not retryable for web search
      throw new WebSearchProviderError('serper', `HTTP ${resp.status}: ${resp.statusText}`, undefined, { retryable: false });
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
    const url = `${DDG_BASE_URL}/html/?q=${encodeURIComponent(query)}`;
    let resp: Response;
    try {
      resp = await fetch(url, {
        method: 'GET',
        headers: { Accept: 'text/html' },
      });
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));
      throw new WebSearchProviderError('duckduckgo', `Network error: ${error.message}`, error, { retryable: true });
    }

    if (!resp.ok) {
      throw new WebSearchProviderError('duckduckgo', `HTTP ${resp.status}: ${resp.statusText}`, undefined, { retryable: resp.status >= 500 });
    }

    const html = await resp.text();
    return parseDuckDuckGoHtml(html, k);
  }

  async function search(query: string, k = 5): Promise<WebResult[]> {
    const provider = settings.webSearchProvider;
    let lastError: Error | null = null;

    if (provider === 'serper' && settings.serperApiKey) {
      try {
        return await searchSerper(query, k);
      } catch (e) {
        lastError = e instanceof Error ? e : new Error(String(e));
        // Re-throw non-retryable auth errors and rate limit errors, but also re-throw retryable WebSearchProviderError
        if (
          lastError instanceof InvalidApiKeyError ||
          lastError instanceof RateLimitError ||
          (lastError instanceof WebSearchProviderError && isRetryable(lastError))
        ) {
          throw lastError;
        }
        // Fall through to DuckDuckGo on other Serper failures
        if (import.meta.env.DEV) console.warn('[websearch] Serper failed, falling back to DuckDuckGo:', lastError.message);
      }
    }

    if (provider !== 'none') {
      try {
        return await searchDuckDuckGo(query, k);
      } catch (e) {
        const ddgError = e instanceof Error ? e : new Error(String(e));
        // Re-throw non-retryable errors from DDG
        if (ddgError instanceof InvalidApiKeyError || ddgError instanceof RateLimitError) {
          throw ddgError;
        }
        // Both providers failed
        if (lastError) {
          throw new WebSearchProviderError(
            'duckduckgo',
            `Both Serper and DuckDuckGo failed. Serper: ${lastError.message}. DuckDuckGo: ${ddgError.message}`,
            ddgError,
            { retryable: false }
          );
        }
        throw ddgError;
      }
    }

    // Provider is 'none' - return empty results
    return [];
  }

  return { search };
}