// Web search client — Serper API (Google) and DuckDuckGo (no key)

import type { Settings, WebResult } from './types';

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
 * Falls back to demo notice if no results found.
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
        title: 'Web search unavailable in demo mode',
        content:
          'Configure a Serper API key in Settings for live Google results, or enable DuckDuckGo (no key required).',
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
 */
export function createWebSearchClient(settings: Settings): WebSearchClient {
  async function searchSerper(query: string, k: number): Promise<WebResult[]> {
    const resp = await fetch('https://google.serper.dev/search', {
      method: 'POST',
      headers: {
        'X-API-KEY': settings.serperApiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ q: query, num: k }),
    });
    if (!resp.ok) throw new Error(`Serper ${resp.status}`);
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
    try {
      const resp = await fetch(url, {
        method: 'GET',
        headers: { Accept: 'text/html' },
      });
      if (!resp.ok) {
        return [
          {
            type: 'web_search',
            title: 'Web search unavailable',
            content: 'Could not reach DuckDuckGo. Try Serper in Settings.',
          },
        ];
      }
      const html = await resp.text();
      return parseDuckDuckGoHtml(html, k);
    } catch {
      return [
        {
          type: 'web_search',
          title: 'Web search unavailable',
          content: 'Network error fetching search results.',
        },
      ];
    }
  }

  async function search(query: string, k = 5): Promise<WebResult[]> {
    const provider = settings.webSearchProvider;
    if (provider === 'serper' && settings.serperApiKey) {
      try {
        return await searchSerper(query, k);
      } catch {
        // fall through to duckduckgo
      }
    }
    if (provider !== 'none') {
      return searchDuckDuckGo(query, k);
    }
    return [];
  }

  return { search };
}
