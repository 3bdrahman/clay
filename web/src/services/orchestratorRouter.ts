/**
 * Router logic for the workflow orchestrator — extracted from orchestrator.ts for modularity.
 * Handles routing context building, web search availability, and source selection.
 */

import type { Settings, SourceType } from '../lib/types';
import type { OrchestratorDeps } from './orchestrator';

export const VALID_SOURCE_TYPES: SourceType[] = ['vectorstore', 'python', 'websearch'];

export const ROUTER_CONFIDENCE_THRESHOLD = 0.6;

export function webSearchAvailability(settings: Settings): string {
  if (settings.webSearchProvider === 'none') return 'disabled by the user — never route here';
  if (settings.webSearchProvider === 'serper') {
    return settings.serperApiKey
      ? 'available (Serper API key configured)'
      : 'selected but no Serper API key configured — requests would fail';
  }
  return import.meta.env.DEV
    ? 'available (DuckDuckGo via the dev proxy)'
    : 'DEGRADED: DuckDuckGo cannot be reached from browser deployments in production (no CORS headers) — route here only for general-knowledge questions the sources above cannot answer';
}

export function buildRouterContext(deps: OrchestratorDeps): string {
  const lines: string[] = [];

  const datasets = deps.analyzer.listDatasets();
  if (datasets.length > 0) {
    const dsLines = datasets.slice(0, 8).map(d => {
      const cols = (d.columns ?? []).slice(0, 12).join(', ');
      return `- ${d.name} (${d.rowCount} rows): [${cols}]`;
    });
    lines.push(`Datasets available for data analysis (source "python"):\n${dsLines.join('\n')}`);
  } else {
    lines.push('Datasets: none loaded — data analysis cannot answer anything');
  }

  const sources = deps.vectorstore.listSources();
  if (sources.length > 0) {
    const docLines = sources.slice(0, 8).map(s => `- ${s.source} (${s.entryCount} chunks)`);
    lines.push(`Documents available for vector search (source "vectorstore"):\n${docLines.join('\n')}`);
  } else {
    lines.push('Documents: none uploaded — vector search cannot answer anything');
  }

  lines.push(`Web search (source "websearch"): ${webSearchAvailability(deps.settings)}`);

  return lines.join('\n\n');
}

export function nextUntriedSource(tried: SourceType[]): SourceType {
  // Prefer the richest untried source: the real data beats documents beats
  // the open web, and web search is CORS-degraded in production deployments.
  const richness: SourceType[] = ['python', 'vectorstore', 'websearch'];
  for (const s of richness) {
    if (!tried.includes(s)) return s;
  }
  return 'vectorstore';
}