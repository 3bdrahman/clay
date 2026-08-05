import type { Document } from './types';
import type { EmbeddingsClient } from './embeddings';

interface VectorEntry {
  id: string;
  text: string;
  source: string;
  page?: number;
  embedding: number[];
}

const VECTOR_CACHE_KEY = 'clay-vector-entries-v1';

export interface VectorStore {
  load(): Promise<void>;
  similaritySearch(query: string, k?: number): Promise<Document[]>;
  addEntries(entries: Array<{ id: string; text: string; source: string; page?: number; embedding: number[] }>): void;
  removeBySource(source: string): number;
  clear(): void;
  stats: { entries: number };
}

function loadCache(): VectorEntry[] {
  try {
    const raw = localStorage.getItem(VECTOR_CACHE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map(e => ({
      ...e,
      embedding: Array.isArray(e.embedding) ? e.embedding : Object.values(e.embedding),
    }));
  } catch {
    return [];
  }
}

function saveCache(entries: VectorEntry[]): void {
  try {
    localStorage.setItem(VECTOR_CACHE_KEY, JSON.stringify(entries));
  } catch (e) {
    console.warn('[vectorstore] failed to save cache:', e);
  }
}

function clearCache(): void {
  try {
    localStorage.removeItem(VECTOR_CACHE_KEY);
  } catch (e) {
    console.warn('[vectorstore] failed to clear cache:', e);
  }
}

export function createVectorStore(embeddings: EmbeddingsClient): VectorStore {
  let entries: VectorEntry[] = [];
  let loaded = false;
  let loadPromise: Promise<void> | null = null;

  async function doLoad(): Promise<void> {
    entries = loadCache();
    loaded = true;
  }

  async function load(): Promise<void> {
    if (loaded) return;
    if (loadPromise) return loadPromise;
    loadPromise = doLoad();
    return loadPromise;
  }

  function cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  }

  async function similaritySearch(query: string, k = 4): Promise<Document[]> {
    await load();
    if (entries.length === 0) return [];
    const queryEmbedding = (await embeddings.embed(query, { inputType: 'query' }))[0];
    if (!queryEmbedding) return [];

    const scored = entries.map(e => ({
      entry: e,
      score: cosineSimilarity(queryEmbedding, e.embedding),
    }));
    scored.sort((a, b) => b.score - a.score);

    return scored.slice(0, k).map(({ entry, score }) => ({
      id: entry.id,
      content: entry.text,
      source: entry.source,
      page: entry.page,
      score,
    }));
  }

  function addEntries(newEntries: Array<{ id: string; text: string; source: string; page?: number; embedding: number[] }>): void {
    const existingIds = new Set(entries.map(e => e.id));
    for (const e of newEntries) {
      if (!existingIds.has(e.id)) {
        entries.push({
          id: e.id,
          text: e.text,
          source: e.source,
          page: e.page,
          embedding: e.embedding,
        });
        existingIds.add(e.id);
      }
    }
    saveCache(entries);
  }

  function removeBySource(source: string): number {
    const before = entries.length;
    entries = entries.filter(e => e.source !== source);
    saveCache(entries);
    return before - entries.length;
  }

  function clear(): void {
    entries = [];
    clearCache();
  }

  return {
    load,
    similaritySearch,
    addEntries,
    removeBySource,
    clear,
    get stats() {
      return { entries: entries.length };
    },
  } as VectorStore & { stats: { entries: number } };
}