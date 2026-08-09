import type { Document, ChunkMetadata } from './types';
import type { EmbeddingsClient } from './embeddings';
import { estimateTokens } from './tokens';
import { openIDB, wrapIDBStore, type IDBStore } from './idb';
import type { BM25Index } from './bm25';
import {
  VectorStoreCorruptedError,
  VectorStoreQuotaExceededError,
  classifyError,
} from './errors';

/**
 * Module-level write coordinator.
 *
 * Why: `addEntries` is synchronous (returns void) but IndexedDB writes are
 * inherently async. Multiple VectorStore instances may exist in tests (and
 * could exist in production during HMR / store resets). When instance A
 * writes an entry and instance B is constructed and loads, B must see A's
 * write — but B's `load()` has no knowledge of A's in-flight IDB operations.
 *
 * Fix: every IDB write is registered on a shared writeQueue promise.
 * Every `load()` awaits writeQueue before reading. Once any `load()`
 * resolves, all writes issued before that load() call have landed in IDB.
 */
let writeQueue: Promise<void> = Promise.resolve();
let writeQueueFailed = false;
let writeQueueFailure: Error | null = null;

function enqueueWrite(op: () => Promise<unknown>): void {
  writeQueue = writeQueue.then<void>(() => op().then(() => undefined, (e: unknown) => {
    writeQueueFailed = true;
    writeQueueFailure = e instanceof Error ? e : new Error(String(e));
    console.error('[vectorstore] async op failed:', e);
  }));
}

export function _resetWriteQueue(): void {
  writeQueue = Promise.resolve();
}

const LEGACY_KEY = 'clay-vector-entries-v1';
const DB_NAME = 'clay-vector-db';
const DB_VERSION = 1;
const STORE_NAME = 'entries';

interface VectorEntry {
  id: string;
  text: string;
  embedding: number[];
  metadata: ChunkMetadata;
}

export interface VectorStoreConfig {
  topK?: number;
  scoreThreshold?: number;
  useMMR?: boolean;
  mmrLambda?: number;
  useHybrid?: boolean;
  hybridAlpha?: number;
  bm25Index?: BM25Index;
}

export interface VectorStore {
  load(): Promise<void>;
  similaritySearch(query: string, k?: number): Promise<Document[]>;
  addEntries(entries: Array<{ id: string; text: string; source: string; sourceHash?: string; page?: number; embedding: number[] }>): void;
  removeBySource(source: string): number;
  clear(): void;
  getSourceHashes(source: string): Set<string>;
  readonly stats: { entries: number };
  readonly persistenceAvailable: boolean;
}

function coerceLegacyEmbedding(v: unknown): number[] | null {
  if (Array.isArray(v) && v.every((n) => typeof n === 'number' && Number.isFinite(n))) return v as number[];
  if (v && typeof v === 'object') {
    const vals = Object.values(v as Record<string, unknown>);
    if (vals.every((n) => typeof n === 'number' && Number.isFinite(n))) return vals as number[];
  }
  return null;
}

function readLegacy(): VectorEntry[] | null {
  try {
    const raw = localStorage.getItem(LEGACY_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const out: VectorEntry[] = [];
    parsed.forEach((e: unknown, i: number) => {
      if (!e || typeof e !== 'object') return;
      const r = e as Record<string, unknown>;
      if (typeof r.id !== 'string' || typeof r.text !== 'string' || typeof r.source !== 'string') return;
      const emb = coerceLegacyEmbedding(r.embedding);
      if (!emb) return;
      out.push({
        id: r.id,
        text: r.text,
        embedding: emb,
        metadata: {
          source: r.source,
          sourceHash: '',
          page: typeof r.page === 'number' ? r.page : undefined,
          charStart: 0,
          charEnd: r.text.length,
          chunkIndex: i,
          tokenCount: estimateTokens(r.text),
          modelId: 'unknown',
          updatedAt: Date.now(),
        },
      });
    });
    return out;
  } catch {
    return null;
  }
}

function normalize(v: number[]): number[] {
  let sum = 0;
  for (const n of v) sum += n * n;
  const norm = Math.sqrt(sum);
  if (norm === 0) return v.slice();
  return v.map((n) => n / norm);
}

function cosineUnit(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // both unit-norm
}

function minMaxNormalize(values: number[]): number[] {
  if (values.length === 0) return values;
  let min = values[0] as number, max = values[0] as number;
  for (const v of values) { if (v < min) min = v; if (v > max) max = v; }
  if (max === min) return values.map(() => 0);
  return values.map((v) => (v - min) / (max - min));
}

interface ScoredCandidate { id: string; score: number; }

function mmrSelect(candidates: ScoredCandidate[], embeddings: Map<string, number[]>, k: number, lambda: number): string[] {
  const selected: string[] = [];
  const remaining = candidates.slice();
  while (selected.length < k && remaining.length > 0) {
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < remaining.length; i++) {
      const c = remaining[i] as ScoredCandidate;
      const relevance = c.score;
      let diversity = 0;
      for (const sid of selected) {
        const a = embeddings.get(c.id);
        const b = embeddings.get(sid);
        if (a && b) diversity = Math.max(diversity, cosineUnit(a, b));
      }
      const mmr = lambda * relevance - (1 - lambda) * diversity;
      if (mmr > bestScore) { bestScore = mmr; bestIdx = i; }
    }
    const picked = remaining.splice(bestIdx, 1)[0];
    if (picked) selected.push(picked.id);
  }
  return selected;
}

/**
 * Classify IndexedDB DOMException into typed vector store errors.
 */
function classifyIDBError(error: unknown, operation: string): Error {
  if (error instanceof DOMException) {
    switch (error.name) {
      case 'QuotaExceededError':
        return new VectorStoreQuotaExceededError(error);
      case 'InvalidStateError':
      case 'TransactionInactiveError':
      case 'DataError':
        return new VectorStoreCorruptedError(`IDB ${operation} failed: ${error.name} - ${error.message}`, error);
      case 'ConstraintError':
        return new VectorStoreCorruptedError(`IDB constraint violation during ${operation}: ${error.message}`, error);
      case 'AbortError':
        return classifyError(error, 'vectorstore', operation);
      default:
        return new VectorStoreCorruptedError(`IDB ${operation} failed: ${error.name} - ${error.message}`, error);
    }
  }
  return classifyError(error, 'vectorstore', operation);
}

export function createVectorStore(embeddings: EmbeddingsClient, config?: VectorStoreConfig): VectorStore {
  const cfg = {
    topK: config?.topK ?? 8,
    scoreThreshold: config?.scoreThreshold ?? 0,
    useMMR: config?.useMMR ?? false,
    mmrLambda: config?.mmrLambda ?? 0.5,
    useHybrid: config?.useHybrid ?? false,
    hybridAlpha: config?.hybridAlpha ?? 0.5,
    bm25Index: config?.bm25Index,
  };

  const memory = new Map<string, VectorEntry>();
  let db: IDBStore<VectorEntry> | null = null;
  let loaded = false;
  let loadingPromise: Promise<void> | null = null;
  let warnOnce = false;
  let knownDimension: number | null = null;
  let persistenceAvailable = false;
  const pendingAdds: VectorEntry[] = [];

  function warnFallbackOnce(): void {
    if (warnOnce) return;
    warnOnce = true;
    console.warn('[vectorstore] IndexedDB unavailable; operating without persistence');
  }

  async function doLoad(): Promise<void> {
    try {
      const idb = await openIDB(DB_NAME, DB_VERSION, (raw) => {
        if (!raw.objectStoreNames.contains(STORE_NAME)) {
          const store = raw.createObjectStore(STORE_NAME, { keyPath: 'id' });
          store.createIndex('source', 'metadata.source', { unique: false });
          store.createIndex('modelId', 'metadata.modelId', { unique: false });
        }
      });
      db = wrapIDBStore<VectorEntry>(idb, STORE_NAME);
      if (pendingAdds.length > 0) {
        const toFlush = pendingAdds.splice(0, pendingAdds.length);
        for (const e of toFlush) await db.put(e);
      }
      const existing = await db.getAll();
      if (existing.length === 0) {
        const legacy = readLegacy();
        if (legacy && legacy.length > 0) {
          await db.putMany(legacy);
          localStorage.removeItem(LEGACY_KEY);
          for (const e of legacy) memory.set(e.id, e);
        } else if (legacy !== null) {
          // Corruption path: parsed OK but zero valid entries. Still clear the legacy key
          // so we don't re-attempt migration on every load. User's data was unreadable; no
          // further recovery possible.
          localStorage.removeItem(LEGACY_KEY);
        }
      } else {
        for (const e of existing) memory.set(e.id, e);
      }
    } catch (e) {
      warnFallbackOnce();
      db = null;
      persistenceAvailable = false;
      // Don't throw here - allow fallback to in-memory mode
    }
    loaded = true;
    if (db !== null) persistenceAvailable = true;
  }

  async function load(): Promise<void> {
    if (loaded) return;
    if (loadingPromise) return loadingPromise;
    loadingPromise = (async () => {
      await writeQueue;
      if (writeQueueFailed && writeQueueFailure) {
        throw writeQueueFailure;
      }
      await doLoad();
      await writeQueue;
      if (writeQueueFailed && writeQueueFailure) {
        throw writeQueueFailure;
      }
    })();
    return loadingPromise;
  }

  async function ensureLoaded(): Promise<void> {
    if (!loaded) await load();
  }

  async function similaritySearch(query: string, k?: number): Promise<Document[]> {
    await ensureLoaded();
    const topK = k ?? cfg.topK;
    if (memory.size === 0) return [];
    let queryEmbeddingRaw: number[][];
    try {
      queryEmbeddingRaw = await embeddings.embed(query, { inputType: 'query' });
    } catch (e) {
      throw classifyError(e, 'embeddings', 'similaritySearch');
    }
    const queryEmbedding = queryEmbeddingRaw[0];
    if (!queryEmbedding) return [];

    // Dense cosine scoring (assumes normalized embeddings).
    const scored: ScoredCandidate[] = [];
    for (const e of memory.values()) {
      scored.push({ id: e.id, score: cosineUnit(queryEmbedding, e.embedding) });
    }
    scored.sort((a, b) => b.score - a.score);
    const denseTop = scored.slice(0, Math.max(topK * 3, 6));

    let merged: ScoredCandidate[];
    if (cfg.useHybrid && cfg.bm25Index) {
      const bm25Hits = cfg.bm25Index.search(query, topK * 3);
      const denseNorm = minMaxNormalize(denseTop.map((s) => s.score));
      const bm25Scores = bm25Hits.map((h) => h.score);
      const bm25Norm = minMaxNormalize(bm25Scores);
      merged = [];
      const allIds = new Set<string>();
      for (const s of denseTop) allIds.add(s.id);
      for (const h of bm25Hits) allIds.add(h.docId);
      const denseLookup = new Map(denseTop.map((s, i) => [s.id, denseNorm[i]!] as const));
      const bm25Lookup = new Map(bm25Hits.map((h, i) => [h.docId, bm25Norm[i]!] as const));
      for (const id of allIds) {
        const d = denseLookup.get(id) ?? 0;
        const b = bm25Lookup.get(id) ?? 0;
        merged.push({ id, score: cfg.hybridAlpha * d + (1 - cfg.hybridAlpha) * b });
      }
      merged.sort((a, b) => b.score - a.score);
    } else {
      merged = denseTop;
    }

    const filtered = merged.filter((c) => c.score >= cfg.scoreThreshold);

    let chosen: string[];
    if (cfg.useMMR) {
      const embMap = new Map<string, number[]>();
      for (const id of new Set(filtered.map((f) => f.id))) {
        const e = memory.get(id);
        if (e) embMap.set(id, e.embedding);
      }
      chosen = mmrSelect(filtered, embMap, topK, cfg.mmrLambda);
    } else {
      chosen = filtered.slice(0, topK).map((c) => c.id);
    }

    const results: Document[] = [];
    for (const id of chosen) {
      const e = memory.get(id);
      if (!e) continue;
      const score = merged.find((m) => m.id === id)?.score ?? 0;
      const doc: Document = {
        id: e.id,
        content: e.text,
        source: e.metadata.source,
        page: e.metadata.page,
        score,
        metadata: e.metadata as unknown as Record<string, unknown>,
      };
      results.push(doc);
    }
    return results;
  }

  function addEntries(newEntries: Array<{ id: string; text: string; source: string; page?: number; embedding: number[] }>): void {
    if (!loaded && !loadingPromise) void load();

    for (const e of newEntries) {
      const emb = normalize(e.embedding);
      if (knownDimension === null) {
        knownDimension = emb.length;
      } else if (emb.length !== knownDimension) {
        throw new VectorStoreCorruptedError(
          `embedding dimension mismatch: stored ${knownDimension}, got ${emb.length}`
        );
      }
      const entry: VectorEntry = {
        id: e.id,
        text: e.text,
        embedding: emb,
        metadata: {
          source: e.source,
          sourceHash: '',
          page: e.page,
          charStart: 0,
          charEnd: e.text.length,
          chunkIndex: memory.size,
          tokenCount: estimateTokens(e.text),
          modelId: 'unknown',
          updatedAt: Date.now(),
        },
      };
      memory.set(entry.id, entry);
      pendingAdds.push(entry);
      if (db) {
        const capture = db;
        enqueueWrite(async () => {
          try {
            await capture.put(entry);
          } catch (e) {
            throw classifyIDBError(e, 'put');
          }
        });
      } else if (!loaded && !loadingPromise) {
        void load();
      }
    }
  }

  function removeBySource(source: string): number {
    const before = memory.size;
    for (const [id, e] of memory) {
      if (e.metadata.source === source) memory.delete(id);
    }
    if (db) enqueueWrite(async () => {
      try {
        await db!.deleteByIndex('source', source);
      } catch (e) {
        throw classifyIDBError(e, 'deleteByIndex');
      }
    });
    return before - memory.size;
  }

  function clear(): void {
    memory.clear();
    knownDimension = null;
    try { localStorage.removeItem(LEGACY_KEY); } catch { /* noop */ }
    if (db) enqueueWrite(async () => {
      try {
        await db!.clear();
      } catch (e) {
        throw classifyIDBError(e, 'clear');
      }
    });
  }

  function getSourceHashes(source: string): Set<string> {
    const hashes = new Set<string>();
    for (const entry of memory.values()) {
      if (entry.metadata.source === source && entry.metadata.sourceHash) {
        hashes.add(entry.metadata.sourceHash);
      }
    }
    return hashes;
  }

  return {
    load,
    similaritySearch,
    addEntries,
    removeBySource,
    clear,
    getSourceHashes,
    get stats() {
      return { entries: memory.size };
    },
    get persistenceAvailable() {
      return persistenceAvailable;
    },
  };
}