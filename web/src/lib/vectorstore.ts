import type { Document, ChunkMetadata } from './types';
import type { EmbeddingsClient } from './embeddings';
import { type IDBStore } from './idb';
import { createHnswIndex, type HnswIndex } from './hnsw';
import {
  VectorStoreCorruptedError,
  VectorStoreQuotaExceededError,
  classifyError,
} from './errors';
import { _resetWriteQueue } from './vectorWriteQueue';
import { LEGACY_EMBEDDING_MODEL_ID } from './vectorLegacyMigration';
import { searchPipeline, type SearchPipelineDeps } from './vectorSearch';
import { doLoad, load as loadVectorStore, type LoadResult } from './vectorLoad';
import { addEntries, removeBySource, clear, type MutationDeps, type MutationConfig } from './vectorMutations';
import { getSourceHashes, listSources } from './vectorQueries';

export { VECTOR_DB_NAME, VECTOR_DB_VERSION } from './idb';
export { _resetWriteQueue } from './vectorWriteQueue';

/**
 * Default retrieval configuration values.
 * These can be overridden via VectorStoreConfig at creation time.
 * 
 * - DEFAULT_TOP_K: 8 results balances precision/recall for typical RAG queries.
 * - DEFAULT_SCORE_THRESHOLD: 0 (no threshold) - cosine similarity can be negative
 *   for orthogonal vectors; filtering at 0 keeps relevant but allows borderline.
 * - DEFAULT_MMR_LAMBDA: 0.5 - equal weight to relevance and diversity.
 *   Standard default for Maximal Marginal Relevance.
 * - DENSE_TOP_K_MULTIPLIER: 3 - fetch 3x top-K from dense search before fusion.
 *   Ensures sufficient candidate pool for MMR reranking.
 * - useMMR defaults false: the shipped service bundle keeps MMR reranking
 *   off deliberately (see clayServices.ts for the decision record).
 * - ANN_ENGAGE_THRESHOLD: 10k chunks — at or above it similaritySearch uses
 *   the in-repo HNSW index (sub-linear, approximate recall); below it the
 *   brute-force cosine scan stays exact. Measured scan cost at 10k chunks is
 *   ~200-370ms per query, paid twice per retrieve and once per retry, so the
 *   index earns its keep above the threshold.
 * - DEFAULT_ANN_EF_SEARCH: 64 — the HNSW search breadth at or above the
 *   candidate pool size.
 */
const DEFAULT_TOP_K = 8;
const DEFAULT_SCORE_THRESHOLD = 0;
const DEFAULT_MMR_LAMBDA = 0.5;
const DENSE_TOP_K_MULTIPLIER = 3;
const ANN_ENGAGE_THRESHOLD = 10_000;
const DEFAULT_ANN_EF_SEARCH = 64;

interface VectorEntry {
  id: string;
  text: string;
  embedding: Float32Array;
  metadata: ChunkMetadata;
}

export interface VectorStoreConfig {
  topK?: number;
  scoreThreshold?: number;
  useMMR?: boolean;
  mmrLambda?: number;
  embeddingModel?: string;
  /** Chunk count at or above which the HNSW index engages. */
  annThreshold?: number;
  /** HNSW search breadth; at or above the candidate pool size. */
  annEfSearch?: number;
}

export interface VectorStore {
  load(): Promise<void>;
  similaritySearch(query: string, k?: number): Promise<Document[]>;
  addEntries(entries: Array<{ id: string; text: string; source: string; sourceHash?: string; page?: number; embedding: number[] }>): void;
  removeBySource(source: string): number;
  clear(): void;
  getSourceHashes(source: string): Set<string>;
  listSources(): Array<{ source: string; entryCount: number }>;
  readonly stats: { entries: number };
  readonly persistenceAvailable: boolean;
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
    topK: config?.topK ?? DEFAULT_TOP_K,
    scoreThreshold: config?.scoreThreshold ?? DEFAULT_SCORE_THRESHOLD,
    useMMR: config?.useMMR ?? false,
    mmrLambda: config?.mmrLambda ?? DEFAULT_MMR_LAMBDA,
    embeddingModel: config?.embeddingModel ?? '',
    annThreshold: config?.annThreshold ?? ANN_ENGAGE_THRESHOLD,
    annEfSearch: config?.annEfSearch ?? DEFAULT_ANN_EF_SEARCH,
  };
  const entryModelId = cfg.embeddingModel || LEGACY_EMBEDDING_MODEL_ID;

  const memory = new Map<string, VectorEntry>();
  let db: IDBStore<VectorEntry> | null = null;
  let loaded = false;
  let loadingPromise: Promise<void> | null = null;
  let warnOnce = false;
  let knownDimension: number | null = null;
  let persistenceAvailable = false;
  let annIndex: HnswIndex | null = null;
  const pendingAdds: VectorEntry[] = [];

  function ensureAnnIndex(): HnswIndex | null {
    if (annIndex !== null) return annIndex;
    if (knownDimension === null || knownDimension === 0) return null;
    const index = createHnswIndex({ dimensions: knownDimension });
    for (const e of memory.values()) index.insert(e.id, e.embedding);
    annIndex = index;
    return index;
  }

  function warnFallbackOnce(cause?: unknown): void {
    if (warnOnce) return;
    warnOnce = true;
    if (import.meta.env.DEV) {
      console.warn('[vectorstore] IndexedDB unavailable; operating without persistence', cause ?? '');
    }
  }

  async function doLoadWrapper(): Promise<LoadResult> {
    return doLoad(
      { embeddingModel: cfg.embeddingModel },
      pendingAdds,
      warnFallbackOnce
    );
  }

  async function load(): Promise<void> {
    return loadVectorStore(
      loaded,
      loadingPromise,
      doLoadWrapper,
      (v) => { loaded = v; },
      (v) => { loadingPromise = v; },
      (v) => { persistenceAvailable = v; },
      (v) => { memory.clear(); for (const [k, val] of v) memory.set(k, val); },
      (v) => { db = v; },
      (v) => { knownDimension = v; },
      (v) => { pendingAdds.length = 0; pendingAdds.push(...v); }
    );
  }

  async function ensureLoaded(): Promise<void> {
    if (!loaded) await load();
  }

  async function similaritySearch(query: string, k?: number): Promise<Document[]> {
    await ensureLoaded();
    const searchConfig = {
      topK: k ?? cfg.topK,
      scoreThreshold: cfg.scoreThreshold,
      useMMR: cfg.useMMR,
      mmrLambda: cfg.mmrLambda,
      annThreshold: cfg.annThreshold,
      annEfSearch: cfg.annEfSearch,
      denseTopKMultiplier: DENSE_TOP_K_MULTIPLIER,
    };
    const deps: SearchPipelineDeps = {
      memory,
      knownDimension,
      annIndex,
      ensureAnnIndex,
      embeddings,
      config: searchConfig,
    };
    return searchPipeline(query, deps);
  }

  const mutationDeps: MutationDeps = {
    memory,
    get db() { return db; },
    get annIndex() { return annIndex; },
    pendingAdds,
    get knownDimension() { return knownDimension; },
    setKnownDimension: (v) => { knownDimension = v; },
    setAnnIndex: (v) => { annIndex = v; },
    load,
    get loaded() { return loaded; },
    get loadingPromise() { return loadingPromise; },
  };

  const mutationConfig: MutationConfig = { entryModelId };

  function addEntriesWrapper(newEntries: Array<{ id: string; text: string; source: string; sourceHash?: string; page?: number; embedding: number[] }>): void {
    addEntries(newEntries, mutationDeps, mutationConfig, classifyIDBError);
  }

  function removeBySourceWrapper(source: string): number {
    return removeBySource(source, mutationDeps, classifyIDBError);
  }

  function clearWrapper(): void {
    clear(mutationDeps, classifyIDBError);
  }

  function getSourceHashesWrapper(source: string): Set<string> {
    return getSourceHashes(source, memory);
  }

  function listSourcesWrapper(): Array<{ source: string; entryCount: number }> {
    return listSources(memory);
  }

  return {
    load,
    similaritySearch,
    addEntries: addEntriesWrapper,
    removeBySource: removeBySourceWrapper,
    clear: clearWrapper,
    getSourceHashes: getSourceHashesWrapper,
    listSources: listSourcesWrapper,
    get stats() {
      return { entries: memory.size };
    },
    get persistenceAvailable() {
      return persistenceAvailable;
    },
  };
}
