/**
 * In-memory LRU embedding cache keyed by `${modelId}:${textHash}`.
 *
 * The `textHash` is opaque to this module — the caller (T4 embeddings client)
 * computes it (SHA-256 of text) and passes it in. T5 only stores and retrieves
 * by the provided composite key. No localStorage, no IndexedDB; intra-session only.
 *
 * LRU semantics rely on JS Map insertion order: on a cache hit, we `delete` then
 * `set` the entry so it moves to the most-recently-used position. When `size`
 * reaches `maxEntries`, the oldest entry (`map.keys().next().value`) is evicted.
 */

export interface EmbeddingCache {
  get(modelId: string, textHash: string): number[] | undefined;
  set(modelId: string, textHash: string, embedding: number[]): void;
  clear(): void;
  readonly size: number;
}

const DEFAULT_MAX_ENTRIES = 2048;

export function createEmbeddingCache(maxEntries: number = DEFAULT_MAX_ENTRIES): EmbeddingCache {
  const store = new Map<string, number[]>();

  const keyOf = (modelId: string, textHash: string): string => `${modelId}:${textHash}`;

  const get = (modelId: string, textHash: string): number[] | undefined => {
    const key = keyOf(modelId, textHash);
    const value = store.get(key);
    if (value === undefined) {
      return undefined;
    }
    // Touch: re-insert at the end so the entry is most-recently-used.
    store.delete(key);
    store.set(key, value);
    return value;
  };

  const set = (modelId: string, textHash: string, embedding: number[]): void => {
    const key = keyOf(modelId, textHash);
    // Overwriting an existing key also refreshes its recency: delete first so
    // the subsequent set places it at the end (recently used).
    if (store.has(key)) {
      store.delete(key);
    } else if (store.size >= maxEntries) {
      // Evict the oldest entry (Map preserves insertion order; the first key
      // is the least-recently-used).
      const oldest = store.keys().next().value;
      if (oldest !== undefined) {
        store.delete(oldest);
      }
    }
    store.set(key, embedding);
  };

  const clear = (): void => {
    store.clear();
  };

  return {
    get,
    set,
    clear,
    get size(): number {
      return store.size;
    },
  };
}
