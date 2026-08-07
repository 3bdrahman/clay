import { describe, it, expect } from 'vitest';
import { createEmbeddingCache } from './embeddingCache';

describe('createEmbeddingCache', () => {
  it('returns the embedding after a set then get', () => {
    const cache = createEmbeddingCache();
    cache.set('m1', 'h1', [0.1, 0.2, 0.3]);
    expect(cache.get('m1', 'h1')).toEqual([0.1, 0.2, 0.3]);
  });

  it('returns undefined for a missing key', () => {
    const cache = createEmbeddingCache();
    expect(cache.get('m1', 'missing')).toBeUndefined();
  });

  it('evicts the oldest entry when maxEntries is exceeded', () => {
    const cache = createEmbeddingCache(2);
    cache.set('m', 'a', [1]);
    cache.set('m', 'b', [2]);
    cache.set('m', 'c', [3]);
    expect(cache.get('m', 'a')).toBeUndefined();
    expect(cache.get('m', 'b')).toEqual([2]);
    expect(cache.get('m', 'c')).toEqual([3]);
    expect(cache.size).toBe(2);
  });

  it('touches the entry on get so eviction is true LRU, not FIFO', () => {
    const cache = createEmbeddingCache(3);
    cache.set('m', 'a', [1]);
    cache.set('m', 'b', [2]);
    cache.set('m', 'c', [3]);
    // Access 'a' — it becomes most-recently-used.
    expect(cache.get('m', 'a')).toEqual([1]);
    // Insert a 4th; 'b' (oldest now) should be evicted, not 'a'.
    cache.set('m', 'd', [4]);
    expect(cache.get('m', 'a')).toEqual([1]);
    expect(cache.get('m', 'b')).toBeUndefined();
    expect(cache.get('m', 'c')).toEqual([3]);
    expect(cache.get('m', 'd')).toEqual([4]);
  });

  it('clears all entries and resets size to 0', () => {
    const cache = createEmbeddingCache();
    cache.set('m1', 'h1', [1]);
    cache.set('m1', 'h2', [2]);
    expect(cache.size).toBe(2);
    cache.clear();
    expect(cache.size).toBe(0);
    expect(cache.get('m1', 'h1')).toBeUndefined();
    expect(cache.get('m1', 'h2')).toBeUndefined();
  });

  it('reflects current count in size after adds, gets, and clear', () => {
    const cache = createEmbeddingCache(2);
    expect(cache.size).toBe(0);
    cache.set('m', 'a', [1]);
    expect(cache.size).toBe(1);
    cache.set('m', 'b', [2]);
    expect(cache.size).toBe(2);
    // get does not change size, even on hit.
    cache.get('m', 'a');
    expect(cache.size).toBe(2);
    // Overwriting an existing key does not grow size.
    cache.set('m', 'a', [9]);
    expect(cache.size).toBe(2);
    // Eviction keeps size at cap.
    cache.set('m', 'c', [3]);
    expect(cache.size).toBe(2);
    cache.clear();
    expect(cache.size).toBe(0);
  });

  it('coexists entries with the same textHash across different modelIds', () => {
    const cache = createEmbeddingCache();
    cache.set('modelA', 'shared-hash', [1, 2]);
    cache.set('modelB', 'shared-hash', [3, 4]);
    expect(cache.get('modelA', 'shared-hash')).toEqual([1, 2]);
    expect(cache.get('modelB', 'shared-hash')).toEqual([3, 4]);
  });

  it('defaults maxEntries to 2048 (cap enforced at 2048)', () => {
    const cache = createEmbeddingCache();
    // Insert 2049 distinct entries; size must cap at 2048 and the very first
    // inserted key must be evicted (it was never accessed, so it is oldest).
    for (let i = 0; i < 2049; i++) {
      cache.set('m', `k${i}`, [i]);
    }
    expect(cache.size).toBe(2048);
    expect(cache.get('m', 'k0')).toBeUndefined();
    // The most recently inserted is still present.
    expect(cache.get('m', 'k2048')).toEqual([2048]);
  });
});
