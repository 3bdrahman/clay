import { describe, it, expect } from 'vitest';
import { createHnswIndex } from './hnsw';

// Deterministic xorshift PRNG so graph construction and recall assertions
// are reproducible run-to-run.
function makeRng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

function makeUnit(dim: number, rng: () => number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = rng() - 0.5;
  let sum = 0;
  for (let i = 0; i < dim; i++) sum += v[i]! * v[i]!;
  const norm = Math.sqrt(sum);
  for (let i = 0; i < dim; i++) v[i] = v[i]! / norm;
  return v;
}

function dot(a: Float32Array, b: Float32Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i]! * b[i]!;
  return d;
}

function bruteForceTopK(
  vectors: Map<string, Float32Array>,
  query: Float32Array,
  k: number,
): string[] {
  return [...vectors.entries()]
    .map(([label, v]) => ({ label, score: dot(query, v) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(e => e.label);
}

describe('hnsw index', () => {
  it('retrieves the same top-K as brute-force scan (recall ≥ 0.9, 384-dim production-like)', () => {
    const rng = makeRng(42);
    const dim = 384;
    const index = createHnswIndex({ dimensions: dim, seed: 7 });
    const vectors = new Map<string, Float32Array>();
    for (let i = 0; i < 400; i++) {
      const label = `v-${i}`;
      const v = makeUnit(dim, rng);
      vectors.set(label, v);
      index.insert(label, v);
    }

    let hits = 0;
    let total = 0;
    for (let q = 0; q < 20; q++) {
      const query = makeUnit(dim, rng);
      const expected = bruteForceTopK(vectors, query, 8);
      const actual = index.search(query, 8, 128).map(e => e.label);
      total += expected.length;
      hits += expected.filter(l => actual.includes(l)).length;
    }
    expect(hits / total).toBeGreaterThanOrEqual(0.9);
  });

  it('retrieves the same top-K as brute-force scan (recall ≥ 0.9, 32-dim)', () => {
    const rng = makeRng(1234);
    const dim = 32;
    const index = createHnswIndex({ dimensions: dim, seed: 7 });
    const vectors = new Map<string, Float32Array>();
    for (let i = 0; i < 400; i++) {
      const label = `v-${i}`;
      const v = makeUnit(dim, rng);
      vectors.set(label, v);
      index.insert(label, v);
    }

    let hits = 0;
    let total = 0;
    for (let q = 0; q < 20; q++) {
      const query = makeUnit(dim, rng);
      const expected = bruteForceTopK(vectors, query, 8);
      const actual = index.search(query, 8).map(e => e.label);
      total += expected.length;
      hits += expected.filter(l => actual.includes(l)).length;
    }
    expect(hits / total).toBeGreaterThanOrEqual(0.9);
  });

  it('sorts results by descending score', () => {
    const rng = makeRng(99);
    const index = createHnswIndex({ dimensions: 16, seed: 7 });
    for (let i = 0; i < 50; i++) index.insert(`v-${i}`, makeUnit(16, rng));
    const results = index.search(makeUnit(16, rng), 8);
    expect(results.length).toBeGreaterThan(1);
    for (let i = 1; i < results.length; i++) {
      expect(results[i]!.score).toBeLessThanOrEqual(results[i - 1]!.score);
    }
  });

  it('excludes deleted entries from results', () => {
    const rng = makeRng(5);
    const index = createHnswIndex({ dimensions: 16, seed: 7 });
    for (let i = 0; i < 50; i++) index.insert(`v-${i}`, makeUnit(16, rng));
    const query = makeUnit(16, rng);
    const topLabel = index.search(query, 1)[0]!.label;

    index.markDeleted(topLabel);
    const after = index.search(query, 8).map(e => e.label);
    expect(after).not.toContain(topLabel);
    expect(after.length).toBe(8);
  });

  it('re-indexing a label under a new vector routes queries to the new vector', () => {
    const rng = makeRng(11);
    const index = createHnswIndex({ dimensions: 16, seed: 7 });
    for (let i = 0; i < 50; i++) index.insert(`v-${i}`, makeUnit(16, rng));

    // Replace v-0 with a vector identical to v-1's neighborhood direction.
    const replacement = makeUnit(16, rng);
    index.insert('v-0', replacement);
    const found = index.search(replacement, 4).map(e => e.label);
    expect(found[0]).toBe('v-0');
    // The stale node must not surface twice or route to the old vector.
    expect(found.filter(l => l === 'v-0')).toHaveLength(1);
  });

  it('is deterministic for a fixed seed and insert order', () => {
    const build = (): Array<{ label: string; score: number }> => {
      const rng = makeRng(777);
      const index = createHnswIndex({ dimensions: 32, seed: 7 });
      for (let i = 0; i < 150; i++) index.insert(`v-${i}`, makeUnit(32, rng));
      return index.search(makeUnit(32, rng), 10);
    };
    expect(build()).toEqual(build());
  });

  it('returns fewer than k when the live index is smaller than k', () => {
    const rng = makeRng(3);
    const index = createHnswIndex({ dimensions: 16, seed: 7 });
    for (let i = 0; i < 3; i++) index.insert(`v-${i}`, makeUnit(16, rng));
    expect(index.search(makeUnit(16, rng), 8)).toHaveLength(3);
  });

  it('tracks the live size after deletions', () => {
    const rng = makeRng(3);
    const index = createHnswIndex({ dimensions: 16, seed: 7 });
    for (let i = 0; i < 10; i++) index.insert(`v-${i}`, makeUnit(16, rng));
    expect(index.size).toBe(10);
    index.markDeleted('v-0');
    index.markDeleted('v-1');
    expect(index.size).toBe(8);
  });
});
