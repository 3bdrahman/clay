import { describe, it, expect } from 'vitest';
import { createBM25Index } from './bm25';

describe('createBM25Index', () => {
  it('returns a matching doc with a positive score for a single-term query', () => {
    const index = createBM25Index();
    index.add('d1', 'the quick brown fox');
    const results = index.search('fox', 5);
    expect(results).toHaveLength(1);
    expect(results[0].docId).toBe('d1');
    expect(results[0].score).toBeGreaterThan(0);
  });

  it('returns an empty array when no doc contains the query term', () => {
    const index = createBM25Index();
    index.add('d1', 'the quick brown fox');
    expect(index.search('missingterm', 5)).toEqual([]);
  });

  it('ranks a doc with higher term frequency above a lower-tf doc for the same query', () => {
    const index = createBM25Index();
    index.add('low', 'cat cat dog bird');
    index.add('high', 'cat cat cat cat dog');
    const results = index.search('cat', 5);
    expect(results[0].docId).toBe('high');
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });

  it('penalizes longer documents when b > 0 (length normalization)', () => {
    const index = createBM25Index({ b: 0.75 });
    index.add('short', 'cat cat cat dog');
    index.add('long', 'cat cat cat dog ' + 'filler filler filler '.repeat(20));
    const results = index.search('cat', 5);
    expect(results[0].docId).toBe('short');
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });

  it('returns empty after removing the only doc (state stays consistent)', () => {
    const index = createBM25Index();
    index.add('d1', 'alpha beta gamma');
    index.remove('d1');
    expect(index.search('alpha', 5)).toEqual([]);
    expect(index.size).toBe(0);
  });

  it('overwrites the postings without double-counting when re-adding the same docId', () => {
    const index = createBM25Index();
    index.add('d1', 'cat cat cat');
    index.add('d1', 'dog dog dog');
    const catResults = index.search('cat', 5);
    expect(catResults).toEqual([]);
    const dogResults = index.search('dog', 5);
    expect(dogResults).toHaveLength(1);
    expect(dogResults[0].docId).toBe('d1');
    expect(index.size).toBe(1);
  });

  it('behaves like a fresh index after clear() is called', () => {
    const index = createBM25Index();
    index.add('d1', 'one two three');
    index.add('d2', 'four five six');
    index.clear();
    expect(index.size).toBe(0);
    expect(index.search('one', 5)).toEqual([]);
    index.add('d3', 'seven eight nine');
    const results = index.search('seven', 5);
    expect(results).toHaveLength(1);
    expect(results[0].docId).toBe('d3');
  });

  it('returns an empty array (no NaN) when searching an empty index', () => {
    const index = createBM25Index();
    const results = index.search('anything', 5);
    expect(results).toEqual([]);
    for (const r of results) {
      expect(Number.isNaN(r.score)).toBe(false);
    }
  });

  it('reflects current doc count in size after adds and removes', () => {
    const index = createBM25Index();
    expect(index.size).toBe(0);
    index.add('a', 'x');
    index.add('b', 'y');
    index.add('c', 'z');
    expect(index.size).toBe(3);
    index.remove('b');
    expect(index.size).toBe(2);
    index.clear();
    expect(index.size).toBe(0);
  });

  it('matches case-insensitively for ASCII and strips punctuation', () => {
    const index = createBM25Index();
    index.add('d1', 'HELLO, World!');
    const results = index.search('hello', 5);
    expect(results).toHaveLength(1);
    expect(results[0].docId).toBe('d1');
    const results2 = index.search('WORLD', 5);
    expect(results2).toHaveLength(1);
    expect(results2[0].docId).toBe('d1');
  });

  it('does NOT collide accented Latin forms with their ASCII base (documented limitation)', () => {
    const index = createBM25Index();
    index.add('d1', 'Niño');
    expect(index.search('nino', 5)).toEqual([]);
    const results = index.search('niño', 5);
    expect(results).toHaveLength(1);
    expect(results[0].docId).toBe('d1');
  });

  it('respects topK and returns only the highest-scoring docs', () => {
    const index = createBM25Index();
    index.add('a', 'term a a a a');
    index.add('b', 'term b b b');
    index.add('c', 'term c c');
    const top1 = index.search('term', 1);
    expect(top1).toHaveLength(1);
    expect(top1[0].docId).toBe('a');
    const top2 = index.search('term', 2);
    expect(top2).toHaveLength(2);
    expect(top2[0].docId).toBe('a');
    expect(top2[1].docId).toBe('b');
  });

  it('honors k1 and b overrides supplied via BM25Config', () => {
    const relaxed = createBM25Index({ k1: 0.1, b: 0 });
    const strict = createBM25Index({ k1: 3, b: 1 });
    relaxed.add('short', 'cat cat cat dog');
    relaxed.add('long', 'cat cat cat dog ' + 'filler '.repeat(30));
    strict.add('short', 'cat cat cat dog');
    strict.add('long', 'cat cat cat dog ' + 'filler '.repeat(30));
    const relaxedResults = relaxed.search('cat', 5);
    const strictResults = strict.search('cat', 5);
    expect(relaxedResults[0].docId).toBe('short');
    expect(strictResults[0].docId).toBe('short');
    const relaxedGap = relaxedResults[0].score - relaxedResults[1].score;
    const strictGap = strictResults[0].score - strictResults[1].score;
    expect(strictGap).toBeGreaterThan(relaxedGap);
  });
});
