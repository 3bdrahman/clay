import { VectorStoreCorruptedError } from './errors';

/**
 * Normalize a vector to unit length.
 * Throws VectorStoreCorruptedError for empty, non-finite, or zero vectors.
 */
export function normalize(v: number[]): Float32Array {
  if (v.length === 0) {
    throw new VectorStoreCorruptedError('empty embedding vector');
  }
  let sum = 0;
  for (const n of v) {
    if (!Number.isFinite(n)) {
      throw new VectorStoreCorruptedError('embedding contains non-finite values');
    }
    sum += n * n;
  }
  const norm = Math.sqrt(sum);
  if (norm === 0) {
    throw new VectorStoreCorruptedError('embedding is the zero vector');
  }
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i]! / norm;
  return out;
}

/**
 * Cosine similarity between two unit-norm vectors.
 * Returns 0 if dimensions don't match (defensive).
 */
export function cosineUnit(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot; // both unit-norm
}

export interface ScoredCandidate {
  id: string;
  score: number;
}

/**
 * Maximal Marginal Relevance (MMR) selection.
 * Balances relevance (cosine score) against diversity (similarity to already-selected).
 * lambda = 1: pure relevance; lambda = 0: pure diversity.
 */
export function mmrSelect(
  candidates: ScoredCandidate[],
  embeddings: Map<string, Float32Array>,
  k: number,
  lambda: number
): string[] {
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