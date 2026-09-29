// Eval metrics — pure computation functions for scoring and recall.

import type { EvalResult } from './runner';

/**
 * Tokenize text into lowercase alphanumeric tokens.
 */
export function tokenize(text: string): string[] {
  return text.toLowerCase().split(/\W+/).filter(t => t.length > 0);
}

/**
 * Compute lexical overlap score between two texts.
 * Uses weighted Jaccard similarity: intersection weight / union weight.
 * Weight of each token is 1 / (1 + log(total_frequency)) to downweight common tokens.
 * Returns a score between 0 and 1.
 */
export function computeLexicalOverlap(generated: string, golden: string): number {
  const genTokens = tokenize(generated);
  const goldTokens = tokenize(golden);

  if (genTokens.length === 0 && goldTokens.length === 0) return 1;
  if (genTokens.length === 0 || goldTokens.length === 0) return 0;

  // Count frequencies
  const genFreq = new Map<string, number>();
  const goldFreq = new Map<string, number>();

  for (const t of genTokens) genFreq.set(t, (genFreq.get(t) ?? 0) + 1);
  for (const t of goldTokens) goldFreq.set(t, (goldFreq.get(t) ?? 0) + 1);

  // Compute weighted intersection and union
  let intersectionWeight = 0;
  let unionWeight = 0;

  const allTokens = new Set([...genFreq.keys(), ...goldFreq.keys()]);
  for (const token of allTokens) {
    const genCount = genFreq.get(token) ?? 0;
    const goldCount = goldFreq.get(token) ?? 0;
    const totalCount = genCount + goldCount;
    const weight = 1 / (1 + Math.log(totalCount));

    if (genCount > 0 && goldCount > 0) {
      intersectionWeight += weight * Math.min(genCount, goldCount);
    }
    unionWeight += weight * Math.max(genCount, goldCount);
  }

  return unionWeight === 0 ? 0 : intersectionWeight / unionWeight;
}

export function computeRecallAtK(relevant: number, total: number): number {
  if (total === 0) return 1;
  return Math.min(1, relevant / total);
}

export function gradeRouting(result: EvalResult): boolean {
  return result.actualSource === result.expectedSource;
}