/**
 * Token-count estimator using a word-count heuristic (words * 1.3).
 * No tokenizer dependency; sufficient for chunk-size budgeting in the browser.
 */

/** Estimate the number of tokens in `text` using a word-count heuristic. */
export function estimateTokens(text: string): number {
  const trimmed = text.trim();
  if (trimmed === '') return 0;
  const words = trimmed.split(/\s+/u).filter(Boolean);
  return Math.ceil(words.length * 1.3);
}
