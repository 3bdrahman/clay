/**
 * Hand-rolled BM25 (Okapi) keyword index. Zero npm dependencies.
 *
 * Used by the hybrid retrieval path (dense vectors + BM25 reciprocal-rank
 * fusion) in the vector store. Maintains an inverted index with per-term
 * document frequencies and term frequencies, plus corpus-level statistics
 * (doc count N, average doc length) needed by the BM25 scoring formula.
 *
 * Default tuning: k1 = 1.5, b = 0.75 (standard Okapi BM25 defaults).
 */

/** A single ranked search hit. */
export interface BM25Result {
  /** The docId passed to {@link BM25Index.add}. */
  docId: string;
  /** Non-negative BM25 score; higher is more relevant. */
  score: number;
}

/** Mutable in-memory BM25 index. */
export interface BM25Index {
  /** Index a document, overwriting any existing entry for the same docId. */
  add(docId: string, text: string): void;
  /** Return the topK highest-scoring documents for the query, sorted desc. */
  search(query: string, topK: number): BM25Result[];
  /** Remove a document and update all derived statistics. No-op if absent. */
  remove(docId: string): void;
  /** Reset the index to an empty state. */
  clear(): void;
  /** Current number of indexed documents. */
  readonly size: number;
}

/** Optional tuning parameters for {@link createBM25Index}. */
export interface BM25Config {
  /** Term-frequency saturation. Typical 1.2–2.0. Default 1.5. */
  k1?: number;
  /** Length normalization strength in [0,1]. Default 0.75. */
  b?: number;
}

interface DocEntry {
  /** Unique terms in this doc → term frequency. */
  tokens: string[];
  /** Number of tokens (used for length normalization). */
  length: number;
}

interface Posting {
  /** Document frequency: number of docs containing this term. */
  df: number;
  /** docId → term frequency in that doc. */
  postings: Map<string, number>;
}

const DEFAULT_K1 = 1.5;
const DEFAULT_B = 0.75;

/**
 * Tokenize text for indexing or querying.
 *
 * Lowercases, strips non-word characters, splits on whitespace, drops
 * tokens shorter than 2 chars.
 *
 * Limitation: `\w` is ASCII-only by default (`[A-Za-z0-9_]`), so accented
 * Latin characters (á, ñ, ü) and CJK are NOT stripped but are kept as
 * distinct tokens — diacritics are not normalized to their ASCII base.
 * This keeps "niño" and "nino" as separate terms. Acceptable for v1;
 * a Unicode-normalizing tokenizer (NFD + strip combining marks) is a
 * future enhancement.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(t => t.length > 1);
}

/**
 * Create a new in-memory BM25 index.
 * @param config - Optional k1/b overrides. Defaults: k1=1.5, b=0.75.
 * @returns A fresh {@link BM25Index} with no documents.
 */
export function createBM25Index(config?: BM25Config): BM25Index {
  const k1 = config?.k1 ?? DEFAULT_K1;
  const b = config?.b ?? DEFAULT_B;

  const docs = new Map<string, DocEntry>();
  const terms = new Map<string, Posting>();
  let totalLength = 0;

  function recomputeAvg(): number {
    return docs.size === 0 ? 0 : totalLength / docs.size;
  }

  function add(docId: string, text: string): void {
    if (docs.has(docId)) {
      remove(docId);
    }
    const tokens = tokenize(text);
    const entry: DocEntry = { tokens, length: tokens.length };
    docs.set(docId, entry);
    totalLength += entry.length;

    const unique = new Set(tokens);
    for (const term of unique) {
      const tf = tokens.reduce((acc, t) => (t === term ? acc + 1 : acc), 0);
      let posting = terms.get(term);
      if (!posting) {
        posting = { df: 0, postings: new Map() };
        terms.set(term, posting);
      }
      posting.df += 1;
      posting.postings.set(docId, tf);
    }
  }

  function remove(docId: string): void {
    const entry = docs.get(docId);
    if (!entry) return;
    totalLength -= entry.length;
    const unique = new Set(entry.tokens);
    for (const term of unique) {
      const posting = terms.get(term);
      if (!posting) continue;
      posting.df -= 1;
      posting.postings.delete(docId);
      if (posting.df <= 0) {
        terms.delete(term);
      }
    }
    docs.delete(docId);
  }

  function clear(): void {
    docs.clear();
    terms.clear();
    totalLength = 0;
  }

  function search(query: string, topK: number): BM25Result[] {
    const N = docs.size;
    const avg = recomputeAvg();
    if (N === 0 || avg === 0 || topK <= 0) return [];

    const queryTerms = new Set(tokenize(query));
    const scores = new Map<string, number>();

    for (const term of queryTerms) {
      const posting = terms.get(term);
      if (!posting) continue;
      const idf = Math.log((N - posting.df + 0.5) / (posting.df + 0.5) + 1);
      for (const [docId, tf] of posting.postings) {
        const doc = docs.get(docId);
        if (!doc) continue;
        const numer = tf * (k1 + 1);
        const denom = tf + k1 * (1 - b + b * (doc.length / avg));
        const contribution = idf * (numer / denom);
        scores.set(docId, (scores.get(docId) ?? 0) + contribution);
      }
    }

    const results: BM25Result[] = [];
    for (const [docId, score] of scores) {
      if (score > 0) {
        results.push({ docId, score });
      }
    }
    results.sort((a, z) => z.score - a.score);
    if (results.length > topK) results.length = topK;
    return results;
  }

  return { add, search, remove, clear, get size(): number { return docs.size; } };
}
