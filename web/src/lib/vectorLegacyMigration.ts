import type { ChunkMetadata } from './types';
import { estimateTokens } from './tokens';

const LEGACY_KEY = 'clay-vector-entries-v1';

/**
 * Sentinel model ID stamped on entries migrated from the legacy localStorage
 * store AND on entries added via a VectorStore whose config omitted
 * `embeddingModel`. The service bundle (clayServices.ts) always passes
 * the picked embedding model; this sentinel exists so model-mismatch
 * detection (`entry.metadata.modelId !== currentEmbeddingModel`) treats
 * legacy/unknown entries as "needs re-embedding" instead of incorrectly
 * matching whatever string we picked. The literal `'legacy'` is intentionally
 * distinct from any real provider model id (which always contain a vendor
 * prefix or namespace separator).
 */
export const LEGACY_EMBEDDING_MODEL_ID = 'legacy';

interface VectorEntry {
  id: string;
  text: string;
  embedding: Float32Array;
  metadata: ChunkMetadata;
}

function isFiniteNumberArray(vals: unknown[]): vals is number[] {
  return vals.every((n) => typeof n === 'number' && Number.isFinite(n));
}

function coerceLegacyEmbedding(v: unknown): number[] | null {
  const candidate: unknown[] | null = Array.isArray(v)
    ? v
    : v && typeof v === 'object'
      ? Object.values(v as Record<string, unknown>)
      : null;
  if (!candidate || !isFiniteNumberArray(candidate)) return null;
  // An all-zero vector scores 0 against every query, so a migrated entry
  // could never match anything. Reject it like any other invalid embedding;
  // when no valid entries remain, readLegacy's zero-valid-entries path clears
  // the legacy key instead of re-attempting migration on every load.
  if (candidate.every((n) => n === 0)) return null;
  return candidate;
}

export function readLegacy(embeddingModel: string): VectorEntry[] | null {
  try {
    const raw = localStorage.getItem(LEGACY_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const modelId = embeddingModel || LEGACY_EMBEDDING_MODEL_ID;
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
        embedding: Float32Array.from(emb),
        metadata: {
          source: r.source,
          sourceHash: '',
          page: typeof r.page === 'number' ? r.page : undefined,
          charStart: 0,
          charEnd: r.text.length,
          chunkIndex: i,
          tokenCount: estimateTokens(r.text),
          modelId,
          updatedAt: Date.now(),
        },
      });
    });
    return out;
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[vectorstore] readLegacy: localStorage parse failed (treating as no migration):', e);
    }
    return null;
  }
}

export function clearLegacyKey(): void {
  try {
    localStorage.removeItem(LEGACY_KEY);
  } catch (e) {
    // localStorage may be disabled (private mode) or the key may be denied
    // by storage isolation. We're clearing anyway; suppressing is correct.
    if (import.meta.env.DEV) {
      console.warn('[vectorstore] clearLegacyKey: localStorage.removeItem failed:', e);
    }
  }
}

export { LEGACY_KEY };