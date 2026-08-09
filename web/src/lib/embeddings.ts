import {
  InvalidApiKeyError,
  RateLimitError,
  ProviderUnreachableError,
  EmbeddingModelMissingError,
  classifyError,
} from './errors';
import { hashText } from './hash';

export type EmbeddingInputType = 'query' | 'passage';

export interface EmbedOptions {
  inputType?: EmbeddingInputType;
  cache?: EmbeddingCacheLike;
}

export interface EmbeddingsClient {
  embed(input: string | string[], opts?: EmbedOptions): Promise<number[][]>;
}

export interface EmbeddingsClientConfig {
  baseUrl: string;
  apiKey: string;
  embeddingModel: string;
  providerLabel?: string;
}

/**
 * Injected cache seam. The concrete `EmbeddingCache` is keyed by
 * `modelId` + a stable text hash (see lib/hash.ts — 32-bit FNV-1a).
 */
export interface EmbeddingCacheLike {
  get(modelId: string, textHash: string): number[] | undefined;
  set(modelId: string, textHash: string, embedding: number[]): void;
}

// NVIDIA NIM asymmetric retrieval models (e.g. nv-embedqa-e5-v5,
// nv-embedqa-mistral-7b-v2) reject requests without `input_type`. Symmetric
// embedding models (e.g. llama-nemotron-embed-v1, nomic-embed-text) accept either.
const ASYMMETRIC_MODEL_PATTERNS: readonly RegExp[] = [
  /nv-embedqa-e5/,
  /nv-embedqa-mistral/,
];

function isAsymmetricModel(modelId: string): boolean {
  const lower = modelId.toLowerCase();
  return ASYMMETRIC_MODEL_PATTERNS.some(p => p.test(lower));
}

/** Provider max input. Inputs over this are rejected, never silently truncated. */
const MAX_INPUT_TOKENS = 8192;
/** Hard cap on number of inputs per API request. */
const MAX_BATCH = 64;
/** Retry policy for transient (429/503) failures. */
const MAX_ATTEMPTS = 3;
const BASE_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 8000;

/** Word-count token estimate (heuristic; matches the 1.3x factor in T1's tokens.ts). */
function estimateTokens(text: string): number {
  const words = text.split(/\s+/).filter(Boolean).length;
  return Math.ceil(words * 1.3);
}

/** L2-normalize a vector in place-free form. Zero-norm → returned unchanged. */
function normalizeVector(v: number[]): number[] {
  let sumSq = 0;
  for (let i = 0; i < v.length; i += 1) sumSq += v[i] * v[i];
  const norm = Math.sqrt(sumSq);
  if (norm === 0) return [...v];
  return v.map((x) => x / norm);
}

/** Parse Retry-After (seconds) from a Headers-like object, if present. */
function parseRetryAfter(raw: unknown, fallback: number): number {
  if (raw == null) return fallback;
  const sec = Number(raw);
  return Number.isFinite(sec) && sec >= 0 ? sec * 1000 : fallback;
}

interface EmbeddingDatum {
  embedding: number[];
}

interface EmbeddingResponse {
  data?: EmbeddingDatum[];
}

type GetterLike = { get: (k: string) => string | null | undefined };
type StringRecord = Record<string, string>;
type MaybeGetter = { get?: unknown };

function hasGet(obj: object): boolean {
  const own = (obj as Record<PropertyKey, unknown>).get;
  if (typeof own === 'function') return true;
  // Headers and Map store `get` on their prototype (not own enumerable), so
  // check the inherited method — Object.entries would miss it.
  const proto = Object.getPrototypeOf(obj) as MaybeGetter | null;
  return proto !== null && typeof proto.get === 'function';
}

function isGetterLike(v: unknown): v is GetterLike {
  if (typeof v !== 'object' || v === null) return false;
  return hasGet(v);
}

function isStringRecord(v: unknown): v is StringRecord {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function readRetryAfter(headers: unknown): string | undefined {
  if (isGetterLike(headers)) return headers.get('retry-after') ?? undefined;
  if (isStringRecord(headers)) return headers['retry-after'];
  return undefined;
}

async function fetchWithRetry(
  url: string,
  headers: Record<string, string>,
  body: Record<string, unknown>,
  providerLabel: string,
): Promise<EmbeddingDatum[]> {
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt += 1) {
    let resp: Response;
    try {
      resp = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
    } catch (e) {
      throw classifyError(e, providerLabel, 'embeddings-fetch');
    }

    if (resp.ok) {
      const data = (await resp.json()) as EmbeddingResponse;
      return data.data ?? [];
    }

    const lastBody = typeof resp.text === 'function' ? await resp.text() : '';

    // Classify specific HTTP errors
    if (resp.status === 401 || resp.status === 403) {
      throw new InvalidApiKeyError(providerLabel, resp.status as 401 | 403);
    }

    if (resp.status === 429) {
      const retryAfter = resp.headers.get('retry-after');
      const retryAfterMs = retryAfter ? parseInt(retryAfter, 10) * 1000 : undefined;
      // Retry on 429
      if (attempt < MAX_ATTEMPTS) {
        const delay = Math.min(
          parseRetryAfter(readRetryAfter(resp.headers), BASE_BACKOFF_MS * attempt),
          MAX_BACKOFF_MS,
        );
        await new Promise<void>((r) => setTimeout(r, delay));
        continue;
      }
      throw new RateLimitError(providerLabel, retryAfterMs);
    }

    if (resp.status === 503) {
      // Retry on 503
      if (attempt < MAX_ATTEMPTS) {
        const delay = Math.min(
          parseRetryAfter(readRetryAfter(resp.headers), BASE_BACKOFF_MS * attempt),
          MAX_BACKOFF_MS,
        );
        await new Promise<void>((r) => setTimeout(r, delay));
        continue;
      }
      throw new ProviderUnreachableError(providerLabel, new Error(`${resp.status} ${resp.statusText}: ${lastBody}`), {
        retryable: true,
      });
    }

    if (resp.status >= 500) {
      throw new ProviderUnreachableError(providerLabel, new Error(`${resp.status} ${resp.statusText}: ${lastBody}`), {
        retryable: true,
      });
    }

    // Non-retryable errors (400, 404, etc.)
    throw new ProviderUnreachableError(providerLabel, new Error(`${resp.status} ${resp.statusText}: ${lastBody}`), {
      retryable: false,
    });
  }
  // Should never reach here
  throw new ProviderUnreachableError(providerLabel, new Error('Exhausted retry attempts'));
}

/**
 * Create an embeddings client for OpenAI-compatible embeddings API.
 * Hardened with batch splitting (MAX_BATCH, aggregate-token budget), retry
 * with exponential backoff on 429/503 honoring Retry-After, input length cap
 * (MAX_INPUT_TOKENS), dimension validation, L2 normalization, and optional
 * injectable cache.
 * Returns are L2-normalized; zero-norm vectors pass through unchanged.
 */
export function createEmbeddingsClient(config: EmbeddingsClientConfig): EmbeddingsClient {
  const baseUrl = config.baseUrl.replace(/\/+$/, '');
  const apiKey = config.apiKey;
  const embeddingModel = config.embeddingModel;
  const providerLabel = config.providerLabel ?? 'provider';

  // Greedy chunking: pack up to MAX_BATCH inputs whose aggregate estimated
  // tokens fit MAX_INPUT_TOKENS. Since per-input length is validated upstream,
  // this only shrinks a batch when many medium inputs jointly exceed the limit.
  function splitIntoBatches(inputs: string[]): string[][] {
    const batches: string[][] = [];
    let current: string[] = [];
    let currentTokens = 0;
    for (const input of inputs) {
      const tokens = estimateTokens(input);
      const wouldExceedBatch = current.length >= MAX_BATCH;
      const wouldExceedTokens = currentTokens + tokens > MAX_INPUT_TOKENS;
      if (current.length > 0 && (wouldExceedBatch || wouldExceedTokens)) {
        batches.push(current);
        current = [];
        currentTokens = 0;
      }
      current.push(input);
      currentTokens += tokens;
    }
    if (current.length > 0) batches.push(current);
    return batches;
  }

  async function embed(input: string | string[], opts?: EmbedOptions): Promise<number[][]> {
    if (!baseUrl) {
      throw new ProviderUnreachableError(providerLabel, undefined, { isTimeout: false });
    }
    if (!embeddingModel) {
      throw new EmbeddingModelMissingError();
    }

    const inputs = Array.isArray(input) ? input : [input];

    // Per-input length cap: reject oversized inputs rather than silently truncate.
    for (const text of inputs) {
      const tokens = estimateTokens(text);
      if (tokens > MAX_INPUT_TOKENS) {
        throw new ProviderUnreachableError(providerLabel, new Error(`Input exceeds ${MAX_INPUT_TOKENS} token limit`), {
          message: `Input exceeds ${MAX_INPUT_TOKENS} token limit (estimated ${tokens}). Split before embedding.`,
          retryable: false,
        });
      }
    }

    // Cache lookup + remaining-uncached partition.
    const cache = opts?.cache;
    const results: number[][] = [];
    const toFetch: string[] = [];
    const fetchIdx: number[] = [];
    for (let i = 0; i < inputs.length; i += 1) {
      const text = inputs[i];
      // Cache holds already-normalized vectors; return as-is to avoid FP drift.
      const cached = cache?.get(embeddingModel, hashText(text));
      if (cached !== undefined) {
        results[i] = cached;
      } else {
        toFetch.push(text);
        fetchIdx.push(i);
      }
    }

    if (toFetch.length === 0) {
      return results;
    }

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    const inputType = isAsymmetricModel(embeddingModel) ? opts?.inputType ?? 'passage' : undefined;

    // Batch + dispatch each chunk via the retrying fetch helper.
    const batches = splitIntoBatches(toFetch);
    const allFetched: number[][] = [];
    for (const batch of batches) {
      const body: Record<string, unknown> = { model: embeddingModel, input: batch };
      if (inputType !== undefined) body.input_type = inputType;
      const data = await fetchWithRetry(`${baseUrl}/embeddings`, headers, body, providerLabel);

      // Dimension validation: all embeddings in a response must share one length.
      let dim = -1;
      const batchVectors = data.map((d) => {
        const vec = d.embedding;
        if (dim === -1) dim = vec.length;
        if (vec.length !== dim) {
          throw new ProviderUnreachableError(
            providerLabel,
            new Error(`Embedding dimension mismatch: expected ${dim}, got ${vec.length}`),
            { retryable: false }
          );
        }
        return normalizeVector(vec);
      });
      allFetched.push(...batchVectors);
    }

    // Write fetched vectors back to cache and into the result slots.
    for (let j = 0; j < allFetched.length; j += 1) {
      const text = toFetch[j];
      const vec = allFetched[j];
      cache?.set(embeddingModel, hashText(text), vec);
      results[fetchIdx[j]] = vec;
    }

    return results;
  }

  return { embed };
}