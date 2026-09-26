// embeddings — local, in-browser embeddings via transformers.js.
//
// The client wraps the embedding worker and adds an internal LRU cache keyed
// by model id + a stable text hash (see lib/hash.ts — 32-bit FNV-1a), so
// repeated queries and re-indexed documents skip inference. The interface is
// one method, one parameter: embed(input) → L2-normalized vectors.

import { createEmbeddingCache, type EmbeddingCache } from './embeddingCache';
import { hashText } from './hash';
import { EMBEDDING_MODEL_ID } from './embeddingModel';

export { EMBEDDING_MODEL_ID };

export interface EmbeddingsClient {
  embed(input: string | string[]): Promise<number[][]>;
}

export interface EmbedWorkerRequest {
  type: 'embed';
  id: number;
  texts: string[];
}

export type EmbedWorkerResponse =
  | { type: 'result'; id: number; embeddings: number[][] }
  | { type: 'error'; id: number; message: string };

export type EmbeddingWorkerLike = {
  postMessage(msg: EmbedWorkerRequest): void;
  addEventListener(type: 'message', listener: (ev: MessageEvent<EmbedWorkerResponse>) => void): void;
};

function defaultWorkerFactory(): Worker {
  return new Worker(new URL('../workers/embeddingWorker.ts', import.meta.url), { type: 'module' });
}

/**
 * Create the local embeddings client. Accepts an injectable worker factory
 * and cache for tests; production uses the bundled worker and a fresh LRU.
 */
export function createEmbeddingsClient(options?: {
  workerFactory?: () => EmbeddingWorkerLike;
  cache?: EmbeddingCache;
}): EmbeddingsClient {
  const worker: EmbeddingWorkerLike = options?.workerFactory
    ? options.workerFactory()
    : defaultWorkerFactory();
  const cache = options?.cache ?? createEmbeddingCache();

  let nextId = 1;
  const pending = new Map<number, { resolve: (v: number[][]) => void; reject: (e: Error) => void }>();

  function isEmbedResponse(v: unknown): v is EmbedWorkerResponse {
    if (typeof v !== 'object' || v === null) return false;
    const r = v as Record<string, unknown>;
    if (r.type === 'result') return typeof r.id === 'number' && Array.isArray(r.embeddings);
    if (r.type === 'error') return typeof r.id === 'number' && typeof r.message === 'string';
    return false;
  }

  worker.addEventListener('message', (ev: MessageEvent) => {
    const data: unknown = ev.data;
    if (!isEmbedResponse(data)) return;
    const msg = data;
    if (msg.type === 'result') {
      const entry = pending.get(msg.id);
      if (entry) {
        pending.delete(msg.id);
        entry.resolve(msg.embeddings);
      }
    } else if (msg.type === 'error') {
      const entry = pending.get(msg.id);
      if (entry) {
        pending.delete(msg.id);
        entry.reject(new Error(msg.message));
      }
    }
  });

  async function embed(input: string | string[]): Promise<number[][]> {
    const inputs = Array.isArray(input) ? input : [input];
    if (inputs.length === 0) return [];

    const results: number[][] = new Array(inputs.length);
    const toEmbed: string[] = [];
    const toEmbedIdx: number[] = [];
    for (let i = 0; i < inputs.length; i += 1) {
      const cached = cache.get(EMBEDDING_MODEL_ID, hashText(inputs[i]));
      if (cached !== undefined) {
        results[i] = cached;
      } else {
        toEmbed.push(inputs[i]);
        toEmbedIdx.push(i);
      }
    }

    if (toEmbed.length > 0) {
      const id = nextId;
      nextId += 1;
      const vectors = await new Promise<number[][]>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ type: 'embed', id, texts: toEmbed });
      });
      for (let j = 0; j < toEmbed.length; j += 1) {
        const vec = vectors[j] ?? [];
        cache.set(EMBEDDING_MODEL_ID, hashText(toEmbed[j]), vec);
        results[toEmbedIdx[j]] = vec;
      }
    }

    return results;
  }

  return { embed };
}
