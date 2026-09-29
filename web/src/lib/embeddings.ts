// embeddings — local, in-browser embeddings via transformers.js.
//
// The client wraps the embedding worker and adds an internal LRU cache keyed
// by model id + a stable text hash (see lib/hash.ts — 32-bit FNV-1a), so
// repeated queries and re-indexed documents skip inference. The interface is
// one method, one parameter: embed(input) → L2-normalized vectors.

import { createEmbeddingCache, getSharedEmbeddingCache, type EmbeddingCache } from './embeddingCache';
import { hashText } from './hash';
import { EMBEDDING_MODEL_ID } from './embeddingModel';

export { EMBEDDING_MODEL_ID, getSharedEmbeddingCache };

export interface EmbeddingsClient {
  embed(input: string | string[]): Promise<number[][]>;
  /**
   * Terminate the underlying worker once pending embeds drain. A no-op when
   * no worker was ever spawned. The service bundle calls this when it is
   * replaced so recreated clients do not leak workers.
   */
  dispose?(): void;
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
  addEventListener(type: 'error', listener: (ev: ErrorEvent) => void): void;
  /** Optional: real workers expose terminate(); test fakes legitimately omit it. */
  terminate?(): void;
};

function defaultWorkerFactory(): Worker {
  return new Worker(new URL('../workers/embeddingWorker.ts', import.meta.url), { type: 'module' });
}

/**
 * Create the local embeddings client. Accepts an injectable worker factory
 * and cache for tests; production uses the bundled worker and a fresh LRU.
 * The worker spawns lazily on the first embed. A worker 'error' event (script
 * failed to load) rejects pending embeds and clears the worker so the next
 * embed spawns a fresh one instead of hanging or respawning in a loop.
 */
export function createEmbeddingsClient(options?: {
  workerFactory?: () => EmbeddingWorkerLike;
  cache?: EmbeddingCache;
}): EmbeddingsClient {
  const cache = options?.cache ?? createEmbeddingCache();

  let nextId = 1;
  const pending = new Map<number, { resolve: (v: number[][]) => void; reject: (e: Error) => void }>();
  let worker: EmbeddingWorkerLike | null = null;
  let doomed = false;

  function ensureWorker(): EmbeddingWorkerLike {
    if (worker === null) {
      doomed = false;
      worker = spawn();
    }
    return worker;
  }

  function terminateIfDrained(): void {
    if (!doomed || pending.size > 0 || worker === null) return;
    const w = worker;
    worker = null;
    doomed = false;
    w.terminate?.();
  }

  function isEmbedResponse(v: unknown): v is EmbedWorkerResponse {
    if (typeof v !== 'object' || v === null) return false;
    const r = v as Record<string, unknown>;
    if (r.type === 'result') return typeof r.id === 'number' && Array.isArray(r.embeddings);
    if (r.type === 'error') return typeof r.id === 'number' && typeof r.message === 'string';
    return false;
  }

  function spawn(): EmbeddingWorkerLike {
    const w: EmbeddingWorkerLike = options?.workerFactory
      ? options.workerFactory()
      : defaultWorkerFactory();

    w.addEventListener('message', (ev: MessageEvent) => {
      const data: unknown = ev.data;
      if (!isEmbedResponse(data)) return;
      const msg = data;
      const entry = pending.get(msg.id);
      if (!entry) return;
      pending.delete(msg.id);
      if (msg.type === 'result') {
        entry.resolve(msg.embeddings);
      } else {
        entry.reject(new Error(msg.message));
      }
      terminateIfDrained();
    });

    // The worker 'error' event fires when the script fails to load — no
    // message is posted, so without this listener every pending embed hangs.
    // The worker is cleared, not respawned here: respawning inside the error
    // handler would loop endlessly when the script keeps failing to load
    // (e.g. an asset 404 under a wrong base path). The next embed spawns.
    w.addEventListener('error', (ev: ErrorEvent) => {
      const message = ev.message || 'embedding worker failed';
      for (const entry of pending.values()) entry.reject(new Error(message));
      pending.clear();
      worker = null;
    });

    return w;
  }

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
        ensureWorker().postMessage({ type: 'embed', id, texts: toEmbed });
      });
      for (let j = 0; j < toEmbed.length; j += 1) {
        const vec = vectors[j];
        if (!Array.isArray(vec) || vec.length === 0 || !vec.every((n) => Number.isFinite(n))) {
          throw new Error(`embedding worker returned an invalid vector for text ${j}`);
        }
        cache.set(EMBEDDING_MODEL_ID, hashText(toEmbed[j]), vec);
        results[toEmbedIdx[j]] = vec;
      }
    }

    return results;
  }

  function dispose(): void {
    doomed = true;
    terminateIfDrained();
  }

  return { embed, dispose };
}
