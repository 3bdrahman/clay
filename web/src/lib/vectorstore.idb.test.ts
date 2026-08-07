import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { _resetWriteQueue } from './vectorstore';
import { createVectorStore } from './vectorstore';
import { createBM25Index } from './bm25';
import type { EmbeddingsClient } from './embeddings';
import { EmbeddingsConfigError } from './embeddings';
import type { Document } from './types';

interface FakeStore { _data: Map<string, unknown>; keyPath: string; }
interface FakeIndexReq { onsuccess: ((cb: () => void) => void) | null; result: unknown; }
interface FakeGetAllReq { onsuccess: ((cb: () => void) => void) | null; result: unknown; }

function installFakeIDB(): void {
  const dbs = new Map<string, Map<string, FakeStore>>();
  (globalThis as Record<string, unknown>).indexedDB = {
    open(name: string) {
      const reqObj = {
        result: undefined as unknown as {
          objectStoreNames: { contains(n: string): boolean };
          createObjectStore(n: string, opts: { keyPath: string }): FakeStore;
          transaction(s: string): { objectStore(n: string): unknown; oncomplete: (() => void) | null };
          close(): void;
        },
        onupgradeneeded: null as ((e: Event) => void) | null,
        onsuccess: null as ((e: Event) => void) | null,
        onerror: null as ((e: Event) => void) | null,
        onblocked: null as ((e: Event) => void) | null,
      };
      queueMicrotask(() => {
        let stores = dbs.get(name);
        if (!stores) { stores = new Map(); dbs.set(name, stores); }
        const dbObj = {
          objectStoreNames: { contains: (n: string) => stores!.has(n) },
          createObjectStore(n: string, opts: { keyPath: string }) {
            const s: FakeStore = { _data: new Map(), keyPath: opts.keyPath };
            stores!.set(n, s);
            return {
              ...s,
              createIndex(name: string, keyPath: string) { return { name, keyPath }; },
            } as unknown as FakeStore & { createIndex(name: string, keyPath: string): unknown };
          },
          transaction(stores_: string) {
            const target = stores!.get(stores_);
            if (!target) throw new Error(`store ${stores_} missing`);
            const storeApi = {
              put(v: unknown) {
                const key = String((v as Record<string, unknown>)[target.keyPath]);
                target._data.set(key, v);
                const r = { onsuccess: null as (() => void) | null };
                queueMicrotask(() => r.onsuccess?.());
                return r;
              },
              getAll() {
                const r: FakeGetAllReq = { onsuccess: null, result: Array.from(target._data.values()) };
                queueMicrotask(() => {
                  if (r.onsuccess) (r.onsuccess as (cb: () => void) => void)(() => undefined);
                });
                return r;
              },
              delete(k: string) {
                target._data.delete(k);
                const r = { onsuccess: null as (() => void) | null };
                queueMicrotask(() => r.onsuccess?.());
                return r;
              },
              clear() {
                target._data.clear();
                const r = { onsuccess: null as (() => void) | null };
                queueMicrotask(() => r.onsuccess?.());
                return r;
              },
              index(n: string) {
                return {
                  getAllKeys(q: unknown) {
                    const keys: IDBValidKey[] = [];
                    for (const [k, v] of target._data) {
                      if (String((v as Record<string, unknown>)[n]) === String(q)) keys.push(k);
                    }
                    const r: FakeIndexReq = { onsuccess: null, result: keys };
                    queueMicrotask(() => {
                      if (r.onsuccess) (r.onsuccess as (cb: () => void) => void)(() => undefined);
                    });
                    return r;
                  },
                };
              },
            };
            const txn = {
              objectStore(_n: string) { return storeApi; },
              oncomplete: null as (() => void) | null,
            };
            queueMicrotask(() => txn.oncomplete?.());
            return txn;
          },
          close() { /* noop */ },
        };
        reqObj.result = dbObj;
        if (reqObj.onupgradeneeded) reqObj.onupgradeneeded(new Event('upgradeneeded'));
        if (reqObj.onsuccess) reqObj.onsuccess(new Event('success'));
      });
      return reqObj;
    },
  };
}

function makeMockEmbeddings(dim = 4): EmbeddingsClient {
  return {
    embed: vi.fn(async (input: string | string[]) => {
      const arr = Array.isArray(input) ? input : [input];
      return arr.map(() => new Array(dim).fill(0).map(() => Math.random() - 0.5));
    }),
  };
}

describe('vectorstore IDB-specific features', () => {
  beforeEach(() => {
    localStorage.clear();
    installFakeIDB();
    _resetWriteQueue();
  });
  afterEach(() => {
    (globalThis as Record<string, unknown>).indexedDB = undefined;
  });

  it('migrates localStorage v1 cache to IDB on first load', async () => {
    localStorage.setItem(
      'clay-vector-entries-v1',
      JSON.stringify([
        { id: '1', text: 'alpha', source: 'a.txt', embedding: [0.1, 0.2, 0.3, 0.4] },
        { id: '2', text: 'beta', source: 'b.txt', embedding: [0.5, 0.6, 0.7, 0.8] },
        { id: '3', text: 'gamma', source: 'c.txt', embedding: [0.9, 0.1, 0.2, 0.3] },
      ]),
    );
    const vs = createVectorStore(makeMockEmbeddings(4));
    await vs.load();
    expect(localStorage.getItem('clay-vector-entries-v1')).toBeNull();
    expect(vs.stats.entries).toBe(3);
  });

  it('migration is idempotent — second load keeps count', async () => {
    localStorage.setItem(
      'clay-vector-entries-v1',
      JSON.stringify([{ id: '1', text: 'a', source: 's', embedding: [0.1, 0.2, 0.3, 0.4] }]),
    );
    const vs = createVectorStore(makeMockEmbeddings(4));
    await vs.load();
    expect(vs.stats.entries).toBe(1);
    await vs.load();
    expect(vs.stats.entries).toBe(1);
  });

  it('score threshold filters low-cosine results', async () => {
    const emb = makeMockEmbeddings(4);
    const vs = createVectorStore(emb, { scoreThreshold: 0.99 });
    await vs.load();
    vs.addEntries([
      { id: '1', text: 'a', source: 's', embedding: [1, 0, 0, 0] },
      { id: '2', text: 'b', source: 's', embedding: [0, 1, 0, 0] },
    ]);
    const embTyped = emb as { embed: ReturnType<typeof vi.fn> };
    embTyped.embed.mockResolvedValueOnce([[1, 0, 0, 0]]);
    const results = await vs.similaritySearch('q', 5);
    expect(results.length).toBe(1);
    expect(results[0].id).toBe('1');
  });

  it('MMR diversifies — near-duplicate loses to orthogonal entry', async () => {
    const emb = makeMockEmbeddings(4);
    const vs = createVectorStore(emb, { useMMR: true, mmrLambda: 0.3, topK: 2 });
    await vs.load();
    vs.addEntries([
      { id: 'near', text: 'a', source: 's', embedding: [1, 0, 0, 0] },
      { id: 'dup', text: 'b', source: 's', embedding: [0.99, 0.1, 0, 0] },
      { id: 'orth', text: 'c', source: 's', embedding: [0, 0, 1, 0] },
    ]);
    const embTyped = emb as { embed: ReturnType<typeof vi.fn> };
    embTyped.embed.mockResolvedValueOnce([[1, 0, 0, 0]]);
    const results = await vs.similaritySearch('q', 2);
    expect(results.length).toBe(2);
    expect(results[0].id).toBe('near');
    expect(results[1].id).toBe('orth');
  });

  it('hybrid BM25+dense surfaces BM25-only-relevant chunk', async () => {
    const emb = makeMockEmbeddings(4);
    const bm25 = createBM25Index();
    bm25.add('kw1', 'typescript react javascript hooks');
    const vs = createVectorStore(emb, { useHybrid: true, hybridAlpha: 0.5, bm25Index: bm25, topK: 3 });
    await vs.load();
    vs.addEntries([
      { id: 'kw1', text: 'typescript react javascript hooks', source: 's', embedding: [0, 0, 0, 1] },
      { id: 'unrelated', text: 'quantum physics', source: 's', embedding: [1, 0, 0, 0] },
    ]);
    const embTyped = emb as { embed: ReturnType<typeof vi.fn> };
    embTyped.embed.mockResolvedValueOnce([[1, 0, 0, 0]]);
    const results = await vs.similaritySearch('javascript', 2);
    expect(results.map((r: Document) => r.id)).toContain('kw1');
  });

  it('removeBySource returns synchronous count', async () => {
    const vs = createVectorStore(makeMockEmbeddings(4));
    await vs.load();
    vs.addEntries([
      { id: '1', text: 'a', source: 'd1', embedding: [0.1, 0, 0, 0] },
      { id: '2', text: 'b', source: 'd2', embedding: [0.1, 0, 0, 0] },
      { id: '3', text: 'c', source: 'd1', embedding: [0.1, 0, 0, 0] },
    ]);
    const removed = vs.removeBySource('d1');
    expect(removed).toBe(2);
    expect(vs.stats.entries).toBe(1);
  });

  it('dimension validation throws EmbeddingsConfigError', async () => {
    const vs = createVectorStore(makeMockEmbeddings(4));
    await vs.load();
    vs.addEntries([{ id: '1', text: 'a', source: 's', embedding: [0.1, 0.2, 0.3, 0.4] }]);
    expect(() =>
      vs.addEntries([{ id: '2', text: 'b', source: 's', embedding: [0.1, 0.2, 0.3] }]),
    ).toThrow(EmbeddingsConfigError);
  });
});

describe('vectorstore in-memory fallback when IDB unavailable', () => {
  beforeEach(() => {
    localStorage.clear();
    (globalThis as Record<string, unknown>).indexedDB = undefined;
    _resetWriteQueue();
  });

  it('warns once and still works in-memory', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const emb = makeMockEmbeddings(4);
    const vs = createVectorStore(emb);
    await vs.load();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    vs.addEntries([{ id: '1', text: 'a', source: 's', embedding: [0.1, 0, 0, 0] }]);
    const embTyped = emb as { embed: ReturnType<typeof vi.fn> };
    embTyped.embed.mockResolvedValueOnce([[0.1, 0, 0, 0]]);
    const results = await vs.similaritySearch('q', 1);
    expect(results.length).toBe(1);
    warnSpy.mockRestore();
  });
});
