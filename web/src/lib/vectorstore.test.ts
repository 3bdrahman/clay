import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createVectorStore, _resetWriteQueue } from './vectorstore';
import type { EmbeddingsClient } from './embeddings';

const mockEmbeddings: EmbeddingsClient = {
  embed: vi.fn(async (input: string | string[]) => {
    const arr = Array.isArray(input) ? input : [input];
    return arr.map(() => new Array(1024).fill(0).map(() => Math.random()));
  }),
};

// Minimal inline fake IDB so the legacy test file persists across loads via IDB.
interface FakeStore { _data: Map<string, unknown>; _indexes: Map<string, string>; keyPath: string; }
function installFakeIDB(): void {
  const dbs = new Map<string, Map<string, FakeStore>>();
  (globalThis as Record<string, unknown>).indexedDB = {
    open(name: string) {
      const reqObj = {
        result: undefined as unknown as { objectStoreNames: { contains(n: string): boolean }; createObjectStore(n: string, opts: { keyPath: string }): FakeStore; transaction(s: string, mode: string): { objectStore(n: string): { put(v: unknown): { onsuccess: (() => void) | null }; getAll(): { onsuccess: ((cb: () => void) => void) | null; result: unknown }; delete(k: string): { onsuccess: (() => void) | null }; clear(): { onsuccess: (() => void) | null }; index(n: string): { getAllKeys(q: unknown): { onsuccess: ((cb: () => void) => void) | null; result: unknown } } }; oncomplete: (() => void) | null; onerror: (() => void) | null; onabort: (() => void) | null; error: Error | null }; createObjectStore(n: string, opts: { keyPath: string }): FakeStore; close(): void },
        onupgradeneeded: null as ((e: Event) => void) | null,
        onsuccess: null as ((e: Event) => void) | null,
        onerror: null as ((e: Event) => void) | null,
        onblocked: null as ((e: Event) => void) | null,
      };
      queueMicrotask(() => {
        let stores = dbs.get(name);
        if (!stores) {
          stores = new Map();
          dbs.set(name, stores);
        }
        const db = {
          _stores: stores,
          objectStoreNames: { contains: (n: string) => stores!.has(n) },
          createObjectStore(n: string, opts: { keyPath: string }) {
            const s: FakeStore = { _data: new Map(), _indexes: new Map(), keyPath: opts.keyPath };
            stores!.set(n, s);
            return {
              ...s,
              createIndex(name: string, keyPath: string) {
                s._indexes.set(name, keyPath);
                return { name, keyPath };
              },
            } as unknown as FakeStore & { createIndex(name: string, keyPath: string): unknown };
          },
          transaction(stores_: string, _mode: string) {
            const s = stores!.get(stores_);
            if (!s) throw new Error(`store ${stores_} missing`);
            return {
              _stores: stores!,
              objectStore(st: string) {
                const target = stores!.get(st);
                if (!target) throw new Error(`store ${st} missing`);
                return {
                  put(v: unknown) {
                    const key = String((v as Record<string, unknown>)[target.keyPath]);
                    target._data.set(key, v);
                    const r: { onsuccess: ((this: IDBRequest, ev: Event) => void) | null; result: IDBValidKey } = { onsuccess: null, result: key };
                    queueMicrotask(() => r.onsuccess?.call(r as unknown as IDBRequest, new Event('success')));
                    return r;
                  },
                  getAll() {
                    const r: { onsuccess: ((this: IDBRequest, ev: Event) => void) | null; result: unknown } = { onsuccess: null, result: Array.from(target._data.values()) };
                    queueMicrotask(() => r.onsuccess?.call(r as unknown as IDBRequest, new Event('success')));
                    return r;
                  },
                  delete(k: string) {
                    target._data.delete(k);
                    const r: { onsuccess: ((this: IDBRequest, ev: Event) => void) | null; result: undefined } = { onsuccess: null, result: undefined };
                    queueMicrotask(() => r.onsuccess?.call(r as unknown as IDBRequest, new Event('success')));
                    return r;
                  },
                  clear() {
                    target._data.clear();
                    const r: { onsuccess: ((this: IDBRequest, ev: Event) => void) | null; result: undefined } = { onsuccess: null, result: undefined };
                    queueMicrotask(() => r.onsuccess?.call(r as unknown as IDBRequest, new Event('success')));
                    return r;
                  },
                  index(n: string) {
                    return {
                      getAllKeys(q: unknown) {
                        const keys: IDBValidKey[] = [];
                        const keyPath = (target as FakeStore)._indexes.get(n) ?? n;
                        const path = keyPath.split('.');
                        for (const [k, v] of target._data) {
                          let cur: unknown = v;
                          for (const p of path) {
                            if (cur && typeof cur === 'object') cur = (cur as Record<string, unknown>)[p];
                            else { cur = undefined; break; }
                          }
                          if (String(cur) === String(q)) keys.push(k);
                        }
                        const r: { onsuccess: ((this: IDBRequest, ev: Event) => void) | null; result: IDBValidKey[] } = { onsuccess: null, result: keys };
                        queueMicrotask(() => r.onsuccess?.call(r as unknown as IDBRequest, new Event('success')));
                        return r;
                      },
                    };
                  },
                };
              },
              oncomplete: null as (() => void) | null,
              onerror: null as (() => void) | null,
              onabort: null as (() => void) | null,
              error: null as Error | null,
            };
          },
          close() { /* noop */ },
        };
        reqObj.result = db;
        if (reqObj.onupgradeneeded) {
          reqObj.onupgradeneeded(new Event('upgradeneeded'));
        }
        if (reqObj.onsuccess) reqObj.onsuccess(new Event('success'));
      });
      return reqObj;
    },
  };
}

describe('createVectorStore', () => {
  let vectorstore: ReturnType<typeof createVectorStore>;

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    installFakeIDB();
    _resetWriteQueue();
    vectorstore = createVectorStore(mockEmbeddings);
  });

  afterEach(() => {
    (globalThis as Record<string, unknown>).indexedDB = undefined;
  });

  it('starts empty', async () => {
    await vectorstore.load();
    expect(vectorstore.stats.entries).toBe(0);
  });

  it('adds entries and caches to localStorage', async () => {
    await vectorstore.load();
    const entries = [
      { id: '1', text: 'hello world', source: 'test.txt', embedding: new Array(1024).fill(0.1) },
      { id: '2', text: 'foo bar', source: 'test.txt', embedding: new Array(1024).fill(0.2) },
    ];
    vectorstore.addEntries(entries);
    expect(vectorstore.stats.entries).toBe(2);

    await new Promise((r) => setTimeout(r, 10));
    expect(localStorage.getItem('clay-vector-entries-v1')).toBeNull();
  });

  it('persists across load calls', async () => {
    await vectorstore.load();
    vectorstore.addEntries([
      { id: '1', text: 'persistent', source: 'test.txt', embedding: new Array(1024).fill(0.5) },
    ]);

    // Create new vectorstore instance (simulates reload)
    const vs2 = createVectorStore(mockEmbeddings);
    await vs2.load();
    expect(vs2.stats.entries).toBe(1);
  });

  it('similaritySearch returns empty when no entries', async () => {
    await vectorstore.load();
    const results = await vectorstore.similaritySearch('query', 4);
    expect(results).toEqual([]);
  });

  it('similaritySearch returns top-K by cosine similarity', async () => {
    await vectorstore.load();
    const base = new Array(1024).fill(0);
    base[0] = 1;
    const orth = new Array(1024).fill(0);
    orth[1] = 1;

    vectorstore.addEntries([
      { id: '1', text: 'parallel', source: 'a', embedding: base },
      { id: '2', text: 'orthogonal', source: 'b', embedding: orth },
    ]);

    // Query parallel to first entry
    (mockEmbeddings.embed as ReturnType<typeof vi.fn>).mockResolvedValueOnce([base]);
    const results = await vectorstore.similaritySearch('query', 2);
    expect(results).toHaveLength(2);
    expect(results[0].id).toBe('1');
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });

  it('removeBySource removes entries and updates cache', async () => {
    await vectorstore.load();
    vectorstore.addEntries([
      { id: '1', text: 'a', source: 'doc1', embedding: new Array(1024).fill(0.1) },
      { id: '2', text: 'b', source: 'doc2', embedding: new Array(1024).fill(0.2) },
      { id: '3', text: 'c', source: 'doc1', embedding: new Array(1024).fill(0.3) },
    ]);
    expect(vectorstore.stats.entries).toBe(3);

    const removed = vectorstore.removeBySource('doc1');
    expect(removed).toBe(2);
    expect(vectorstore.stats.entries).toBe(1);

    const vs2 = createVectorStore(mockEmbeddings);
    await vs2.load();
    expect(vs2.stats.entries).toBe(1);
  });

  it('clear removes all entries and clears cache', async () => {
    await vectorstore.load();
    vectorstore.addEntries([
      { id: '1', text: 'a', source: 'doc1', embedding: new Array(1024).fill(0.1) },
    ]);
    expect(vectorstore.stats.entries).toBe(1);

    vectorstore.clear();
    expect(vectorstore.stats.entries).toBe(0);
    expect(localStorage.getItem('clay-vector-entries-v1')).toBeNull();
  });

  it('deduplicates by id on addEntries', async () => {
    await vectorstore.load();
    const entry = { id: 'dup', text: 'x', source: 's', embedding: new Array(1024).fill(0.1) };
    vectorstore.addEntries([entry]);
    vectorstore.addEntries([entry]);
    expect(vectorstore.stats.entries).toBe(1);
  });

  it('handles localStorage corruption gracefully', async () => {
    localStorage.setItem('clay-vector-entries-v1', 'not json');
    await vectorstore.load();
    expect(vectorstore.stats.entries).toBe(0);
  });

  it('skips entries with invalid embedding arrays', async () => {
    localStorage.setItem(
      'clay-vector-entries-v1',
      JSON.stringify([
        { id: '1', text: 'a', source: 's', embedding: [NaN, 0.2] },
        { id: '2', text: 'b', source: 's', embedding: [Infinity, 0.2] },
        { id: '3', text: 'c', source: 's', embedding: [null, 0.2] },
        { id: '4', text: 'd', source: 's', embedding: 'not-an-array' },
      ]),
    );
    await vectorstore.load();
    expect(vectorstore.stats.entries).toBe(0);
  });

  it('skips entries with missing required fields', async () => {
    localStorage.setItem(
      'clay-vector-entries-v1',
      JSON.stringify([
        { id: '1', text: 'a', embedding: [0.1] }, // missing source
        { text: 'b', source: 's', embedding: [0.1] }, // missing id
        { id: '3', source: 's', embedding: [0.1] }, // missing text
        null,
        'string-entry',
      ]),
    );
    await vectorstore.load();
    expect(vectorstore.stats.entries).toBe(0);
  });

  it('preserves numeric page field when present', async () => {
    const entries = [
      {
        id: '1',
        text: 'p1',
        source: 'a.pdf',
        page: 3,
        embedding: new Array(4).fill(0.1),
      },
    ];
    vectorstore.addEntries(entries);
    const vs2 = createVectorStore(mockEmbeddings);
    await vs2.load();
    expect(vs2.stats.entries).toBe(1);
    const results = await vs2.similaritySearch('x', 1);
    expect(results[0].page).toBe(3);
  });

  it('accepts legacy cache where embedding was stored as a sparse object', async () => {
    const obj: Record<string, number> = {};
    obj['0'] = 0.1;
    obj['1'] = 0.2;
    obj['2'] = 0.3;
    localStorage.setItem(
      'clay-vector-entries-v1',
      JSON.stringify([{ id: '1', text: 'a', source: 's', embedding: obj }]),
    );
    const vs2 = createVectorStore(mockEmbeddings);
    await vs2.load();
    expect(vs2.stats.entries).toBe(1);
  });

  describe('modelId propagation (issue #7)', () => {
    it('stamps entries with the configured embeddingModel, not "unknown"', async () => {
      const vs = createVectorStore(mockEmbeddings, { embeddingModel: 'nv-embedqa-e5-v5' });
      await vs.load();
      vs.addEntries([
        { id: '1', text: 'a', source: 's', embedding: new Array(4).fill(0.1) },
      ]);
      const results = await vs.similaritySearch('a', 1);
      expect(results[0].metadata?.['modelId']).toBe('nv-embedqa-e5-v5');
    });

    it('uses "(unspecified)" when no embeddingModel is configured', async () => {
      const vs = createVectorStore(mockEmbeddings);
      await vs.load();
      vs.addEntries([
        { id: '1', text: 'a', source: 's', embedding: new Array(4).fill(0.1) },
      ]);
      const results = await vs.similaritySearch('a', 1);
      expect(results[0].metadata?.['modelId']).not.toBe('unknown');
      expect(results[0].metadata?.['modelId']).toBe('(unspecified)');
    });
  });
});