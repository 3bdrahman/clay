/**
 * Shared fake IndexedDB helper for vectorstore tests. Mirrors the surface
 * that `web/src/lib/idb.ts` exercises (open + upgrade + transactions +
 * put/getAll/delete/clear + indexes with getAll/getAllKeys).
 */
import { vi } from 'vitest';

interface FakeRecord {
  key: string;
  value: Record<string, unknown>;
}

interface FakeIndex {
  name: string;
  keyPath: string;
  lookup: Map<string, Set<string>>;
}

interface FakeRequest<T = unknown> {
  result: T | undefined;
  error: Error | null;
  onsuccess: ((this: unknown, ev: Event) => void) | null;
  onerror: ((this: unknown, ev: Event) => void) | null;
}

interface FakeCursorRequest {
  result: unknown;
  error: Error | null;
  onsuccess: ((this: unknown, ev: Event) => void) | null;
}

type FakeIndexOps = {
  getAll(key: string): FakeRequest<Record<string, unknown>[]>;
  getAllKeys(key: string): FakeRequest<string[]>;
  openCursor(): FakeCursorRequest;
};

type FakeStoreOps = {
  put(value: Record<string, unknown>): FakeRequest<string>;
  getAll(): FakeRequest<Record<string, unknown>[]>;
  delete(key: string): FakeRequest<undefined>;
  clear(): FakeRequest<undefined>;
  createIndex(name: string, keyPath: string): FakeIndex;
  index(name: string): FakeIndexOps;
};

type FakeStore = FakeObjectStore & FakeStoreOps;

interface FakeObjectStore {
  name: string;
  keyPath: string;
  records: FakeRecord[];
  indexes: Map<string, FakeIndex>;
}

interface FakeTransaction {
  storeNames: string[];
  mode: 'readonly' | 'readwrite';
  objectStore(name: string): FakeStore;
  oncomplete: ((this: unknown, ev: Event) => void) | null;
  onerror: ((this: unknown, ev: Event) => void) | null;
  onabort: ((this: unknown, ev: Event) => void) | null;
  error: Error | null;
}

export interface FakeDatabase {
  name: string;
  version: number;
  stores: Map<string, FakeStore>;
  objectStoreNames: { contains(n: string): boolean };
  createObjectStore(name: string, opts: { keyPath: string }): FakeStore;
  transaction(names: string[], mode: 'readonly' | 'readwrite'): FakeTransaction;
  close(): void;
}

interface FakeOpenRequest {
  result: FakeDatabase | undefined;
  error: Error | null;
  onsuccess: ((this: unknown, ev: Event) => void) | null;
  onerror: ((this: unknown, ev: Event) => void) | null;
  onupgradeneeded: ((this: unknown, ev: Event) => void) | null;
}

export interface FakeIDB {
  databases: Map<string, FakeDatabase>;
  open(name: string, version: number): FakeOpenRequest;
}

function extractValue(value: unknown, keyPath: string): string {
  const parts = keyPath.split('.');
  let cur: unknown = value;
  for (const p of parts) {
    if (cur === null || cur === undefined || typeof cur !== 'object') return '';
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur === undefined || cur === null ? '' : String(cur);
}

function makeReq<T>(result: T): FakeRequest<T> {
  const req: FakeRequest<T> = { result, error: null, onsuccess: null, onerror: null };
  queueMicrotask(() => {
    if (req.onsuccess) req.onsuccess.call(req, new Event('success'));
  });
  return req;
}

function attachOps(store: FakeObjectStore): FakeStore {
  const obj = store as FakeStore;
  obj.put = (value: Record<string, unknown>): FakeRequest<string> => {
    const key = extractValue(value, store.keyPath);
    if (key === '') {
      return { result: undefined, error: new Error('Invalid key'), onsuccess: null, onerror: null };
    }
    const existingIdx = store.records.findIndex((r) => r.key === key);
    const record: FakeRecord = { key, value };
    if (existingIdx >= 0) store.records[existingIdx] = record;
    else store.records.push(record);
    for (const idx of store.indexes.values()) {
      const idxKey = extractValue(value, idx.keyPath);
      let bucket = idx.lookup.get(idxKey);
      if (!bucket) {
        bucket = new Set();
        idx.lookup.set(idxKey, bucket);
      }
      bucket.add(key);
    }
    return makeReq(key);
  };

  obj.getAll = (): FakeRequest<Record<string, unknown>[]> =>
    makeReq(store.records.map((r) => ({ ...r.value })));

  obj.delete = (key: string): FakeRequest<undefined> => {
    const idx = store.records.findIndex((r) => r.key === key);
    if (idx >= 0) store.records.splice(idx, 1);
    for (const i of store.indexes.values()) {
      for (const [k, bucket] of [...i.lookup]) {
        bucket.delete(key);
        if (bucket.size === 0) i.lookup.delete(k);
      }
    }
    return makeReq(undefined);
  };

  obj.clear = (): FakeRequest<undefined> => {
    store.records.length = 0;
    for (const idx of store.indexes.values()) idx.lookup.clear();
    return makeReq(undefined);
  };

  obj.createIndex = (name: string, keyPath: string): FakeIndex => {
    const idx: FakeIndex = { name, keyPath, lookup: new Map() };
    store.indexes.set(name, idx);
    for (const rec of store.records) {
      const k = extractValue(rec.value, keyPath);
      let bucket = idx.lookup.get(k);
      if (!bucket) {
        bucket = new Set();
        idx.lookup.set(k, bucket);
      }
      bucket.add(rec.key);
    }
    return idx;
  };

  obj.index = (name: string): FakeIndexOps => {
    const idx = store.indexes.get(name);
    if (!idx) throw new Error(`Index ${name} not found on store ${store.name}`);
    return {
      getAll(key: string): FakeRequest<Record<string, unknown>[]> {
        const ids = idx.lookup.get(key);
        if (!ids) return makeReq([]);
        const out: Record<string, unknown>[] = [];
        for (const rec of store.records) {
          if (ids.has(rec.key)) out.push({ ...rec.value });
        }
        return makeReq(out);
      },
      getAllKeys(key: string): FakeRequest<string[]> {
        const ids = idx.lookup.get(key);
        if (!ids) return makeReq([]);
        return makeReq(Array.from(ids));
      },
      openCursor(): FakeCursorRequest {
        return { result: null, error: null, onsuccess: null };
      },
    };
  };

  return obj;
}

function makeDatabase(name: string, version: number): FakeDatabase {
  const stores = new Map<string, FakeStore>();
  const objectStoreNames = {
    contains(n: string): boolean {
      return stores.has(n);
    },
  };
  function createObjectStore(storeName: string, opts: { keyPath: string }): FakeStore {
    const base: FakeObjectStore = {
      name: storeName,
      keyPath: opts.keyPath,
      records: [],
      indexes: new Map(),
    };
    const store = attachOps(base);
    stores.set(storeName, store);
    return store;
  }
  function transaction(storeNames: string[], mode: 'readonly' | 'readwrite'): FakeTransaction {
    const tx: FakeTransaction = {
      storeNames,
      mode,
      oncomplete: null,
      onerror: null,
      onabort: null,
      error: null,
      objectStore(n: string): FakeStore {
        const s = stores.get(n);
        if (!s) throw new Error(`Object store ${n} not found`);
        return s;
      },
    };
    queueMicrotask(() => {
      if (tx.oncomplete) tx.oncomplete.call(tx, new Event('complete'));
    });
    return tx;
  }
  function close(): void {}
  return { name, version, stores, objectStoreNames, createObjectStore, transaction, close };
}

export function createFakeIDB(): FakeIDB {
  const databases = new Map<string, FakeDatabase>();
  return {
    databases,
    open(name: string, version: number): FakeOpenRequest {
      const request: FakeOpenRequest = {
        result: undefined,
        error: null,
        onsuccess: null,
        onerror: null,
        onupgradeneeded: null,
      };
      let db = databases.get(name);
      const isNew = !db;
      if (!db) {
        db = makeDatabase(name, version);
        databases.set(name, db);
      }
      queueMicrotask(() => {
        try {
          if (isNew) {
            request.result = db;
            if (request.onupgradeneeded) request.onupgradeneeded.call(request, new Event('upgradeneeded'));
            if (request.onsuccess) request.onsuccess.call(request, new Event('success'));
          } else if (version > db.version) {
            db.version = version;
            request.result = db;
            if (request.onupgradeneeded) request.onupgradeneeded.call(request, new Event('upgradeneeded'));
            if (request.onsuccess) request.onsuccess.call(request, new Event('success'));
          } else {
            request.result = db;
            if (request.onsuccess) request.onsuccess.call(request, new Event('success'));
          }
        } catch (e) {
          request.error = e instanceof Error ? e : new Error(String(e));
          if (request.onerror) request.onerror.call(request, new Event('error'));
        }
      });
      return request;
    },
  };
}

export function installFakeIDB(): FakeIDB {
  const fake = createFakeIDB();
  vi.stubGlobal('indexedDB', fake);
  return fake;
}
