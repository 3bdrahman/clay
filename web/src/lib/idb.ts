/** Hand-rolled async IndexedDB wrapper. No npm deps. */
export const VECTOR_DB_NAME = 'clay-vector-db';
export const VECTOR_DB_VERSION = 2;

export interface IDBStore<T> {
  put(value: T): Promise<void>;
  putMany(values: T[], batchSize?: number): Promise<void>;
  getAll(): Promise<T[]>;
  delete(key: string): Promise<void>;
  deleteByIndex(indexName: string, key: IDBValidKey): Promise<number>;
  clear(): Promise<void>;
  close(): void;
}

export async function openIDB(
  dbName: string,
  version: number,
  upgrade: (db: IDBDatabase) => void,
): Promise<IDBDatabase> {
  if (typeof indexedDB === 'undefined') {
    throw new Error('IndexedDB is unavailable in this environment');
  }
  return new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(dbName, version);
    if (upgrade) req.onupgradeneeded = () => upgrade(req.result);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('indexedDB.open failed'));
    req.onblocked = () => reject(new Error('indexedDB.open blocked'));
  });
}

function reqAsPromise<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IDBRequest failed'));
  });
}

export function wrapIDBStore<T>(db: IDBDatabase, storeName: string): IDBStore<T> {
  function tx(mode: IDBTransactionMode): IDBObjectStore {
    return db.transaction(storeName, mode).objectStore(storeName);
  }
  return {
    async put(value) {
      await reqAsPromise(tx('readwrite').put(value));
    },
    async putMany(values, batchSize = 50) {
      for (let i = 0; i < values.length; i += batchSize) {
        const slice = values.slice(i, i + batchSize);
        for (const v of slice) {
          await reqAsPromise(tx('readwrite').put(v));
        }
      }
    },
    async getAll() {
      return reqAsPromise<T[]>(tx('readonly').getAll() as IDBRequest<T[]>);
    },
    async delete(key) {
      await reqAsPromise(tx('readwrite').delete(key));
    },
    async deleteByIndex(indexName, key) {
      const txn = db.transaction(storeName, 'readwrite');
      const store = txn.objectStore(storeName);
      const idx = store.index(indexName);
      const keys = await reqAsPromise<IDBValidKey[]>(idx.getAllKeys(key));
      for (const k of keys) {
        await reqAsPromise(store.delete(k));
      }
      return keys.length;
    },
    async clear() {
      await reqAsPromise(tx('readwrite').clear());
    },
    close() {
      db.close();
    },
  };
}

export const SANDBOX_STORE_NAME = 'sandbox';
export const VECTOR_STORE_NAME = 'entries';

/**
 * The shared upgrade for the clay vector DB: creates the vector entries store
 * AND the sandbox csv store, so whichever connection opens first (both use
 * VECTOR_DB_VERSION) leaves both stores present regardless of open order.
 */
export function clayDBUpgrade(db: IDBDatabase): void {
  if (!db.objectStoreNames.contains(VECTOR_STORE_NAME)) {
    const store = db.createObjectStore(VECTOR_STORE_NAME, { keyPath: 'id' });
    store.createIndex('source', 'metadata.source', { unique: false });
    store.createIndex('modelId', 'metadata.modelId', { unique: false });
  }
  if (!db.objectStoreNames.contains(SANDBOX_STORE_NAME)) {
    db.createObjectStore(SANDBOX_STORE_NAME, { keyPath: 'name' });
  }
}
