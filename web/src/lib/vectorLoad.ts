import type { IDBStore } from './idb';
import { openIDB, wrapIDBStore, clayDBUpgrade, VECTOR_STORE_NAME, VECTOR_DB_NAME, VECTOR_DB_VERSION } from './idb';
import { readLegacy, LEGACY_KEY } from './vectorLegacyMigration';
import type { ChunkMetadata } from './types';
import { awaitWriteQueue, getWriteQueueState } from './vectorWriteQueue';

export interface VectorEntry {
  id: string;
  text: string;
  embedding: Float32Array;
  metadata: ChunkMetadata;
}

export interface LoadConfig {
  embeddingModel: string;
}

export interface LoadResult {
  memory: Map<string, VectorEntry>;
  db: IDBStore<VectorEntry> | null;
  persistenceAvailable: boolean;
  pendingAdds: VectorEntry[];
  knownDimension: number | null;
}

export async function doLoad(
  config: LoadConfig,
  pendingAdds: VectorEntry[],
  warnFallbackOnce: (cause?: unknown) => void
): Promise<LoadResult> {
  const memory = new Map<string, VectorEntry>();
  let db: IDBStore<VectorEntry> | null = null;
  let persistenceAvailable = false;
  let knownDimension: number | null = null;

  try {
    const idb = await openIDB(VECTOR_DB_NAME, VECTOR_DB_VERSION, clayDBUpgrade);
    db = wrapIDBStore<VectorEntry>(idb, VECTOR_STORE_NAME);
    if (pendingAdds.length > 0) {
      const toFlush = pendingAdds.splice(0, pendingAdds.length);
      for (const e of toFlush) await db.put(e);
    }
    const existing = await db.getAll();
    if (existing.length === 0) {
      const legacy = readLegacy(config.embeddingModel);
      if (legacy && legacy.length > 0) {
        await db.putMany(legacy);
        localStorage.removeItem(LEGACY_KEY);
        for (const e of legacy) memory.set(e.id, e);
      } else if (legacy !== null) {
        // Corruption path: parsed OK but zero valid entries. Still clear the legacy key
        // so we don't re-attempt migration on every load. User's data was unreadable; no
        // further recovery possible.
        localStorage.removeItem(LEGACY_KEY);
      }
    } else {
      for (const e of existing) {
        // Entries written before the Float32Array representation load as
        // number[]; normalize the in-memory representation so every entry
        // is uniform for the scan and index paths.
        memory.set(e.id, { ...e, embedding: Float32Array.from(e.embedding) });
      }
    }
  } catch (e) {
    warnFallbackOnce(e);
    db = null;
    persistenceAvailable = false;
    // Don't throw here - allow fallback to in-memory mode
  }

  if (db !== null) persistenceAvailable = true;

  return { memory, db, persistenceAvailable, pendingAdds, knownDimension };
}

export async function load(
  loaded: boolean,
  loadingPromise: Promise<void> | null,
  doLoadFn: () => Promise<LoadResult>,
  setLoaded: (v: boolean) => void,
  setLoadingPromise: (v: Promise<void> | null) => void,
  setPersistenceAvailable: (v: boolean) => void,
  setMemory: (v: Map<string, VectorEntry>) => void,
  setDb: (v: IDBStore<VectorEntry> | null) => void,
  setKnownDimension: (v: number | null) => void,
  setPendingAdds: (v: VectorEntry[]) => void
): Promise<void> {
  if (loaded) return;
  if (loadingPromise) return loadingPromise;

  const promise = (async () => {
    await awaitWriteQueue();
    const { failed, failure } = getWriteQueueState();
    if (failed && failure) {
      throw failure;
    }
    const result = await doLoadFn();
    setMemory(result.memory);
    setDb(result.db);
    setPersistenceAvailable(result.persistenceAvailable);
    setKnownDimension(result.knownDimension);
    setPendingAdds(result.pendingAdds);
    await awaitWriteQueue();
    const { failed: failed2, failure: failure2 } = getWriteQueueState();
    if (failed2 && failure2) {
      throw failure2;
    }
    setLoaded(true);
    setLoadingPromise(null);
  })();

  setLoadingPromise(promise);
  return promise;
}