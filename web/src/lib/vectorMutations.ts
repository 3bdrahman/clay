import type { IDBStore } from './idb';
import type { HnswIndex } from './hnsw';
import { VectorStoreCorruptedError } from './errors';
import { enqueueWrite } from './vectorWriteQueue';
import { clearLegacyKey } from './vectorLegacyMigration';
import { normalize } from './vectorSearchMath';
import { estimateTokens } from './tokens';
import type { ChunkMetadata } from './types';

export interface VectorEntry {
  id: string;
  text: string;
  embedding: Float32Array;
  metadata: ChunkMetadata;
}

export interface MutationConfig {
  entryModelId: string;
}

export interface MutationDeps {
  memory: Map<string, VectorEntry>;
  db: IDBStore<VectorEntry> | null;
  annIndex: HnswIndex | null;
  pendingAdds: VectorEntry[];
  knownDimension: number | null;
  setKnownDimension: (v: number | null) => void;
  setAnnIndex: (v: HnswIndex | null) => void;
  load: () => Promise<void>;
  loaded: boolean;
  loadingPromise: Promise<void> | null;
}

export function addEntries(
  newEntries: Array<{ id: string; text: string; source: string; sourceHash?: string; page?: number; embedding: number[] }>,
  deps: MutationDeps,
  config: MutationConfig,
  classifyIDBError: (error: unknown, operation: string) => Error
): void {
  if (!deps.loaded && !deps.loadingPromise) void deps.load();

  for (const e of newEntries) {
    const emb = normalize(e.embedding);
    if (deps.knownDimension === null) {
      deps.setKnownDimension(emb.length);
    } else if (emb.length !== deps.knownDimension) {
      throw new VectorStoreCorruptedError(
        `embedding dimension mismatch: stored ${deps.knownDimension}, got ${emb.length}`
      );
    }
    const entry: VectorEntry = {
      id: e.id,
      text: e.text,
      embedding: emb,
      metadata: {
        source: e.source,
        sourceHash: e.sourceHash ?? '',
        page: e.page,
        charStart: 0,
        charEnd: e.text.length,
        chunkIndex: deps.memory.size,
        tokenCount: estimateTokens(e.text),
        modelId: config.entryModelId,
        updatedAt: Date.now(),
      },
    };
    deps.memory.set(entry.id, entry);
    if (deps.annIndex !== null) deps.annIndex.insert(entry.id, entry.embedding);
    deps.pendingAdds.push(entry);
    if (deps.db) {
      const capture = deps.db;
      enqueueWrite(async () => {
        try {
          await capture.put(entry);
        } catch (e) {
          throw classifyIDBError(e, 'put');
        }
      });
    } else if (!deps.loaded && !deps.loadingPromise) {
      void deps.load();
    }
  }
}

export function removeBySource(
  source: string,
  deps: MutationDeps,
  classifyIDBError: (error: unknown, operation: string) => Error
): number {
  const before = deps.memory.size;
  const removedIds: string[] = [];
  for (const [id, e] of deps.memory) {
    if (e.metadata.source === source) {
      deps.memory.delete(id);
      removedIds.push(id);
    }
  }
  if (deps.annIndex !== null) {
    for (const id of removedIds) deps.annIndex.markDeleted(id);
  }
  if (deps.db) enqueueWrite(async () => {
    try {
      await deps.db!.deleteByIndex('source', source);
    } catch (e) {
      throw classifyIDBError(e, 'deleteByIndex');
    }
  });
  return before - deps.memory.size;
}

export function clear(
  deps: MutationDeps,
  classifyIDBError: (error: unknown, operation: string) => Error
): void {
  deps.memory.clear();
  deps.setKnownDimension(null);
  deps.setAnnIndex(null);
  clearLegacyKey();
  if (deps.db) enqueueWrite(async () => {
    try {
      await deps.db!.clear();
    } catch (e) {
      throw classifyIDBError(e, 'clear');
    }
  });
}