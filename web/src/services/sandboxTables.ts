// Live Arquero tables live outside Zustand (they're not JSON-serializable).
// The raw csv that rebuilds them persists to IndexedDB — localStorage cannot
// hold large CSVs, and the persisted store rows carry metadata only.

import type { ColumnTable } from 'arquero';
import type { DatasetMeta } from './analyzer';
import type { SandboxDataset } from '../store';
import { parseCsvNormalized } from './csvNormalize';
import { openIDB, wrapIDBStore, type IDBStore } from '../lib/idb';
import { VECTOR_DB_NAME, VECTOR_DB_VERSION, clayDBUpgrade } from '../lib/vectorstore';
import { VectorStoreQuotaExceededError } from '../lib/errors';

const SANDBOX_STORE_NAME = 'sandbox';

interface PersistedSandboxCsv {
  name: string;
  csv: string;
}

async function openSandboxStore(): Promise<IDBStore<PersistedSandboxCsv>> {
  const db = await openIDB(VECTOR_DB_NAME, VECTOR_DB_VERSION, clayDBUpgrade);
  return wrapIDBStore<PersistedSandboxCsv>(db, SANDBOX_STORE_NAME);
}

let persistenceWarned = false;

function warnPersistenceUnavailableOnce(cause: unknown): void {
  if (persistenceWarned) return;
  persistenceWarned = true;
  if (import.meta.env.DEV) {
    console.warn('[sandboxTables] sandbox csv persistence unavailable (datasets will not survive a reload):', cause);
  }
}

export async function persistSandboxCsv(name: string, csv: string): Promise<void> {
  let store: IDBStore<PersistedSandboxCsv>;
  try {
    store = await openSandboxStore();
  } catch (e) {
    // Same optional-persistence contract as the vectorstore: an unavailable
    // IDB (private mode, unsupported browser) degrades to in-memory only.
    warnPersistenceUnavailableOnce(e);
    return;
  }
  try {
    await store.put({ name, csv });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'QuotaExceededError') {
      throw new VectorStoreQuotaExceededError(e);
    }
    throw e;
  } finally {
    store.close();
  }
}

export async function loadPersistedSandboxCsvs(): Promise<Map<string, string>> {
  try {
    const store = await openSandboxStore();
    try {
      const rows = await store.getAll();
      return new Map(rows.map(r => [r.name, r.csv]));
    } finally {
      store.close();
    }
  } catch (e) {
    // Persistence is best-effort: with IDB unavailable (private mode,
    // unsupported browser) datasets cannot survive a reload — the same
    // in-memory fallback contract the vectorstore uses.
    if (import.meta.env.DEV) {
      console.warn('[sandboxTables] persisted csv load failed (datasets will need re-upload):', e);
    }
    return new Map();
  }
}

export async function deletePersistedSandboxCsv(name: string): Promise<void> {
  let store: IDBStore<PersistedSandboxCsv>;
  try {
    store = await openSandboxStore();
  } catch (e) {
    warnPersistenceUnavailableOnce(e);
    return;
  }
  try {
    await store.delete(name);
  } finally {
    store.close();
  }
}

export async function clearPersistedSandboxCsvs(): Promise<void> {
  let store: IDBStore<PersistedSandboxCsv>;
  try {
    store = await openSandboxStore();
  } catch (e) {
    warnPersistenceUnavailableOnce(e);
    return;
  }
  try {
    await store.clear();
  } finally {
    store.close();
  }
}

const tables = new Map<string, ColumnTable>();

export function registerSandboxTable(name: string, table: ColumnTable): void {
  tables.set(name, table);
}

export function unregisterSandboxTable(name: string): void {
  tables.delete(name);
}

export function getSandboxTable(name: string): ColumnTable | undefined {
  return tables.get(name);
}

export function listSandboxTableNames(): string[] {
  return [...tables.keys()];
}

export function clearSandboxTables(): void {
  tables.clear();
}

export interface RehydratedSandbox {
  tables: Map<string, unknown>;
  metadata: DatasetMeta;
}

/**
 * Rebuild live Arquero tables from the persisted sandbox dataset rows using
 * the same normalized parse path as first load — a dataset's column types
 * must not depend on which path loaded it. Datasets whose raw csv is missing
 * (legacy persisted state) are dropped rather than replaced with fabricated
 * empty tables, so the analyzer reports their absence instead of querying
 * garbage.
 */
export function rehydrateSandboxTables(datasets: readonly SandboxDataset[]): RehydratedSandbox {
  const rehydrated: Map<string, unknown> = new Map();
  const metadata: DatasetMeta = {};
  for (const d of datasets) {
    if (d.csv === undefined) continue;
    rehydrated.set(d.name, parseCsvNormalized(d.csv));
    metadata[d.name] = { columns: d.columns, rowCount: d.rowCount };
  }
  return { tables: rehydrated, metadata };
}
