// Live Arquero tables live outside Zustand (they're not JSON-serializable).
// Persisting CSVs in the sandboxDataset row lets us rehydrate them on reload.

import type { ColumnTable } from 'arquero';
import type { DatasetMeta } from './analyzer';
import type { SandboxDataset } from '../store';
import { parseCsvNormalized } from './csvNormalize';

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
