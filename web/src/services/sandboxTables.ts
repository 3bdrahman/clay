// Live Arquero tables live outside Zustand (they're not JSON-serializable).
// Persisting CSVs in the sandboxDataset row lets us rehydrate them on reload.

import type { ColumnTable } from 'arquero';

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
