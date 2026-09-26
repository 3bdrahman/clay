import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  rehydrateSandboxTables,
  persistSandboxCsv,
  loadPersistedSandboxCsvs,
  deletePersistedSandboxCsv,
  clearPersistedSandboxCsvs,
} from './sandboxTables';
import type { SandboxDataset } from '../store';

interface FakeStore { _data: Map<string, unknown>; keyPath: string; }

function installFakeIDB(): void {
  const dbs = new Map<string, Map<string, FakeStore>>();
  (globalThis as Record<string, unknown>).indexedDB = {
    open(name: string) {
      const reqObj = {
        result: undefined as unknown as {
          objectStoreNames: { contains(n: string): boolean };
          createObjectStore(n: string, opts: { keyPath: string }): FakeStore;
          transaction(s: string): { objectStore(n: string): { put(v: unknown): { onsuccess: (() => void) | null }; getAll(): { onsuccess: ((cb: () => void) => void) | null; result: unknown }; delete(k: string): { onsuccess: (() => void) | null }; clear(): { onsuccess: (() => void) | null } }; oncomplete: (() => void) | null };
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
        reqObj.result = {
          objectStoreNames: { contains: (n: string) => stores!.has(n) },
          createObjectStore(n: string, opts: { keyPath: string }) {
            const s: FakeStore = { _data: new Map(), keyPath: opts.keyPath };
            stores!.set(n, s);
            return {
              ...s,
              createIndex(name: string, keyPath: string) { return { name, keyPath }; },
            } as unknown as FakeStore & { createIndex(name: string, keyPath: string): unknown };
          },
          transaction(s: string) {
            const target = stores!.get(s);
            if (!target) throw new Error(`store ${s} missing`);
            return {
              objectStore(_s: string) {
                return {
                  put(v: unknown) {
                    const key = String((v as Record<string, unknown>)[target.keyPath]);
                    target._data.set(key, v);
                    const r = { onsuccess: null as (() => void) | null };
                    queueMicrotask(() => r.onsuccess?.());
                    return r;
                  },
                  getAll() {
                    const r = { onsuccess: null as ((cb: () => void) => void) | null, result: [...target._data.values()] as unknown };
                    queueMicrotask(() => r.onsuccess?.(r.result));
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
                };
              },
              oncomplete: null,
            };
          },
          close() { /* noop */ },
        };
        reqObj.onupgradeneeded?.(new Event('upgradeneeded'));
        reqObj.onsuccess?.(new Event('success'));
      });
      return reqObj;
    },
  };
}

describe('rehydrateSandboxTables', () => {
  it('rehydrates via the same normalized path as first load', () => {
    // A sample dataset whose first load normalized "$1,234.56" to a number;
    // reload re-parses the raw persisted CSV and must produce the same types.
    const dataset: SandboxDataset = {
      name: 'employees',
      fileName: 'sample/employees.csv',
      columns: ['department', 'salary'],
      rowCount: 2,
      loadedAt: 0,
      csv: 'department,salary\nEngineering,$100,000\nSales,$80,000',
      isSample: true,
    };

    const { tables, metadata } = rehydrateSandboxTables([dataset]);

    const rows = (tables.get('employees') as { objects: () => Array<Record<string, unknown>> }).objects();
    expect(rows[0]?.['salary']).toBe(100000);
    expect(rows[1]?.['salary']).toBe(80000);
    expect(metadata.employees).toEqual({ columns: ['department', 'salary'], rowCount: 2 });
  });

  it('drops datasets without persisted csv instead of fabricating empty tables', () => {
    const dataset: SandboxDataset = {
      name: 'legacy',
      fileName: 'legacy.csv',
      columns: ['a', 'b'],
      rowCount: 5,
      loadedAt: 0,
    };

    const { tables, metadata } = rehydrateSandboxTables([dataset]);

    expect(tables.has('legacy')).toBe(false);
    expect(metadata.legacy).toBeUndefined();
  });
});

describe('sandbox csv persistence', () => {
  beforeEach(() => {
    installFakeIDB();
  });

  afterEach(() => {
    (globalThis as Record<string, unknown>).indexedDB = undefined;
  });

  it('persists and loads a csv round-trip', async () => {
    await persistSandboxCsv('employees', 'department,salary\nEng,100000');
    const csvs = await loadPersistedSandboxCsvs();
    expect(csvs.get('employees')).toBe('department,salary\nEng,100000');
  });

  it('deletes a single persisted csv', async () => {
    await persistSandboxCsv('a', 'x\n1');
    await persistSandboxCsv('b', 'y\n2');
    await deletePersistedSandboxCsv('a');
    const csvs = await loadPersistedSandboxCsvs();
    expect(csvs.has('a')).toBe(false);
    expect(csvs.has('b')).toBe(true);
  });

  it('clears every persisted csv', async () => {
    await persistSandboxCsv('a', 'x\n1');
    await persistSandboxCsv('b', 'y\n2');
    await clearPersistedSandboxCsvs();
    const csvs = await loadPersistedSandboxCsvs();
    expect(csvs.size).toBe(0);
  });

  it('returns an empty map when IDB is unavailable', async () => {
    (globalThis as Record<string, unknown>).indexedDB = undefined;
    const csvs = await loadPersistedSandboxCsvs();
    expect(csvs.size).toBe(0);
  });
});
