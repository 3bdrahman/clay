import { describe, it, expect } from 'vitest';
import { rehydrateSandboxTables } from './sandboxTables';
import type { SandboxDataset } from '../store';

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
