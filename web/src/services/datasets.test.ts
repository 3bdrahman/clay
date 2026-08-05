import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest';
import { loadSampleDatasets, parseUserCsv } from '../services/datasets';

describe('loadSampleDatasets', () => {
  const originalFetch = globalThis.fetch;
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch = vi.fn();
    globalThis.fetch = mockFetch;
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
  });

  it('loads datasets from index.json and CSV files', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ files: ['employees.csv', 'projects.csv'] }),
      })
      .mockResolvedValueOnce({
        ok: true,
        text: async () => 'department,salary\nEngineering,100000\nSales,80000',
      })
      .mockResolvedValueOnce({
        ok: true,
        text: async () => 'status,budget\nActive,50000\nCompleted,30000',
      });

    const { tables, metadata, rawCsv } = await loadSampleDatasets();

    expect(tables.has('aq')).toBe(true);
    expect(tables.has('employees')).toBe(true);
    expect(tables.has('projects')).toBe(true);
    expect(metadata.employees.columns).toEqual(['department', 'salary']);
    expect(metadata.employees.rowCount).toBe(2);
    expect(rawCsv.employees).toContain('Engineering');
    expect(rawCsv.projects).toContain('Active');
  });

  it('falls back to known filenames when index.json fails', async () => {
    mockFetch
      .mockResolvedValueOnce({ ok: false })
      .mockResolvedValueOnce({
        ok: true,
        text: async () => 'name,value\ntest,123',
      })
      .mockResolvedValueOnce({ ok: false })
      .mockResolvedValueOnce({ ok: false });

    const { tables, metadata } = await loadSampleDatasets();

    expect(tables.has('employees')).toBe(true);
    expect(metadata.employees.rowCount).toBe(1);
  });

  it('skips missing files gracefully', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ files: ['missing.csv'] }),
      })
      .mockResolvedValueOnce({ ok: false });

    const { tables } = await loadSampleDatasets();
    expect(tables.has('missing')).toBe(false);
  });
});

describe('parseUserCsv', () => {
  it('parses CSV into Arquero table', () => {
    const csv = 'name,age\nAlice,30\nBob,25';
    const { table, columns, rowCount } = parseUserCsv(csv);

    expect(columns).toEqual(['name', 'age']);
    expect(rowCount).toBe(2);
    const rows = table.objects() as Array<{ name: string; age: number }>;
    expect(rows[0]).toEqual({ name: 'Alice', age: 30 });
    expect(rows[1]).toEqual({ name: 'Bob', age: 25 });
  });

  it('normalizes numeric strings', () => {
    const csv = 'price\n100\n200.50';
    const { table } = parseUserCsv(csv);
    const rows = table.objects() as Array<{ price: number }>;
    expect(rows[0].price).toBe(100);
    expect(rows[1].price).toBe(200.5);
  });

  it('normalizes currency strings', () => {
    const csv = 'amount\n$1,234.56\n$99';
    const { table } = parseUserCsv(csv);
    const rows = table.objects() as Array<{ amount: number }>;
    expect(rows[0].amount).toBe(1234.56);
    expect(rows[1].amount).toBe(99);
  });

  it('normalizes percentages', () => {
    const csv = 'rate\n50%\n25.5%';
    const { table } = parseUserCsv(csv);
    const rows = table.objects();
    expect(rows[0].rate).toBe(0.5);
    expect(rows[1].rate).toBe(0.255);
  });

  it('handles empty strings as null', () => {
    const csv = 'name,value\nAlice,\nBob,100';
    const { table } = parseUserCsv(csv);
    const rows = table.objects();
    expect(rows[0].value).toBeNull();
    expect(rows[1].value).toBe(100);
  });
});