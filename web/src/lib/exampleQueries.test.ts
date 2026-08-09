import { describe, expect, it } from 'vitest';
import { deriveDataQueries, deriveDocumentQueries } from './exampleQueries';

describe('deriveDataQueries', () => {
  it('returns an empty list when no datasets are loaded', () => {
    expect(deriveDataQueries([])).toEqual([]);
  });

  it('uses real column names instead of bundled sample column names', () => {
    const out = deriveDataQueries([
      { name: 'rideshare', fileName: 'rideshare.csv', columns: ['city', 'fare_amount', 'tip_pct'], rowCount: 1000 },
    ]);
    expect(out).toEqual([
      'Total fare_amount',
    ]);
  });

  it('combines numeric and categorical columns for a per-group query', () => {
    const out = deriveDataQueries([
      { name: 'staff', fileName: 'staff.csv', columns: ['department', 'salary'], rowCount: 50 },
    ]);
    expect(out).toEqual([
      'Average salary by department',
      'Total salary',
    ]);
  });

  it('suggests joining when more than one dataset is loaded', () => {
    const out = deriveDataQueries([
      { name: 'a', fileName: 'a.csv', columns: ['x'], rowCount: 1 },
      { name: 'b', fileName: 'b.csv', columns: ['y'], rowCount: 1 },
    ]);
    expect(out).toContain('Join these datasets');
  });

  it('falls back to dataset-name queries when no recognized columns exist', () => {
    const out = deriveDataQueries([
      { name: 'logs', fileName: 'logs.csv', columns: ['foo', 'bar'], rowCount: 5 },
    ]);
    expect(out).toEqual([
      'Summarize logs',
      'Show the first rows of logs',
    ]);
  });

  it('caps suggestions at three entries', () => {
    const out = deriveDataQueries([
      { name: 'a', fileName: 'a.csv', columns: ['department', 'salary'], rowCount: 1 },
      { name: 'b', fileName: 'b.csv', columns: ['category', 'price'], rowCount: 1 },
      { name: 'c', fileName: 'c.csv', columns: ['type', 'revenue'], rowCount: 1 },
    ]);
    expect(out.length).toBeLessThanOrEqual(3);
  });

  it('does not reference columns from bundled sample datasets', () => {
    const out = deriveDataQueries([
      { name: 'custom', fileName: 'custom.csv', columns: ['team', 'cost'], rowCount: 1 },
    ]);
    const forbidden = ['employees', 'projects', 'budget', 'rating'];
    for (const q of out) {
      for (const f of forbidden) {
        expect(q.toLowerCase()).not.toContain(f);
      }
    }
  });
});

describe('deriveDocumentQueries', () => {
  it('returns an empty list when no documents are loaded', () => {
    expect(deriveDocumentQueries([])).toEqual([]);
  });

  it('references a single uploaded document by fileName', () => {
    const out = deriveDocumentQueries([{ fileName: 'notes.md' }]);
    expect(out[0]).toContain('notes.md');
    expect(out).toHaveLength(3);
  });

  it('summarizes across multiple uploaded documents', () => {
    const out = deriveDocumentQueries([
      { fileName: 'a.pdf' },
      { fileName: 'b.md' },
    ]);
    expect(out[0]).toContain('2 uploaded documents');
  });

  it('lists up to three file names in the action-items query', () => {
    const out = deriveDocumentQueries([
      { fileName: 'a.pdf' },
      { fileName: 'b.md' },
      { fileName: 'c.txt' },
      { fileName: 'd.json' },
    ]);
    expect(out[1]).toContain('a.pdf');
    expect(out[1]).toContain('c.txt');
    expect(out[1]).not.toContain('d.json');
  });
});
