import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest';
import { loadSampleDatasets, SampleDatasetLoadError } from '../services/datasets';

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

    expect(tables.size).toBe(2);
    expect(tables.has('employees')).toBe(true);
    expect(tables.has('projects')).toBe(true);
    expect(metadata.employees.columns).toEqual(['department', 'salary']);
    expect(metadata.employees.rowCount).toBe(2);
    expect(rawCsv.employees).toContain('Engineering');
    expect(rawCsv.projects).toContain('Active');
  });

  it('throws when index.json fails to load (issue #6: no silent fallback)', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 404, statusText: 'Not Found' });

    await expect(loadSampleDatasets()).rejects.toThrow(/index/i);
  });

  it('throws with the HTTP status when index.json returns 5xx', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 503, statusText: 'Service Unavailable' });

    await expect(loadSampleDatasets()).rejects.toThrow(/503/);
  });

  it('throws SampleDatasetLoadError when a single file fails (issue #16: no partial silent load)', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ files: ['missing.csv'] }),
      })
      .mockResolvedValueOnce({ ok: false, status: 404, statusText: 'Not Found' });

    const p = loadSampleDatasets();
    await expect(p).rejects.toThrow(SampleDatasetLoadError);
    await expect(p).rejects.toThrow(/404/i);
  });

  it('aggregates multiple failures into one SampleDatasetLoadError with all failed names (issue #16)', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ files: ['ok.csv', 'bad1.csv', 'bad2.csv'] }),
      })
      .mockResolvedValueOnce({ ok: true, text: async () => 'a,b\n1,2' })
      .mockResolvedValueOnce({ ok: false, status: 500, statusText: 'ISE' })
      .mockResolvedValueOnce({ ok: false, status: 503, statusText: 'Unavailable' });

    let caught: unknown;
    try {
      await loadSampleDatasets();
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(SampleDatasetLoadError);
    const err = caught as SampleDatasetLoadError;
    expect(err.failedFiles).toEqual(['bad1', 'bad2']);
    expect(err.succeededFiles).toEqual(['ok']);
    expect(err.partialResult.tables.has('ok')).toBe(true);
    expect(err.partialResult.rawCsv.ok).toContain('1,2');
    expect(err.message).toMatch(/bad1/);
    expect(err.message).toMatch(/bad2/);
  });

  it('surfaces a non-partial failure message when every file fails (issue #16)', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ files: ['bad1.csv', 'bad2.csv'] }),
      })
      .mockResolvedValueOnce({ ok: false, status: 404, statusText: 'Not Found' })
      .mockResolvedValueOnce({ ok: false, status: 500, statusText: 'ISE' });

    await expect(loadSampleDatasets()).rejects.toThrow(/could not be loaded/i);
  });
});
