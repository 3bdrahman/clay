import * as aq from 'arquero';
import type { DatasetMeta } from '../services/analyzer';

export interface SampleLoadResult {
  tables: Map<string, unknown>;
  metadata: DatasetMeta;
  rawCsv: Record<string, string>;
}

// Get the base path for correct asset loading on GitHub Pages
const getBasePath = (): string => {
  return import.meta.env.BASE_URL || '/';
};

export async function loadSampleDatasets(): Promise<SampleLoadResult> {
  const tables = new Map<string, unknown>();
  const metadata: DatasetMeta = {};
  const rawCsv: Record<string, string> = {};

  const basePath = getBasePath();
  const indexResp = await fetch(`${basePath}data/datasets/index.json`, { cache: 'no-store' });
  let names: string[] = [];
  if (indexResp.ok) {
    const idx = (await indexResp.json()) as { files?: string[] };
    names = (idx.files || []).filter(n => n.endsWith('.csv'));
  } else {
    names = ['employees.csv', 'projects.csv', 'feedback.csv'];
  }

  await Promise.all(
    names.map(async fileName => {
      const name = fileName.replace(/\.csv$/i, '');
      try {
        const resp = await fetch(`${basePath}data/datasets/${fileName}`, { cache: 'no-store' });
        if (!resp.ok) return;
        const text = await resp.text();
        const table = aq.fromCSV(text);
        const rows = table.objects() as Array<Record<string, unknown>>;
        const normalized = rows.map(normalizeRow);
        const normalizedTable = aq.from(normalized);
        const columns = typeof normalizedTable.columnNames === 'function'
          ? normalizedTable.columnNames()
          : Object.keys(normalized[0] ?? {});
        const rowCount = typeof normalizedTable.numRows === 'function'
          ? normalizedTable.numRows()
          : normalized.length;
        tables.set(name, normalizedTable);
        metadata[name] = { columns, rowCount };
        rawCsv[name] = text;
      } catch (e) {
        if (import.meta.env.DEV) console.warn(`[clay] failed to load dataset ${name}:`, e);
      }
    })
  );

  tables.set('aq', aq);

  return { tables, metadata, rawCsv };
}

export function parseUserCsv(csv: string): {
  table: ReturnType<typeof aq.from>;
  columns: string[];
  rowCount: number;
} {
  // Pre-process CSV to normalize currency values in data rows (not header)
  // Only matches clear currency patterns: $X,XXX.XX or X,XXX,XXX.XX (thousands separators)
  const currencyRegex = /\$[\d,]+\.?\d*|\b\d{1,3}(,\d{3})+\.?\d*\b/g;
  
  const lines = csv.split('\n');
  if (lines.length < 2) {
    const table = aq.fromCSV(csv);
    const rows = table.objects() as Array<Record<string, unknown>>;
    const normalized = rows.map(normalizeRow);
    const normalizedTable = aq.from(normalized);
    const columns = typeof normalizedTable.columnNames === 'function'
      ? normalizedTable.columnNames()
      : Object.keys(normalized[0] ?? {});
    const rowCount = typeof normalizedTable.numRows === 'function'
      ? normalizedTable.numRows()
      : normalized.length;
    return { table: normalizedTable, columns, rowCount };
  }
  
  const header = lines[0];
  const dataLines = lines.slice(1).map(line => 
    line.replace(currencyRegex, (match) => match.replace(/[$,]/g, ''))
  );
  const processedCsv = [header, ...dataLines].join('\n');
  
  const table = aq.fromCSV(processedCsv);
  const rows = table.objects() as Array<Record<string, unknown>>;
  const normalized = rows.map(normalizeRow);
  const normalizedTable = aq.from(normalized);
  const columns = typeof normalizedTable.columnNames === 'function'
    ? normalizedTable.columnNames()
    : Object.keys(normalized[0] ?? {});
  const rowCount = typeof normalizedTable.numRows === 'function'
    ? normalizedTable.numRows()
    : normalized.length;
  return { table: normalizedTable, columns, rowCount };
}

function normalizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(row)) {
    if (value === null || value === undefined || value === '') {
      out[key] = null;
      continue;
    }
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
        out[key] = parseFloat(trimmed);
        continue;
      }
      if (/^\$?-?[\d,]+\.?\d*$/.test(trimmed)) {
        out[key] = parseFloat(trimmed.replace(/[$,]/g, ''));
        continue;
      }
      if (/^-?\d+(\.\d+)?%$/.test(trimmed)) {
        out[key] = parseFloat(trimmed.replace('%', '')) / 100;
        continue;
      }
      out[key] = value;
    } else {
      out[key] = value;
    }
  }
  return out;
}

