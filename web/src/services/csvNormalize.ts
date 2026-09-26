// CSV parsing with type normalization — the single seam every CSV table
// flows through (user uploads, sample datasets, sandbox rehydration), so a
// dataset's column types never depend on which path loaded it.

import * as aq from 'arquero';
import type { ColumnTable } from 'arquero';

// Matches unambiguous currency amounts in data rows (not the header):
// $1234.56, $1,234.56, or 1,234,567.89 (thousands separators). Quoted fields
// with prose commas are unaffected — the pattern needs digit groups of
// exactly three after the comma.
const CURRENCY_REGEX = /\$[\d,]+\.?\d*|\b\d{1,3}(,\d{3})+\.?\d*\b/g;

export function normalizeRow(row: Record<string, unknown>): Record<string, unknown> {
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

export function parseCsvNormalized(csv: string): ColumnTable {
  const lines = csv.split('\n');
  // Only data lines get currency preprocessing — the header keeps its names.
  const processed = lines.length < 2
    ? csv
    : [lines[0] ?? '', ...lines.slice(1).map(line => line.replace(CURRENCY_REGEX, (match) => match.replace(/[$,]/g, '')))].join('\n');
  const table = aq.fromCSV(processed);
  const rows = table.objects() as Array<Record<string, unknown>>;
  return aq.from(rows.map(normalizeRow));
}
