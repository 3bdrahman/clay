import { AnalysisToolError } from './errors/analysisErrors';
import type { AnalysisToolContext } from './types';
import { quantile, pearson, spearman, isFiniteNumber, coerceNumber } from './statistics';
import { MAX_FILTER_LIMIT, DEFAULT_FILTER_LIMIT } from './analysisToolSchemas';

// Re-export public API
export type { AnalysisToolContext } from './types';
export { AnalysisToolError } from './errors/analysisErrors';
export { quantile, pearson, spearman } from './statistics';
export { TOOL_SCHEMAS, MAX_FILTER_LIMIT, DEFAULT_FILTER_LIMIT } from './analysisToolSchemas';
export { executeToolCall } from './analysisToolDispatch';

// ============================================================================
// Internal helpers (not re-exported)
// ============================================================================

export function rowsOf(table: unknown): Array<Record<string, unknown>> {
  const t = table as { objects?: () => unknown[] } | undefined;
  if (t && typeof t.objects === 'function') {
    return t.objects() as Array<Record<string, unknown>>;
  }
  if (Array.isArray(table)) {
    return table as Array<Record<string, unknown>>;
  }
  throw new AnalysisToolError('rowsOf', 'not a table or array');
}

export function columnOf(table: unknown, column: string): unknown[] {
  const t = table as { array?: (col: string) => unknown[] } | undefined;
  if (t && typeof t.array === 'function') {
    return t.array(column);
  }
  const rows = rowsOf(table);
  return rows.map(r => r[column]);
}

function getTable(ctx: AnalysisToolContext, dataset: string): unknown {
  const table = ctx.datasets.get(dataset);
  if (!table) {
    throw new AnalysisToolError('getTable', `unknown dataset: ${dataset}`);
  }
  return table;
}

function getColumnValues(table: unknown, column: string): unknown[] {
  return columnOf(table, column);
}

// ============================================================================
// Tool implementations
// ============================================================================

export function listDatasetsTool(ctx: AnalysisToolContext): Array<{ name: string; rowCount: number; columns: string[] }> {
  const result: Array<{ name: string; rowCount: number; columns: string[] }> = [];
  for (const [name, table] of ctx.datasets) {
    if (name === 'aq') continue;
    const meta = ctx.metadata[name];
    let rowCount = meta?.rowCount ?? 0;
    let columns = meta?.columns ?? [];
    if (table && typeof (table as { numRows?: () => number }).numRows === 'function') {
      try {
        rowCount = (table as { numRows: () => number }).numRows();
        const names = (table as { columnNames?: () => string[] }).columnNames?.();
        if (Array.isArray(names) && names.length > 0) columns = names;
      } catch (e) {
        if (import.meta.env.DEV) {
          console.warn('[analysisTools] listDatasetsTool live-table inspection failed (falling back to metadata):', e);
        }
        rowCount = meta?.rowCount ?? 0;
      }
    }
    result.push({ name, rowCount, columns });
  }
  return result;
}

export function profileColumnTool(
  ctx: AnalysisToolContext,
  args: { dataset: string; column: string }
): object {
  const table = getTable(ctx, args.dataset);
  const rows = rowsOf(table);
  if (rows.length > 0 && !(args.column in rows[0]!)) {
    throw new AnalysisToolError('profile_column', `column "${args.column}" not found in dataset "${args.dataset}"`);
  }
  const values = getColumnValues(table, args.column);
  if (values.length === 0) {
    throw new AnalysisToolError('profile_column', `dataset "${args.dataset}" is empty`);
  }

  let count = 0;
  let missing = 0;
  const uniqueSet = new Set<unknown>();
  const numericValues: number[] = [];
  const freqMap = new Map<unknown, number>();

  for (const v of values) {
    const isMissing = v === null || v === undefined || v === '';
    if (isMissing) {
      missing++;
    } else {
      count++;
      uniqueSet.add(v);
      freqMap.set(v, (freqMap.get(v) ?? 0) + 1);
      if (isFiniteNumber(v)) numericValues.push(v);
    }
  }

  const unique = uniqueSet.size;
  const numericRatio = count > 0 ? numericValues.length / count : 0;

  if (numericRatio >= 0.6 && numericValues.length > 0) {
    numericValues.sort((a, b) => a - b);
    const n = numericValues.length;
    const sum = numericValues.reduce((a, b) => a + b, 0);
    const mean = sum / n;
    let varianceSum = 0;
    for (const v of numericValues) {
      const d = v - mean;
      varianceSum += d * d;
    }
    const stddev = n >= 2 ? Math.sqrt(varianceSum / (n - 1)) : undefined;
    return {
      type: 'numeric',
      count,
      missing,
      unique,
      ...(numericValues.length < count ? { nonNumeric: count - numericValues.length } : {}),
      min: numericValues[0]!,
      max: numericValues[n - 1]!,
      mean,
      median: quantile(numericValues, 0.5),
      p25: quantile(numericValues, 0.25),
      p75: quantile(numericValues, 0.75),
      stddev,
    };
  }

  const topValues = Array.from(freqMap.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([value, count]) => ({ value, count }));

  return {
    type: 'categorical',
    count,
    missing,
    unique,
    topValues,
  };
}

export function aggregateTool(
  ctx: AnalysisToolContext,
  args: {
    dataset: string;
    groupBy: string;
    measures: Array<{ column: string; fn: 'sum' | 'mean' | 'count' | 'min' | 'max'; as?: string }>;
  }
): Array<Record<string, unknown>> {
  const table = getTable(ctx, args.dataset);
  const rows = rowsOf(table);
  const groups = new Map<string, Array<Record<string, unknown>>>();

  for (const row of rows) {
    const key = String(row[args.groupBy] ?? '');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(row);
  }

  const result: Array<Record<string, unknown>> = [];
  for (const [groupKey, groupRows] of groups) {
    const out: Record<string, unknown> = { [args.groupBy]: groupKey };
    for (const m of args.measures) {
      const vals = groupRows
        .map(r => r[m.column])
        .filter(v => v !== null && v !== undefined && v !== '');
      const asName = m.as ?? `${m.fn}_${m.column}`;
      let computed: unknown;
      switch (m.fn) {
        case 'count':
          computed = vals.length;
          break;
        case 'sum': {
          const nums = vals.filter(isFiniteNumber);
          computed = nums.reduce((a, b) => a + b, 0);
          break;
        }
        case 'mean': {
          const nums = vals.filter(isFiniteNumber);
          computed = nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
          break;
        }
        case 'min': {
          const nums = vals.filter(isFiniteNumber);
          computed = nums.length > 0 ? Math.min(...nums) : null;
          break;
        }
        case 'max': {
          const nums = vals.filter(isFiniteNumber);
          computed = nums.length > 0 ? Math.max(...nums) : null;
          break;
        }
      }
      out[asName] = computed;
    }
    result.push(out);
  }
  return result;
}

export function filterSampleTool(
  ctx: AnalysisToolContext,
  args: { dataset: string; where?: { column: string; op: 'eq' | 'contains' | 'gt' | 'lt'; value: unknown }; limit?: number }
): Array<Record<string, unknown>> {
  const table = getTable(ctx, args.dataset);
  const rows = rowsOf(table);
  let filtered = rows;

  if (args.where) {
    const { column, op, value } = args.where;
    filtered = rows.filter(row => {
      const cell = row[column];
      switch (op) {
        case 'eq': {
          if (cell === value) return true;
          // The LLM passes JSON values: a numeric-looking string never
          // strict-matches a numeric cell, which reads as "no rows match".
          // Coerce both sides when both are numeric-looking.
          const a = coerceNumber(cell);
          const b = coerceNumber(value);
          return a !== undefined && b !== undefined && a === b;
        }
        case 'contains':
          return String(cell ?? '').includes(String(value));
        case 'gt': {
          const a = coerceNumber(cell);
          const b = coerceNumber(value);
          return a !== undefined && b !== undefined && a > b;
        }
        case 'lt': {
          const a = coerceNumber(cell);
          const b = coerceNumber(value);
          return a !== undefined && b !== undefined && a < b;
        }
        default:
          return false;
      }
    });
  }

  const limit = Math.min(args.limit ?? DEFAULT_FILTER_LIMIT, MAX_FILTER_LIMIT);
  return filtered.slice(0, limit);
}

export function correlateTool(
  ctx: AnalysisToolContext,
  args: { dataset: string; columnA: string; columnB: string; method?: 'pearson' | 'spearman' }
): { correlation: number; method: string; n: number } {
  const table = getTable(ctx, args.dataset);
  const rows = rowsOf(table);
  const pairs: Array<[number, number]> = [];
  for (const row of rows) {
    const a = row[args.columnA];
    const b = row[args.columnB];
    if (isFiniteNumber(a) && isFiniteNumber(b)) pairs.push([a, b]);
  }
  if (pairs.length === 0) {
    throw new AnalysisToolError('correlate', 'no rows where both columns have numeric values');
  }
  const valsA = pairs.map(p => p[0]);
  const valsB = pairs.map(p => p[1]);
  const n = pairs.length;
  const method = args.method ?? 'pearson';
  const correlation = method === 'spearman' ? spearman(valsA, valsB) : pearson(valsA, valsB);
  return { correlation, method, n };
}

export async function runCodeTool(args: { code: string }, executor: (code: string) => Promise<unknown>): Promise<unknown> {
  return executor(args.code);
}