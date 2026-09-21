import { RagError, RagErrorCode } from './errors';
import type { ToolDefinition } from './types';

export interface AnalysisToolContext {
  datasets: Map<string, unknown>;
  metadata: Record<string, { columns: string[]; rowCount: number }>;
}

export class AnalysisToolError extends RagError {
  constructor(tool: string, problem: string, cause?: Error) {
    super({
      code: RagErrorCode.UNKNOWN_ERROR,
      message: `Analysis tool "${tool}" failed: ${problem}`,
      cause,
      retryable: false,
      context: { tool, problem },
    });
  }
}

const MAX_FILTER_LIMIT = 50;
const DEFAULT_FILTER_LIMIT = 5;

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

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

export function quantile(sortedValues: number[], p: number): number {
  const n = sortedValues.length;
  if (n === 0) return NaN;
  const idx = p * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedValues[lo]!;
  return sortedValues[lo]! + (sortedValues[hi]! - sortedValues[lo]!) * (idx - lo);
}

export function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2 || n !== ys.length) return 0;
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i]!;
    sumY += ys[i]!;
  }
  const meanX = sumX / n;
  const meanY = sumY / n;
  let num = 0;
  let denX = 0;
  let denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i]! - meanX;
    const dy = ys[i]! - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  if (denX === 0 || denY === 0) return 0;
  return num / Math.sqrt(denX * denY);
}

function rankWithAverageTies(values: number[]): number[] {
  const n = values.length;
  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array<number>(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n && indexed[j]!.v === indexed[i]!.v) j++;
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) {
      ranks[indexed[k]!.i] = avgRank;
    }
    i = j;
  }
  return ranks;
}

export function spearman(xs: number[], ys: number[]): number {
  if (xs.length !== ys.length) return 0;
  const rx = rankWithAverageTies(xs);
  const ry = rankWithAverageTies(ys);
  return pearson(rx, ry);
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
        case 'sum':
          computed = vals.reduce((a, b) => (isFiniteNumber(a) ? a : 0) + (isFiniteNumber(b) ? b : 0), 0);
          break;
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
        case 'eq':
          return cell === value;
        case 'contains':
          return String(cell ?? '').includes(String(value));
        case 'gt':
          return isFiniteNumber(cell) && isFiniteNumber(value) && cell > value;
        case 'lt':
          return isFiniteNumber(cell) && isFiniteNumber(value) && cell < value;
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

export function runCodeTool(args: { code: string }, executor: (code: string) => unknown): unknown {
  return executor(args.code);
}

export const TOOL_SCHEMAS: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'list_datasets',
      description: 'List all available datasets with their row counts and column names',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'profile_column',
      description: 'Get statistical profile of a single column (numeric or categorical)',
      parameters: {
        type: 'object',
        properties: {
          dataset: { type: 'string', description: 'Dataset name' },
          column: { type: 'string', description: 'Column name' },
        },
        required: ['dataset', 'column'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'aggregate',
      description: 'Group rows by a column and compute aggregate measures',
      parameters: {
        type: 'object',
        properties: {
          dataset: { type: 'string', description: 'Dataset name' },
          groupBy: { type: 'string', description: 'Column to group by' },
          measures: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                column: { type: 'string' },
                fn: { type: 'string', enum: ['sum', 'mean', 'count', 'min', 'max'] },
                as: { type: 'string' },
              },
              required: ['column', 'fn'],
            },
            minItems: 1,
          },
        },
        required: ['dataset', 'groupBy', 'measures'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'filter_sample',
      description: 'Filter rows and return a sample (default 5, max 50)',
      parameters: {
        type: 'object',
        properties: {
          dataset: { type: 'string', description: 'Dataset name' },
          where: {
            type: 'object',
            properties: {
              column: { type: 'string' },
              op: { type: 'string', enum: ['eq', 'contains', 'gt', 'lt'] },
              value: {},
            },
            required: ['column', 'op', 'value'],
          },
          limit: { type: 'number', minimum: 1, maximum: MAX_FILTER_LIMIT },
        },
        required: ['dataset'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'correlate',
      description: 'Compute correlation between two numeric columns',
      parameters: {
        type: 'object',
        properties: {
          dataset: { type: 'string', description: 'Dataset name' },
          columnA: { type: 'string', description: 'First column' },
          columnB: { type: 'string', description: 'Second column' },
          method: { type: 'string', enum: ['pearson', 'spearman'], default: 'pearson' },
        },
        required: ['dataset', 'columnA', 'columnB'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_code',
      description: 'Execute arbitrary Arquero code in the sandbox',
      parameters: {
        type: 'object',
        properties: {
          code: { type: 'string', description: 'Arquero code to execute' },
        },
        required: ['code'],
      },
    },
  },
];

export function executeToolCall(
  ctx: AnalysisToolContext,
  name: string,
  args: Record<string, unknown>,
  executor?: (code: string) => unknown
): unknown {
  switch (name) {
    case 'list_datasets':
      return listDatasetsTool(ctx);
    case 'profile_column': {
      if (!args.dataset || typeof args.dataset !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'profile_column requires dataset string');
      }
      if (!args.column || typeof args.column !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'profile_column requires column string');
      }
      return profileColumnTool(ctx, { dataset: args.dataset, column: args.column });
    }
    case 'aggregate': {
      if (!args.dataset || typeof args.dataset !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'aggregate requires dataset string');
      }
      if (!args.groupBy || typeof args.groupBy !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'aggregate requires groupBy string');
      }
      if (!Array.isArray(args.measures) || args.measures.length === 0) {
        throw new AnalysisToolError('executeToolCall', 'aggregate requires non-empty measures array');
      }
      return aggregateTool(ctx, {
        dataset: args.dataset,
        groupBy: args.groupBy,
        measures: args.measures as Array<{ column: string; fn: 'sum' | 'mean' | 'count' | 'min' | 'max'; as?: string }>,
      });
    }
    case 'filter_sample': {
      if (!args.dataset || typeof args.dataset !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'filter_sample requires dataset string');
      }
      return filterSampleTool(ctx, {
        dataset: args.dataset,
        where: args.where as { column: string; op: 'eq' | 'contains' | 'gt' | 'lt'; value: unknown } | undefined,
        limit: typeof args.limit === 'number' ? args.limit : undefined,
      });
    }
    case 'correlate': {
      if (!args.dataset || typeof args.dataset !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'correlate requires dataset string');
      }
      if (!args.columnA || typeof args.columnA !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'correlate requires columnA string');
      }
      if (!args.columnB || typeof args.columnB !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'correlate requires columnB string');
      }
      return correlateTool(ctx, {
        dataset: args.dataset,
        columnA: args.columnA,
        columnB: args.columnB,
        method: args.method as 'pearson' | 'spearman' | undefined,
      });
    }
    case 'run_code': {
      if (!args.code || typeof args.code !== 'string' || args.code.trim() === '') {
        throw new AnalysisToolError('executeToolCall', 'run_code requires non-empty code string');
      }
      if (!executor) {
        throw new AnalysisToolError('executeToolCall', 'run_code requires executor function');
      }
      return runCodeTool({ code: args.code }, executor);
    }
    default:
      throw new AnalysisToolError('executeToolCall', `unknown tool: ${name}`);
  }
}