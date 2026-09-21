import { describe, it, expect, vi } from 'vitest';
import * as aq from 'arquero';
import {
  quantile,
  pearson,
  spearman,
  rowsOf,
  columnOf,
  listDatasetsTool,
  profileColumnTool,
  aggregateTool,
  filterSampleTool,
  correlateTool,
  runCodeTool,
  executeToolCall,
  AnalysisToolError,
  TOOL_SCHEMAS,
  type AnalysisToolContext,
} from './analysisTools';

describe('analysisTools - stats helpers', () => {
  describe('quantile', () => {
    it('returns median for odd-length array', () => {
      expect(quantile([1, 2, 3, 4, 5], 0.5)).toBe(3);
    });

    it('returns p25 for odd-length array', () => {
      expect(quantile([1, 2, 3, 4, 5], 0.25)).toBe(2);
    });

    it('returns p75 for odd-length array', () => {
      expect(quantile([1, 2, 3, 4, 5], 0.75)).toBe(4);
    });

    it('interpolates for even-length array', () => {
      expect(quantile([1, 2], 0.5)).toBe(1.5);
    });

    it('returns NaN for empty array', () => {
      expect(quantile([], 0.5)).toBeNaN();
    });
  });

  describe('pearson', () => {
    it('returns 1 for perfect positive correlation', () => {
      expect(pearson([1, 2, 3], [2, 4, 6])).toBe(1);
    });

    it('returns -1 for perfect negative correlation', () => {
      expect(pearson([1, 2, 3], [6, 4, 2])).toBe(-1);
    });

    it('returns near 0 for uncorrelated data', () => {
      // xs: [1,2,3,4], ys: [1,4,4,1] - symmetric, Σ(x−x̄)(y−ȳ) = 0 exactly
      const r = pearson([1, 2, 3, 4], [1, 4, 4, 1]);
      expect(Math.abs(r)).toBeLessThan(0.001);
    });

    it('returns 0 when n < 2', () => {
      expect(pearson([1], [2])).toBe(0);
      expect(pearson([], [])).toBe(0);
    });

    it('returns 0 when variance is 0', () => {
      expect(pearson([1, 1, 1], [2, 3, 4])).toBe(0);
      expect(pearson([1, 2, 3], [5, 5, 5])).toBe(0);
    });
  });

  describe('spearman', () => {
    it('returns 1 for perfect monotonic relationship', () => {
      expect(spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])).toBe(1);
    });

    it('handles ties with average ranks', () => {
      // xs: [1, 2, 2, 3] -> ranks: [1, 2.5, 2.5, 4]
      // ys: [10, 20, 20, 30] -> ranks: [1, 2.5, 2.5, 4]
      // pearson on ranks = 1
      expect(spearman([1, 2, 2, 3], [10, 20, 20, 30])).toBe(1);
    });

    it('returns -1 for perfect negative monotonic', () => {
      expect(spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])).toBe(-1);
    });
  });
});

describe('analysisTools - rowsOf', () => {
  it('extracts rows from arquero table', () => {
    const table = aq.from([{ a: 1, b: 'x' }, { a: 2, b: 'y' }]);
    const rows = rowsOf(table);
    expect(rows).toEqual([{ a: 1, b: 'x' }, { a: 2, b: 'y' }]);
  });

  it('passes through plain array', () => {
    const arr = [{ a: 1 }, { a: 2 }];
    expect(rowsOf(arr)).toBe(arr);
  });

  it('throws AnalysisToolError for non-table non-array', () => {
    expect(() => rowsOf(42)).toThrow(AnalysisToolError);
    expect(() => rowsOf(42)).toThrow('not a table or array');
  });

  it('throws AnalysisToolError for null', () => {
    expect(() => rowsOf(null)).toThrow(AnalysisToolError);
  });
});

describe('analysisTools - listDatasetsTool', () => {
  const employees = aq.from([
    { department: 'Engineering', salary_usd: 100000 },
    { department: 'Sales', salary_usd: 80000 },
  ]);
  const projects = aq.from([
    { status: 'Active', budget_usd: 50000 },
    { status: 'Completed', budget_usd: 30000 },
    { status: 'Active', budget_usd: 40000 },
  ]);

  const ctx: AnalysisToolContext = {
    datasets: new Map([
      ['aq', aq],
      ['employees', employees],
      ['projects', projects],
    ]),
    metadata: {
      employees: { columns: ['department', 'salary_usd'], rowCount: 2 },
      projects: { columns: ['status', 'budget_usd'], rowCount: 3 },
    },
  };

  it('returns dataset summaries skipping aq entry', () => {
    const result = listDatasetsTool(ctx);
    expect(result).toHaveLength(2);
    expect(result.map(d => d.name)).toEqual(['employees', 'projects']);
    expect(result.find(d => d.name === 'employees')?.rowCount).toBe(2);
    expect(result.find(d => d.name === 'employees')?.columns).toEqual(['department', 'salary_usd']);
    expect(result.find(d => d.name === 'projects')?.rowCount).toBe(3);
    expect(result.find(d => d.name === 'projects')?.columns).toEqual(['status', 'budget_usd']);
  });
});

describe('analysisTools - profileColumnTool', () => {
  const table = aq.from([
    { id: 1, value: 10, category: 'A' },
    { id: 2, value: 20, category: 'B' },
    { id: 3, value: 30, category: 'A' },
    { id: 4, value: 40, category: 'C' },
    { id: 5, value: 50, category: 'A' },
  ]);

  const ctx: AnalysisToolContext = {
    datasets: new Map([
      ['aq', aq],
      ['data', table],
    ]),
    metadata: {
      data: { columns: ['id', 'value', 'category'], rowCount: 5 },
    },
  };

  it('profiles numeric column with exact stats', () => {
    const result = profileColumnTool(ctx, { dataset: 'data', column: 'value' });
    expect(result.type).toBe('numeric');
    expect(result.count).toBe(5);
    expect(result.missing).toBe(0);
    expect(result.unique).toBe(5);
    expect(result.min).toBe(10);
    expect(result.max).toBe(50);
    expect(result.mean).toBe(30);
    expect(result.median).toBe(30);
    expect(result.p25).toBe(20);
    expect(result.p75).toBe(40);
    expect(result.stddev).toBeCloseTo(Math.sqrt(250), 5); // sample stddev
  });

  it('profiles categorical column with topValues', () => {
    const result = profileColumnTool(ctx, { dataset: 'data', column: 'category' });
    expect(result.type).toBe('categorical');
    expect(result.count).toBe(5);
    expect(result.missing).toBe(0);
    expect(result.unique).toBe(3);
    expect(result.topValues).toHaveLength(3);
    expect(result.topValues[0]).toEqual({ value: 'A', count: 3 });
    expect(result.topValues[1]).toEqual({ value: 'B', count: 1 });
    expect(result.topValues[2]).toEqual({ value: 'C', count: 1 });
  });

  it('counts missing values (null, undefined, empty string)', () => {
    const tableWithMissing = aq.from([
      { v: 1 },
      { v: null },
      { v: undefined },
      { v: '' },
      { v: 5 },
    ]);
    const ctxMissing: AnalysisToolContext = {
      datasets: new Map([['aq', aq], ['data', tableWithMissing]]),
      metadata: { data: { columns: ['v'], rowCount: 5 } },
    };
    const result = profileColumnTool(ctxMissing, { dataset: 'data', column: 'v' });
    expect(result.count).toBe(2); // only 1 and 5 are non-missing
    expect(result.missing).toBe(3);
  });

  it('throws AnalysisToolError for unknown dataset', () => {
    expect(() => profileColumnTool(ctx, { dataset: 'unknown', column: 'value' })).toThrow(AnalysisToolError);
  });

  it('throws AnalysisToolError for unknown column', () => {
    expect(() => profileColumnTool(ctx, { dataset: 'data', column: 'nonexistent' })).toThrow(AnalysisToolError);
  });

  it('on an empty table reports the dataset is empty', () => {
    const emptyTable = aq.from([]);
    const ctxEmpty: AnalysisToolContext = {
      datasets: new Map([['aq', aq], ['data', emptyTable]]),
      metadata: { data: { columns: [], rowCount: 0 } },
    };
    expect(() => profileColumnTool(ctxEmpty, { dataset: 'data', column: 'any' })).toThrow(AnalysisToolError);
    expect(() => profileColumnTool(ctxEmpty, { dataset: 'data', column: 'any' })).toThrow('is empty');
  });
});

describe('analysisTools - aggregateTool', () => {
  const table = aq.from([
    { dept: 'Eng', salary: 100, bonus: 10 },
    { dept: 'Eng', salary: 200, bonus: 20 },
    { dept: 'Sales', salary: 150, bonus: 15 },
    { dept: 'Sales', salary: 250, bonus: 25 },
  ]);

  const ctx: AnalysisToolContext = {
    datasets: new Map([['aq', aq], ['employees', table]]),
    metadata: { employees: { columns: ['dept', 'salary', 'bonus'], rowCount: 4 } },
  };

  it('groups by column and computes sum and count', () => {
    const result = aggregateTool(ctx, {
      dataset: 'employees',
      groupBy: 'dept',
      measures: [
        { column: 'salary', fn: 'sum', as: 'total_salary' },
        { column: 'bonus', fn: 'count', as: 'bonus_count' },
      ],
    });
    expect(result).toHaveLength(2);
    const eng = result.find(r => r.dept === 'Eng');
    const sales = result.find(r => r.dept === 'Sales');
    expect(eng?.total_salary).toBe(300);
    expect(eng?.bonus_count).toBe(2);
    expect(sales?.total_salary).toBe(400);
    expect(sales?.bonus_count).toBe(2);
  });

  it('computes mean, min, max correctly', () => {
    const result = aggregateTool(ctx, {
      dataset: 'employees',
      groupBy: 'dept',
      measures: [
        { column: 'salary', fn: 'mean', as: 'avg_salary' },
        { column: 'salary', fn: 'min', as: 'min_salary' },
        { column: 'salary', fn: 'max', as: 'max_salary' },
      ],
    });
    const eng = result.find(r => r.dept === 'Eng');
    const sales = result.find(r => r.dept === 'Sales');
    expect(eng?.avg_salary).toBe(150);
    expect(eng?.min_salary).toBe(100);
    expect(eng?.max_salary).toBe(200);
    expect(sales?.avg_salary).toBe(200);
    expect(sales?.min_salary).toBe(150);
    expect(sales?.max_salary).toBe(250);
  });

  it('uses default as name when not provided', () => {
    const result = aggregateTool(ctx, {
      dataset: 'employees',
      groupBy: 'dept',
      measures: [{ column: 'salary', fn: 'sum' }],
    });
    expect(result[0]).toHaveProperty('sum_salary');
  });
});

describe('analysisTools - filterSampleTool', () => {
  const table = aq.from([
    { id: 1, name: 'Alice', score: 90 },
    { id: 2, name: 'Bob', score: 80 },
    { id: 3, name: 'Charlie', score: 70 },
    { id: 4, name: 'David', score: 60 },
    { id: 5, name: 'Eve', score: 50 },
    { id: 6, name: 'Frank', score: 40 },
  ]);

  const ctx: AnalysisToolContext = {
    datasets: new Map([['aq', aq], ['students', table]]),
    metadata: { students: { columns: ['id', 'name', 'score'], rowCount: 6 } },
  };

  it('filters with eq operator', () => {
    const result = filterSampleTool(ctx, { dataset: 'students', where: { column: 'name', op: 'eq', value: 'Alice' } });
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe('Alice');
  });

  it('filters with contains operator', () => {
    const result = filterSampleTool(ctx, { dataset: 'students', where: { column: 'name', op: 'contains', value: 'li' } });
    expect(result).toHaveLength(2); // Alice and Charlie both contain 'li'
    expect(result.map(r => r.name).sort()).toEqual(['Alice', 'Charlie']);
  });

  it('filters with gt operator', () => {
    const result = filterSampleTool(ctx, { dataset: 'students', where: { column: 'score', op: 'gt', value: 75 } });
    expect(result).toHaveLength(2);
    expect(result.map(r => r.score)).toEqual([90, 80]);
  });

  it('filters with lt operator', () => {
    const result = filterSampleTool(ctx, { dataset: 'students', where: { column: 'score', op: 'lt', value: 65 } });
    expect(result).toHaveLength(3); // 60, 50, 40 are all < 65
    expect(result.map(r => r.score).sort((a, b) => b - a)).toEqual([60, 50, 40]);
  });

  it('defaults limit to 5', () => {
    const bigTable = aq.from(Array.from({ length: 100 }, (_, i) => ({ id: i, val: i })));
    const bigCtx: AnalysisToolContext = {
      datasets: new Map([['aq', aq], ['big', bigTable]]),
      metadata: { big: { columns: ['id', 'val'], rowCount: 100 } },
    };
    const result = filterSampleTool(bigCtx, { dataset: 'big' });
    expect(result).toHaveLength(5);
  });

  it('caps limit at 50', () => {
    const bigTable = aq.from(Array.from({ length: 100 }, (_, i) => ({ id: i, val: i })));
    const bigCtx: AnalysisToolContext = {
      datasets: new Map([['aq', aq], ['big', bigTable]]),
      metadata: { big: { columns: ['id', 'val'], rowCount: 100 } },
    };
    const result = filterSampleTool(bigCtx, { dataset: 'big', limit: 100 });
    expect(result).toHaveLength(50);
  });

  it('returns all rows when no where clause', () => {
    const result = filterSampleTool(ctx, { dataset: 'students', limit: 10 });
    expect(result).toHaveLength(6);
  });
});

describe('analysisTools - correlateTool', () => {
  const table = aq.from([
    { x: 1, y: 2 },
    { x: 2, y: 4 },
    { x: 3, y: 6 },
    { x: 4, y: 8 },
    { x: 5, y: 10 },
  ]);

  const ctx: AnalysisToolContext = {
    datasets: new Map([['aq', aq], ['data', table]]),
    metadata: { data: { columns: ['x', 'y'], rowCount: 5 } },
  };

  it('computes pearson correlation by default', () => {
    const result = correlateTool(ctx, { dataset: 'data', columnA: 'x', columnB: 'y' });
    expect(result.correlation).toBe(1);
    expect(result.method).toBe('pearson');
    expect(result.n).toBe(5);
  });

  it('computes spearman correlation when specified', () => {
    const result = correlateTool(ctx, { dataset: 'data', columnA: 'x', columnB: 'y', method: 'spearman' });
    expect(result.correlation).toBe(1);
    expect(result.method).toBe('spearman');
    expect(result.n).toBe(5);
  });

  it('throws AnalysisToolError for unknown dataset', () => {
    expect(() => correlateTool(ctx, { dataset: 'unknown', columnA: 'x', columnB: 'y' })).toThrow(AnalysisToolError);
  });

  it('throws AnalysisToolError for unknown column', () => {
    expect(() => correlateTool(ctx, { dataset: 'data', columnA: 'x', columnB: 'unknown' })).toThrow(AnalysisToolError);
  });

  it('pairs values row-wise when a column has missing values', () => {
    // columnA: [1, null, 3, 4], columnB: [10, 20, 30, 40]
    // True pairs (both numeric): (1,10), (3,30), (4,40) -> n=3
    // Buggy independent filtering would pair: (1,10), (3,20), (4,30) -> wrong correlation
    const tableWithMissing = aq.from([
      { a: 1, b: 10 },
      { a: null, b: 20 },
      { a: 3, b: 30 },
      { a: 4, b: 40 },
    ]);
    const ctxMissing: AnalysisToolContext = {
      datasets: new Map([['aq', aq], ['data', tableWithMissing]]),
      metadata: { data: { columns: ['a', 'b'], rowCount: 4 } },
    };
    const result = correlateTool(ctxMissing, { dataset: 'data', columnA: 'a', columnB: 'b' });
    // Expected correlation of [(1,10), (3,30), (4,40)]
    const expected = pearson([1, 3, 4], [10, 30, 40]);
    expect(result.correlation).toBeCloseTo(expected, 10);
    expect(result.n).toBe(3);
  });

  it('throws when no rows have both columns numeric', () => {
    const tableNoPairs = aq.from([
      { a: 1, b: 'x' },
      { a: 'y', b: 2 },
    ]);
    const ctxNoPairs: AnalysisToolContext = {
      datasets: new Map([['aq', aq], ['data', tableNoPairs]]),
      metadata: { data: { columns: ['a', 'b'], rowCount: 2 } },
    };
    expect(() => correlateTool(ctxNoPairs, { dataset: 'data', columnA: 'a', columnB: 'b' })).toThrow(AnalysisToolError);
    expect(() => correlateTool(ctxNoPairs, { dataset: 'data', columnA: 'a', columnB: 'b' })).toThrow('no rows where both columns have numeric values');
  });
});

describe('analysisTools - runCodeTool', () => {
  it('executes code via executor and returns result', () => {
    const executor = vi.fn((code: string) => {
      if (code.includes('return 42')) return 42;
      return undefined;
    });
    const result = runCodeTool({ code: 'return 42' }, executor);
    expect(result).toBe(42);
    expect(executor).toHaveBeenCalledWith('return 42');
  });
});

describe('analysisTools - executeToolCall', () => {
  const table = aq.from([{ a: 1, b: 2 }]);
  const ctx: AnalysisToolContext = {
    datasets: new Map([['aq', aq], ['data', table]]),
    metadata: { data: { columns: ['a', 'b'], rowCount: 1 } },
  };

  const executor = vi.fn((code: string) => `executed: ${code}`);

  it('dispatches list_datasets', () => {
    const result = executeToolCall(ctx, 'list_datasets', {}, executor);
    expect(Array.isArray(result)).toBe(true);
    expect(result).toHaveLength(1);
  });

  it('dispatches profile_column', () => {
    const result = executeToolCall(ctx, 'profile_column', { dataset: 'data', column: 'a' }, executor);
    expect(result).toHaveProperty('type');
  });

  it('dispatches aggregate', () => {
    const result = executeToolCall(ctx, 'aggregate', {
      dataset: 'data',
      groupBy: 'a',
      measures: [{ column: 'b', fn: 'sum' }],
    }, executor);
    expect(Array.isArray(result)).toBe(true);
  });

  it('dispatches filter_sample', () => {
    const result = executeToolCall(ctx, 'filter_sample', { dataset: 'data' }, executor);
    expect(Array.isArray(result)).toBe(true);
  });

  it('dispatches correlate', () => {
    const result = executeToolCall(ctx, 'correlate', { dataset: 'data', columnA: 'a', columnB: 'b' }, executor);
    expect(result).toHaveProperty('correlation');
  });

  it('dispatches run_code', () => {
    const result = executeToolCall(ctx, 'run_code', { code: '1+1' }, executor);
    expect(result).toBe('executed: 1+1');
  });

  it('throws AnalysisToolError for unknown tool name', () => {
    expect(() => executeToolCall(ctx, 'unknown_tool', {}, executor)).toThrow(AnalysisToolError);
  });

  it('throws AnalysisToolError for bad args (missing dataset)', () => {
    expect(() => executeToolCall(ctx, 'profile_column', { column: 'a' }, executor)).toThrow(AnalysisToolError);
  });

  it('throws AnalysisToolError for bad args (unknown dataset)', () => {
    expect(() => executeToolCall(ctx, 'profile_column', { dataset: 'unknown', column: 'a' }, executor)).toThrow(AnalysisToolError);
  });

  it('throws AnalysisToolError for bad args (empty measures)', () => {
    expect(() => executeToolCall(ctx, 'aggregate', { dataset: 'data', groupBy: 'a', measures: [] }, executor)).toThrow(AnalysisToolError);
  });

  it('throws AnalysisToolError for bad args (empty code)', () => {
    expect(() => executeToolCall(ctx, 'run_code', { code: '' }, executor)).toThrow(AnalysisToolError);
  });
});

describe('analysisTools - TOOL_SCHEMAS', () => {
  it('has 6 tool definitions', () => {
    expect(TOOL_SCHEMAS).toHaveLength(6);
  });

  it('each schema has name, description, and parameters', () => {
    for (const schema of TOOL_SCHEMAS) {
      expect(schema.type).toBe('function');
      expect(typeof schema.function.name).toBe('string');
      expect(typeof schema.function.description).toBe('string');
      expect(schema.function.parameters).toBeDefined();
      expect(schema.function.parameters.type).toBe('object');
    }
  });

  it('includes all expected tool names', () => {
    const names = TOOL_SCHEMAS.map(s => s.function.name).sort();
    expect(names).toEqual([
      'aggregate',
      'correlate',
      'filter_sample',
      'list_datasets',
      'profile_column',
      'run_code',
    ]);
  });
});

describe('analysisTools - columnOf', () => {
  const table = aq.from([
    { a: 1, b: 'x' },
    { a: 2, b: 'y' },
    { a: 3, b: 'z' },
  ]);

  it('extracts column from arquero table via array()', () => {
    const result = columnOf(table, 'a');
    expect(result).toEqual([1, 2, 3]);
  });

  it('extracts column from plain array of row objects', () => {
    const plainArray = [{ a: 10 }, { a: 20 }, { a: 30 }];
    const result = columnOf(plainArray, 'a');
    expect(result).toEqual([10, 20, 30]);
  });

  it('throws AnalysisToolError for non-table non-array input', () => {
    expect(() => columnOf(42, 'a')).toThrow(AnalysisToolError);
    expect(() => columnOf(42, 'a')).toThrow('not a table or array');
  });

  it('throws AnalysisToolError for null input', () => {
    expect(() => columnOf(null, 'a')).toThrow(AnalysisToolError);
  });
});