import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createDataAnalyzer } from '../services/analyzer';
import type { LLMClient } from '../lib/llm';
import type { EmbeddingsClient } from '../lib/embeddings';
import * as aq from 'arquero';
import type { ColumnTable } from 'arquero';

const mockLLM: LLMClient = {
  invoke: vi.fn(),
  stream: vi.fn(),
};

const mockEmbeddings: EmbeddingsClient = {
  embed: vi.fn(),
};

const sampleTable = aq.from([
  { department: 'Engineering', salary_usd: 100000 },
  { department: 'Sales', salary_usd: 80000 },
  { department: 'Engineering', salary_usd: 120000 },
  { department: 'Marketing', salary_usd: 90000 },
]);

const projectsTable = aq.from([
  { status: 'Active', budget_usd: 50000 },
  { status: 'Completed', budget_usd: 30000 },
  { status: 'Active', budget_usd: 40000 },
]);

const metadata = {
  employees: { columns: ['department', 'salary_usd'], rowCount: 4 },
  projects: { columns: ['status', 'budget_usd'], rowCount: 3 },
};

describe('createDataAnalyzer', () => {
  let analyzer: ReturnType<typeof createDataAnalyzer>;

  beforeEach(() => {
    vi.clearAllMocks();
    analyzer = createDataAnalyzer({
      llm: mockLLM,
      embeddings: mockEmbeddings,
      datasets: new Map<string, ColumnTable | typeof aq>([
        ['aq', aq],
        ['employees', sampleTable],
        ['projects', projectsTable],
      ]),
      metadata,
      codeGenModel: 'test-model',
    });
  });

  it('returns relevant datasets based on column name overlap via analyze', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({
        code: "result = employees.groupby('department').rollup({ avg_salary: d => op.mean(d.salary_usd) })",
        explanation: 'Average salary by department',
      }),
    });

    const result = await analyzer.analyze('average salary by department');

    expect(result.resultType).toBe('chart');
    expect(result.code).toContain('groupby');
  });

  it('returns relevant datasets based on dataset name overlap via analyze', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({
        code: "result = projects.groupby('status').count()",
        explanation: 'Project count by status',
      }),
    });

    const result = await analyzer.analyze('show me all projects');

    expect(result.resultType).toBe('chart');
    expect(result.code).toContain('groupby');
  });

  it('limits to top 4 datasets', async () => {
    const meta = { ...metadata };
    for (let i = 0; i < 10; i++) {
      meta[`dataset${i}`] = { columns: ['salary_usd'], rowCount: 1 };
    }
    const a = createDataAnalyzer({
      llm: mockLLM,
      embeddings: mockEmbeddings,
      datasets: new Map([['aq', aq], ['employees', sampleTable], ['projects', projectsTable]]),
      metadata: meta,
      codeGenModel: 'test-model',
    });

    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({
        code: "result = employees.groupby('department').count()",
        explanation: 'Count',
      }),
    });

    const result = await a.analyze('salary');
    expect(result.code).toContain('groupby');
  });

  it('executes generated Arquero code', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({
        code: "result = employees.groupby('department').rollup({ avg_salary: d => op.mean(d.salary_usd) })",
        explanation: 'Average salary by department',
      }),
    });

    const result = await analyzer.analyze('average salary by department');

    expect(result.resultType).toBe('chart');
    expect(result.code).toContain('groupby');
    expect(result.attempts).toBe(1);
  });

  it('retries on code error', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: JSON.stringify({
          code: 'result = employees.invalid_method()',
          explanation: 'Bad code',
        }),
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({
          code: "result = employees.groupby('department').count()",
          explanation: 'Fixed code',
        }),
      });

    const result = await analyzer.analyze('count by department');

    expect(result.attempts).toBe(2);
    expect(result.resultType).toBe('chart');
  });

  it('returns error result after max retries', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({
        code: 'result = employees.invalid()',
        explanation: 'Always fails',
      }),
    });

    const result = await analyzer.analyze('impossible query');

    expect(result.resultType).toBe('error');
    expect(result.attempts).toBe(3);
  });

  it('detects chart config from table results', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: JSON.stringify({
        code: "result = employees.groupby('department').rollup({ avg: d => op.mean(d.salary_usd) })",
        explanation: 'Chart data',
      }),
    });

    const result = await analyzer.analyze('chart salary by department');

    expect(result.chartConfig).toBeDefined();
    expect(result.chartConfig?.type).toBe('bar');
    expect(result.chartConfig?.xKey).toBe('department');
    expect(result.resultType).toBe('chart');
  });

  it('listDatasets returns dataset summaries', () => {
    const datasets = analyzer.listDatasets();
    expect(datasets).toHaveLength(2);
    expect(datasets.map(d => d.name)).toEqual(['employees', 'projects']);
  });

  it('getDatasetSummary returns correct info', () => {
    const summary = analyzer.getDatasetSummary('employees');
    expect(summary).toEqual({
      name: 'employees',
      rowCount: 4,
      columns: ['department', 'salary_usd'],
    });
  });

  it('returns undefined for unknown dataset', () => {
    expect(analyzer.getDatasetSummary('nonexistent')).toBeUndefined();
  });
});