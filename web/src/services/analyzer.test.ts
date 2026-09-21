import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createDataAnalyzer } from '../services/analyzer';
import type { LLMClient } from '../lib/llm';
import type { EmbeddingsClient } from '../lib/embeddings';
import * as aq from 'arquero';
import type { ColumnTable } from 'arquero';
import { AnalysisBudgetExceededError, RateLimitError } from '../lib/errors';

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

function createAnalyzer(overrides: Partial<{ maxToolLoopTokens: number }> = {}) {
  return createDataAnalyzer({
    llm: mockLLM,
    embeddings: mockEmbeddings,
    datasets: new Map<string, ColumnTable | typeof aq>([
      ['aq', aq],
      ['employees', sampleTable],
      ['projects', projectsTable],
    ]),
    metadata,
    codeGenModel: 'test-model',
    maxToolLoopTokens: overrides.maxToolLoopTokens,
  });
}

describe('createDataAnalyzer', () => {
  let analyzer: ReturnType<typeof createDataAnalyzer>;

  beforeEach(() => {
    vi.clearAllMocks();
    analyzer = createAnalyzer();
  });

  // ============================================================================
  // Existing single-shot behavior tests (adapted for tool-loop)
  // ============================================================================

  it('returns relevant datasets based on column name overlap via analyze', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [
          { id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } },
        ],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({
          answer: 'Average salary by department: Engineering 110000, Sales 80000, Marketing 90000',
          insights: [{ finding: 'Engineering highest', evidence: 'avg 110k', confidence: 'high' }],
        }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('average salary by department');

    expect(result.resultType).toBe('scalar'); // no chart in synthesis
    expect(result.insights).toHaveLength(1);
    expect(result.insights?.[0].finding).toBe('Engineering highest');
  });

  it('returns relevant datasets based on dataset name overlap via analyze', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [
          { id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } },
        ],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({
          answer: 'Projects: 2 Active, 1 Completed',
          insights: [{ finding: 'Most projects active', evidence: '2 of 3', confidence: 'high' }],
        }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('show me all projects');

    expect(result.insights).toHaveLength(1);
    expect(result.insights?.[0].finding).toBe('Most projects active');
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

    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ answer: 'Count by department', insights: [] }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await a.analyze('salary');
    expect(result.explanation).toBe('Count by department');
  });

  it('executes generated Arquero code via run_code tool', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [
          { id: 'c1', type: 'function', function: { name: 'run_code', arguments: JSON.stringify({ code: "result = employees.groupby('department').rollup({ avg_salary: d => op.mean(d.salary_usd) })" }) } },
        ],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ answer: 'Average salary by department', insights: [] }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('average salary by department');

    expect(result.explanation).toBe('Average salary by department');
    expect(result.attempts).toBe(2); // 2 iterations (tool call + synthesis)
  });

  it('returns the user question in result.question, not the generated code', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ answer: 'Count by department', insights: [] }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('count employees by department');

    expect(result.question).toBe('count employees by department');
  });

  it('retries on tool error', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [
          { id: 'c1', type: 'function', function: { name: 'run_code', arguments: JSON.stringify({ code: 'result = employees.invalid_method()' }) } },
        ],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [
          { id: 'c2', type: 'function', function: { name: 'run_code', arguments: JSON.stringify({ code: "result = employees.groupby('department').count()" }) } },
        ],
        finishReason: 'tool_calls',
        usage: { totalTokens: 200 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ answer: 'Fixed count', insights: [] }),
        finishReason: 'stop',
        usage: { totalTokens: 300 },
      });

    const result = await analyzer.analyze('count by department');

    expect(result.attempts).toBe(3);
    expect(result.explanation).toBe('Fixed count');
  });

  it('throws AnalysisBudgetExceededError after max tool iterations', async () => {
    // Always return tool calls that fail
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: '',
      toolCalls: [
        { id: 'c1', type: 'function', function: { name: 'run_code', arguments: JSON.stringify({ code: 'result = employees.invalid()' }) } },
      ],
      finishReason: 'tool_calls',
      usage: { totalTokens: 100 },
    });

    await expect(analyzer.analyze('impossible query')).rejects.toThrow(AnalysisBudgetExceededError);

    try {
      await analyzer.analyze('impossible query');
    } catch (e) {
      expect(e).toBeInstanceOf(AnalysisBudgetExceededError);
      const err = e as AnalysisBudgetExceededError;
      expect(err.context?.tripped).toBe('iterations');
      expect(err.context?.limit).toBe(8);
    }
  });

  it('detects chart config from synthesis', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({
          answer: 'Chart data',
          insights: [],
          chart: { type: 'bar', title: 'Salary by Dept', xKey: 'department', yKeys: ['avg_salary'], data: [{ department: 'Engineering', avg_salary: 110000 }] },
        }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('chart salary by department');

    expect(result.chartConfig).toBeDefined();
    expect(result.chartConfig?.type).toBe('bar');
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

  // ============================================================================
  // New tool-loop tests (FAILING-FIRST)
  // ============================================================================

  it('tool loop executes tools and returns synthesis insights', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c2', type: 'function', function: { name: 'profile_column', arguments: JSON.stringify({ dataset: 'employees', column: 'salary_usd' }) } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 200 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({
          answer: 'The answer',
          insights: [{ finding: 'F1', evidence: 'E1', confidence: 'high' }],
          chart: { type: 'bar', title: 'T', xKey: 'name', yKeys: ['value'], data: [{ name: 'a', value: 1 }] },
        }),
        finishReason: 'stop',
        usage: { totalTokens: 300 },
      });

    const result = await analyzer.analyze('test question');

    expect(result.insights).toHaveLength(1);
    expect(result.insights?.[0].finding).toBe('F1');
    expect(result.chartConfig).toBeDefined();
    expect(result.resultType).toBe('chart');
    expect(result.question).toBe('test question');

    // Verify the 3rd invocation received tool-result messages
    const invocations = (mockLLM.invoke as ReturnType<typeof vi.fn>).mock.calls;
    expect(invocations).toHaveLength(3);
    const thirdCallMessages = invocations[2]?.[0]?.messages;
    expect(thirdCallMessages).toBeDefined();
    const hasToolMessage = thirdCallMessages?.some((m: { role: string }) => m.role === 'tool');
    expect(hasToolMessage).toBe(true);
  });

  it('budget exceeded throws AnalysisBudgetExceededError', async () => {
    // Always return tool_calls (9+ turns) to exceed MAX_TOOL_ITERATIONS (8)
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockResolvedValue({
      content: '',
      toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
      finishReason: 'tool_calls',
      usage: { totalTokens: 100 },
    });

    await expect(analyzer.analyze('test')).rejects.toThrow(AnalysisBudgetExceededError);

    try {
      await analyzer.analyze('test');
    } catch (e) {
      expect(e).toBeInstanceOf(AnalysisBudgetExceededError);
      const err = e as AnalysisBudgetExceededError;
      expect(err.context?.tripped).toBe('iterations');
      expect(err.context?.limit).toBe(8);
    }
  });

  it('token budget exceeded trips the tokens budget', async () => {
    const analyzerWithSmallBudget = createAnalyzer({ maxToolLoopTokens: 100 });

    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 500 }, // Exceeds 100 budget
      });

    await expect(analyzerWithSmallBudget.analyze('test')).rejects.toThrow(AnalysisBudgetExceededError);

    try {
      await analyzerWithSmallBudget.analyze('test');
    } catch (e) {
      expect(e).toBeInstanceOf(AnalysisBudgetExceededError);
      const err = e as AnalysisBudgetExceededError;
      expect(err.context?.tripped).toBe('tokens');
      expect(err.context?.limit).toBe(100);
    }
  });

  it('abort signal stops the loop', async () => {
    const controller = new AbortController();
    controller.abort(); // Abort before first invoke

    await expect(analyzer.analyze('test', controller.signal)).rejects.toThrow();
  });

  it('hook errors do not break the loop', async () => {
    const throwingHooks = {
      onToolStart: () => { throw new Error('hook error'); },
      onToolEnd: () => { throw new Error('hook error'); },
    };

    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ answer: 'Success despite hook errors', insights: [] }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('test', undefined, throwingHooks);

    expect(result.explanation).toBe('Success despite hook errors');
  });

  it('malformed tool calls feed an error result back', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: 'not-json' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ answer: 'Recovered', insights: [] }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('test');

    expect(result.explanation).toBe('Recovered');

    // Verify the 2nd invocation's messages contain the error result
    const invocations = (mockLLM.invoke as ReturnType<typeof vi.fn>).mock.calls;
    const secondCallMessages = invocations[1]?.[0]?.messages;
    const hasErrorToolMessage = secondCallMessages?.some(
      (m: { role: string; content: string }) => m.role === 'tool' && m.content.includes('malformed')
    );
    expect(hasErrorToolMessage).toBe(true);
  });

  it('tool results are truncated to 4000 chars', async () => {
    const hugeResult = 'x'.repeat(5000);

    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'run_code', arguments: JSON.stringify({ code: `result = "${hugeResult}"` }) } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ answer: 'Done', insights: [] }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    await analyzer.analyze('test');

    // Check that the tool message was truncated
    const invocations = (mockLLM.invoke as ReturnType<typeof vi.fn>).mock.calls;
    const secondCallMessages = invocations[1]?.[0]?.messages;
    const toolMessage = secondCallMessages?.find((m: { role: string }) => m.role === 'tool');
    expect(toolMessage).toBeDefined();
    expect(toolMessage!.content.length).toBeLessThanOrEqual(4000 + 20); // 4000 + ellipsis
    expect(toolMessage!.content).toContain('[truncated]');
  });

  // ============================================================================
  // Capability fallback tests (FAILING-FIRST for todo 7)
  // ============================================================================

  it('falls back to single-shot when the model never calls tools', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '{"answer":"x"}',
        finishReason: 'stop',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ code: 'result = 1', explanation: 'done' }),
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('test question');

    expect(result.mode).toBe('single-shot');
    expect(result.fallbackReason).toBe('no-first-tool-call');
  });

  it('falls back when the provider rejects the tools param', async () => {
    const { GenerationFailedError } = await import('../lib/errors');

    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce(new GenerationFailedError('provider', new Error('400 Bad Request: tools not supported')))
      .mockResolvedValueOnce({
        content: JSON.stringify({ code: 'result = 1', explanation: 'done' }),
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('test question');

    expect(result.mode).toBe('single-shot');
    expect(result.fallbackReason).toBe('provider-rejected-tools');
  });

  it('falls back after 2 consecutive malformed tool calls', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: 'not-json' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c2', type: 'function', function: { name: 'profile_column', arguments: 'also-bad' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 200 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ code: 'result = 1', explanation: 'done' }),
        usage: { totalTokens: 300 },
      });

    const result = await analyzer.analyze('test question');

    expect(result.mode).toBe('single-shot');
    expect(result.fallbackReason).toBe('malformed-tool-calls');
  });

  it('tool path records mode and toolTrace', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({
          answer: 'The answer',
          insights: [{ finding: 'F1', evidence: 'E1', confidence: 'high' }],
        }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('test question');

    expect(result.mode).toBe('tools');
    expect(result.toolTrace).toBeDefined();
    expect(Array.isArray(result.toolTrace)).toBe(true);
    expect(result.toolTrace!.length).toBeGreaterThan(0);
    for (const entry of result.toolTrace!) {
      expect(entry).toHaveProperty('tool');
      expect(entry).toHaveProperty('calls');
      expect(entry).toHaveProperty('durationMs');
      expect(typeof entry.tool).toBe('string');
      expect(typeof entry.calls).toBe('number');
      expect(typeof entry.durationMs).toBe('number');
    }
  });

  // ============================================================================
  // Fix validation tests (FAILING-FIRST for the three fixes)
  // ============================================================================

  it('returns a helpful guard result when no datasets are loaded', async () => {
    const emptyAnalyzer = createDataAnalyzer({
      llm: mockLLM,
      embeddings: mockEmbeddings,
      datasets: new Map([['aq', aq]]), // Only 'aq', no real datasets
      metadata: {},
      codeGenModel: 'test-model',
    });

    const result = await emptyAnalyzer.analyze('any question');

    expect(result.resultType).toBe('scalar');
    expect(result.attempts).toBe(0);
    expect(result.explanation).toContain('No datasets are loaded yet');
    expect(mockLLM.invoke).not.toHaveBeenCalled();
  });

  it('normalizes unknown confidence to low', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({
          answer: 'The answer',
          insights: [{ finding: 'F1', evidence: 'E1', confidence: 'extremely-high' }],
        }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('test question');

    expect(result.insights).toHaveLength(1);
    expect(result.insights?.[0].confidence).toBe('low');
  });

  it('drops an invalid synthesis chart', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
        finishReason: 'tool_calls',
        usage: { totalTokens: 100 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({
          answer: 'The answer',
          insights: [],
          chart: { type: 'scatter', title: 'Invalid', xKey: 'x', yKeys: ['y'], data: [{ x: 1, y: 2 }] },
        }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('test question');

    expect(result.chartConfig).toBeUndefined();
    expect(result.resultType).toBe('scalar');
  });

  // ============================================================================
  // Mid-loop 429 per-call retry tests
  // ============================================================================

  it('retries on RateLimitError and succeeds within MAX_LLM_CALL_RETRIES', async () => {
    vi.useFakeTimers();
    let attempt = 0;
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockImplementation(() => {
      attempt++;
      if (attempt <= 2) {
        return Promise.reject(new RateLimitError('test-provider', 1000));
      }
      // First successful call returns tool calls (iteration 1)
      if (attempt === 3) {
        return Promise.resolve({
          content: '',
          toolCalls: [{ id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } }],
          finishReason: 'tool_calls',
          usage: { totalTokens: 100 },
        });
      }
      // Second successful call returns synthesis (iteration 2)
      return Promise.resolve({
        content: JSON.stringify({ answer: 'Success after retries', insights: [] }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });
    });

    const promise = analyzer.analyze('test question');
    // Drive through retries (base backoff 1000ms * 2^attempt)
    await vi.advanceTimersByTimeAsync(3000);
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result.explanation).toBe('Success after retries');
    expect(attempt).toBe(4); // 2 retries + 2 successful calls
    vi.useRealTimers();
  });

  it('exhausts retries and surfaces RateLimitError after MAX_LLM_CALL_RETRIES+1 attempts', async () => {
    vi.useFakeTimers();
    let attempt = 0;
    (mockLLM.invoke as ReturnType<typeof vi.fn>).mockImplementation(() => {
      attempt++;
      return Promise.reject(new RateLimitError('test-provider', 1000));
    });

    const promise = analyzer.analyze('test question').catch((e: unknown) => e);
    // Drive through all retries (3 retries = 4 total attempts)
    await vi.advanceTimersByTimeAsync(15000);
    await vi.runAllTimersAsync();
    const error = await promise;

    expect(error).toBeInstanceOf(RateLimitError);
    expect(attempt).toBe(4); // initial + 3 retries
    vi.useRealTimers();
  });

  // ============================================================================
  // toolTrace tokensUsed distribution tests
  // ============================================================================

  it('distributes iteration tokens evenly across tool calls in toolTrace', async () => {
    (mockLLM.invoke as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        content: '',
        toolCalls: [
          { id: 'c1', type: 'function', function: { name: 'list_datasets', arguments: '{}' } },
          { id: 'c2', type: 'function', function: { name: 'profile_column', arguments: JSON.stringify({ dataset: 'employees', column: 'salary_usd' }) } },
        ],
        finishReason: 'tool_calls',
        usage: { totalTokens: 120 },
      })
      .mockResolvedValueOnce({
        content: JSON.stringify({ answer: 'Done', insights: [] }),
        finishReason: 'stop',
        usage: { totalTokens: 200 },
      });

    const result = await analyzer.analyze('test question');

    expect(result.toolTrace).toBeDefined();
    expect(result.toolTrace!.length).toBe(2);
    // 120 tokens / 2 calls = 60 each (Math.round)
    for (const entry of result.toolTrace!) {
      expect(entry.tokensUsed).toBe(60);
    }
    // Loop-level tokensUsed includes both iterations (120 + 200 = 320)
    expect(result.toolTrace!.reduce((sum, e) => sum + e.tokensUsed, 0)).toBe(120);
  });
});