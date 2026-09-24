import { describe, it, expect, vi } from 'vitest';
import { runEval, formatReport, gradeQuestionSet, type EvalQuestion, computeLexicalOverlap, scoreWithJudge } from './runner';
import { generateEvalQuestions } from './dynamicQuestions';
import type { Settings } from '../lib/types';

const TEST_SETTINGS: Settings = {
  provider: 'openrouter',
  openrouterApiKey: import.meta.env.VITE_EVAL_API_KEY ?? '',
  groqApiKey: '',
  apiKey: '',
  embeddingApiKey: '',
  webSearchProvider: 'duckduckgo',
  serperApiKey: '',
  temperature: 0,
  maxRetries: 3,
  theme: 'system',
  localServerUrl: '',
  localModels: { chat: '', embeddings: '' },
  localCatalog: [],
  localCatalogFetchedAt: 0,
  pickedModelsOverride: {
    routing: '',
    codeGen: '',
    answer: '',
    eval: '',
    embedding: '',
  },
};

// Create test datasets and documents for dynamic question generation
const TEST_DATASETS = [
  {
    name: 'employees',
    fileName: 'employees.csv',
    columns: ['id', 'name', 'department', 'salary', 'hire_date'],
    rowCount: 100,
    sampleRows: [
      { id: 1, name: 'Alice', department: 'Engineering', salary: 120000, hire_date: '2020-01-15' },
      { id: 2, name: 'Bob', department: 'Sales', salary: 90000, hire_date: '2019-03-22' },
    ],
  },
  {
    name: 'projects',
    fileName: 'projects.csv',
    columns: ['id', 'name', 'budget', 'status', 'start_date'],
    rowCount: 50,
    sampleRows: [
      { id: 1, name: 'Project Alpha', budget: 500000, status: 'active', start_date: '2023-01-01' },
    ],
  },
];

const TEST_DOCUMENTS = [
  { fileName: 'handbook.pdf' },
  { fileName: 'benefits.md' },
];

// Generate dynamic questions for testing
const TEST_QUESTIONS = generateEvalQuestions(TEST_DATASETS, TEST_DOCUMENTS) as EvalQuestion[];

describe('Eval golden set (issue #4)', () => {
  it('contains schema-bound questions without bundled-sample column names', () => {
    const set = TEST_QUESTIONS;
    expect(set.length).toBeGreaterThanOrEqual(15);

    for (const q of set) {
      expect(q.id).toMatch(/^[a-z]+-\d{3}$/);
      expect(['data_analysis', 'documents', 'web_search']).toContain(q.category);
      expect(['python', 'vectorstore', 'websearch']).toContain(q.expectedSource);
      expect(q.goldenAnswer.length).toBeGreaterThan(10);
    }
  });

  it('no longer references legacy expectedDatasets or expectedColumns fields', () => {
    const set = TEST_QUESTIONS;
    for (const q of set) {
      expect(q).not.toHaveProperty('expectedDatasets');
      expect(q).not.toHaveProperty('expectedColumns');
    }
  });

  it('does not reference bundled sample CSV filenames in questions or expected fields', () => {
    const set = TEST_QUESTIONS;
    const forbidden = ['employees.csv', 'projects.csv', 'feedback.csv'];
    for (const q of set) {
      const blob = JSON.stringify(q).toLowerCase();
      for (const f of forbidden) {
        expect(blob).not.toContain(f);
      }
    }
  });

  it('uses expectedColumnIntent for data_analysis questions', () => {
    const set = TEST_QUESTIONS;
    const dataQs = set.filter((q) => q.category === 'data_analysis');
    expect(dataQs.length).toBeGreaterThan(0);
    for (const q of dataQs) {
      expect(Array.isArray(q.expectedColumnIntent)).toBe(true);
      expect((q.expectedColumnIntent ?? []).length).toBeGreaterThan(0);
    }
  });
});

describe('gradeQuestionSet', () => {
  it('produces a summary with the same total as the input set', () => {
    const set = TEST_QUESTIONS;
    const fakeResults = set.map((q) => ({
      questionId: q.id,
      question: q.question,
      category: q.category,
      expectedSource: q.expectedSource,
      actualSource: q.expectedSource,
      routingCorrect: true,
      retrievedChunks: q.minRelevantChunks || 1,
      relevantChunks: q.minRelevantChunks || 1,
      recallAtK: 1,
      answer: 'golden match',
      latencyMs: 0,
    }));
    const summary = gradeQuestionSet(set, fakeResults);
    expect(summary.total).toBe(set.length);
    expect(summary.passed).toBe(set.length);
    expect(summary.failed).toBe(0);
    expect(summary.routingAccuracy).toBe(1);
  });

  it('counts failures when actualSource mismatches expectedSource', () => {
    const set = TEST_QUESTIONS;
    const fakeResults = set.map((q, i) => ({
      questionId: q.id,
      question: q.question,
      category: q.category,
      expectedSource: q.expectedSource,
      actualSource: i % 2 === 0 ? q.expectedSource : 'python',
      routingCorrect: i % 2 === 0,
      retrievedChunks: 0,
      relevantChunks: 0,
      recallAtK: 0,
      answer: '',
      latencyMs: 0,
    }));
    const summary = gradeQuestionSet(set, fakeResults);
    expect(summary.failed).toBeGreaterThan(0);
    expect(summary.routingAccuracy).toBeLessThan(1);
  });
});

describe('formatReport', () => {
  it('renders a markdown report with totals and per-question status', () => {
    const summary = {
      total: 2,
      passed: 1,
      failed: 1,
      routingAccuracy: 0.5,
      avgRecallAtK: 0.5,
      avgLatencyMs: 100,
      byCategory: {
        documents: { total: 1, passed: 1, routingAccuracy: 1 },
        data_analysis: { total: 1, passed: 0, routingAccuracy: 0 },
      },
      results: [
        {
          questionId: 'docs-001',
          question: 'Q1?',
          category: 'documents',
          expectedSource: 'vectorstore',
          actualSource: 'vectorstore',
          routingCorrect: true,
          retrievedChunks: 1,
          relevantChunks: 1,
          recallAtK: 1,
          answer: 'a',
          latencyMs: 100,
        },
        {
          questionId: 'data-001',
          question: 'Q2?',
          category: 'data_analysis',
          expectedSource: 'python',
          actualSource: 'vectorstore',
          routingCorrect: false,
          retrievedChunks: 0,
          relevantChunks: 0,
          recallAtK: 0,
          answer: '',
          latencyMs: 100,
        },
      ],
    };
    const md = formatReport(summary);
    expect(md).toContain('# Clay Eval Report');
    expect(md).toContain('**Total Questions:** 2');
    expect(md).toContain('docs-001');
    expect(md).toContain('data-001');
    expect(md).toContain('Routing Accuracy');
  });
});

describe('E2E Eval (requires VITE_EVAL_API_KEY)', () => {
  it('runs full golden test set against the live provider', async () => {
    if (!TEST_SETTINGS.apiKey) {
      return;
    }
    const summary = await runEval(TEST_SETTINGS, TEST_QUESTIONS, (done, total, q) => {
      void done;
      void total;
      void q;
    });
    const report = formatReport(summary);
    expect(summary.routingAccuracy).toBeGreaterThanOrEqual(0.6);
    expect(summary.avgRecallAtK).toBeGreaterThanOrEqual(0.3);
    expect(summary.passed / summary.total).toBeGreaterThanOrEqual(0.4);
    expect(report).toContain('# Clay Eval Report');
  }, 300000);
});

describe('computeLexicalOverlap', () => {
  it('returns 1.0 for identical texts', () => {
    const text = 'The average salary is 100000';
    const score = computeLexicalOverlap(text, text);
    expect(score).toBe(1.0);
  });

  it('returns ~0 for completely disjoint texts', () => {
    const score = computeLexicalOverlap('apple banana', 'cherry date');
    expect(score).toBeLessThan(0.1);
  });

  it('returns intermediate score for partial overlap', () => {
    const score = computeLexicalOverlap('The average salary is 100000', 'The average salary is 120000');
    expect(score).toBeGreaterThan(0.3);
    expect(score).toBeLessThan(1.0);
  });

  it('handles empty strings', () => {
    expect(computeLexicalOverlap('', '')).toBe(1.0);
    expect(computeLexicalOverlap('hello', '')).toBe(0);
    expect(computeLexicalOverlap('', 'world')).toBe(0);
  });

  it('is case-insensitive', () => {
    const score = computeLexicalOverlap('HELLO WORLD', 'hello world');
    expect(score).toBe(1.0);
  });

  it('ignores punctuation', () => {
    const score = computeLexicalOverlap('Hello, world!', 'Hello world');
    expect(score).toBe(1.0);
  });
});

describe('scoreWithJudge', () => {
  it('returns score and rationale from mocked LLM', async () => {
    const mockLLM = {
      invoke: vi.fn().mockResolvedValue({
        content: JSON.stringify({ score: 0.8, rationale: 'Good coverage' }),
        usage: { totalTokens: 50 },
      }),
    };

    const result = await scoreWithJudge(mockLLM, 'test-model', 'What is X?', 'Answer A', 'Answer B');
    expect(result.score).toBe(0.8);
    expect(result.rationale).toBe('Good coverage');
  });

  it('clamps score to 0-1 range', async () => {
    const mockLLM = {
      invoke: vi.fn().mockResolvedValue({
        content: JSON.stringify({ score: 1.5, rationale: 'Too high' }),
        usage: { totalTokens: 50 },
      }),
    };

    const result = await scoreWithJudge(mockLLM, 'test-model', 'Q', 'A', 'B');
    expect(result.score).toBe(1.0);
  });

  it('handles LLM errors gracefully', async () => {
    const mockLLM = {
      invoke: vi.fn().mockRejectedValue(new Error('LLM down')),
    };

    const result = await scoreWithJudge(mockLLM, 'test-model', 'Q', 'A', 'B');
    expect(result.score).toBe(0);
    expect(result.rationale).toBe('Judge scoring failed');
  });

  it('handles invalid JSON from LLM', async () => {
    const mockLLM = {
      invoke: vi.fn().mockResolvedValue({
        content: 'not json',
        usage: { totalTokens: 50 },
      }),
    };

    const result = await scoreWithJudge(mockLLM, 'test-model', 'Q', 'A', 'B');
    expect(result.score).toBe(0);
    expect(result.rationale).toBe('Judge scoring failed');
  });
});

