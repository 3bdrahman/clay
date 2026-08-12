import { describe, it, expect } from 'vitest';
import questions from './questions.json';
import { runEval, formatReport, gradeQuestionSet, type EvalQuestion } from './runner';
import type { Settings } from '../lib/types';

const TEST_SETTINGS: Settings = {
  apiKey: import.meta.env.VITE_NIM_API_KEY ?? '',
  embeddingApiKey: '',
  webSearchProvider: 'duckduckgo',
  serperApiKey: '',
  temperature: 0,
  maxRetries: 3,
  theme: 'system',
};

describe('Eval golden set (issue #4)', () => {
  it('contains schema-bound questions without bundled-sample column names', () => {
    const set = questions as EvalQuestion[];
    expect(set.length).toBeGreaterThanOrEqual(20);

    for (const q of set) {
      expect(q.id).toMatch(/^[a-z]+-\d{3}$/);
      expect(['data_analysis', 'documents', 'web_search']).toContain(q.category);
      expect(['python', 'vectorstore', 'websearch']).toContain(q.expectedSource);
      expect(q.goldenAnswer.length).toBeGreaterThan(10);
    }
  });

  it('no longer references legacy expectedDatasets or expectedColumns fields', () => {
    const set = questions as EvalQuestion[];
    for (const q of set) {
      expect(q).not.toHaveProperty('expectedDatasets');
      expect(q).not.toHaveProperty('expectedColumns');
    }
  });

  it('does not reference bundled sample CSV filenames in questions or expected fields', () => {
    const set = questions as EvalQuestion[];
    const forbidden = ['employees.csv', 'projects.csv', 'feedback.csv'];
    for (const q of set) {
      const blob = JSON.stringify(q).toLowerCase();
      for (const f of forbidden) {
        expect(blob).not.toContain(f);
      }
    }
  });

  it('uses expectedColumnIntent for data_analysis questions', () => {
    const set = questions as EvalQuestion[];
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
    const set = questions as EvalQuestion[];
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
    const set = questions as EvalQuestion[];
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

describe('E2E Eval (requires VITE_NIM_API_KEY)', () => {
  it('runs full golden test set against live NIM', async () => {
    if (!TEST_SETTINGS.apiKey) {
      return;
    }
    const summary = await runEval(TEST_SETTINGS, questions as EvalQuestion[], (done, total, q) => {
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
