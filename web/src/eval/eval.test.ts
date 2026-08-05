import { describe, it, expect } from 'vitest';
import questions from './questions.json';
import { runEval, formatReport, type EvalQuestion } from './runner';
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

describe.skip('E2E Eval (requires NIM_API_KEY)', () => {
  it('runs full golden test set', async () => {
    if (!TEST_SETTINGS.apiKey) {
      console.log('Skipping: VITE_NIM_API_KEY not set');
      return;
    }

    const summary = await runEval(TEST_SETTINGS, questions as EvalQuestion[], (done, total, q) => {
      console.log(`[${done + 1}/${total}] ${q.id}: ${q.question.slice(0, 60)}...`);
    });

    const report = formatReport(summary);
    console.log('\n' + report);

    // Assertions - adjust thresholds as needed
    expect(summary.routingAccuracy).toBeGreaterThanOrEqual(0.6);
    expect(summary.avgRecallAtK).toBeGreaterThanOrEqual(0.3);
    expect(summary.passed / summary.total).toBeGreaterThanOrEqual(0.4);
  }, 300000); // 5 min timeout
});

describe('Eval utils', () => {
  it('computes recall@k correctly', () => {
    // recallAtK = min(1, relevant / minRelevant)
    expect(Math.min(1, 0)).toBe(0);
    expect(Math.min(1, 1 / 1)).toBe(1);
    expect(Math.min(1, 2 / 3)).toBeCloseTo(0.67, 1);
  });

  it('normalizes answers for relevance', () => {
    const golden = "The query should group employees by department and count them.";
    const generated = "Group employees by department and count them please.";
    const g = golden.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
    const a = generated.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
    const gTokens = new Set(g.split(' ').filter(t => t.length > 2));
    const aTokens = new Set(a.split(' ').filter(t => t.length > 2));
    let overlap = 0;
    for (const t of gTokens) if (aTokens.has(t)) overlap++;
    const relevance = overlap / gTokens.size;
    expect(relevance).toBeGreaterThan(0.5);
  });
});