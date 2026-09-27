// Golden-summary gate — the no-LLM eval that runs in CI on every push.
//
// The schema-bound golden question set is regenerated deterministically and
// its lexical-overlap aggregates (overlap between each question and its
// golden answer, via the same computeLexicalOverlap the eval runner uses)
// are compared against a committed baseline. Aggregates that drop more than
// OVERLAP_TOLERANCE below the baseline fail the gate, so golden-set drift —
// a degraded golden answer, a mispaired question, a template change that
// hurts coherence — is caught continuously. Improvements are allowed.
//
// Regenerate the baseline after intentional golden-set changes:
//   EVAL_UPDATE_BASELINE=1 npx vitest run src/eval/goldenSummary.test.ts
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { describe, it, expect } from 'vitest';
import { computeLexicalOverlap, type EvalQuestion } from './runner';
import { TEST_QUESTIONS } from './fixtures';

// import.meta.url is http-schemed under the vitest module runner, so the
// baseline is resolved from the working directory: the nearest directory
// with a package.json and an src/eval (walking up covers repo-root runs).
function findBaselinePath(): string {
  let dir = process.cwd();
  for (;;) {
    if (existsSync(join(dir, 'package.json')) && existsSync(join(dir, 'src', 'eval'))) {
      return join(dir, 'src', 'eval', 'goldenBaseline.json');
    }
    const parent = dirname(dir);
    if (parent === dir) {
      return join(process.cwd(), 'src', 'eval', 'goldenBaseline.json');
    }
    dir = parent;
  }
}

const BASELINE_PATH = findBaselinePath();
const UPDATE_BASELINE = process.env.EVAL_UPDATE_BASELINE === '1';
// Must sit below the baseline's min per-question overlap, or the min gate's
// floor goes negative and that part of the gate can never trip.
const OVERLAP_TOLERANCE = 0.03;

interface GoldenBaseline {
  version: number;
  aggregates: {
    avgQuestionGoldenOverlap: number;
    minQuestionGoldenOverlap: number;
    perCategory: Record<string, number>;
  };
}

function computeGoldenAggregates(questions: EvalQuestion[]): GoldenBaseline['aggregates'] {
  const overlaps = questions.map(q => ({
    category: q.category,
    overlap: computeLexicalOverlap(q.question, q.goldenAnswer),
  }));
  const avg = (vals: number[]) => vals.reduce((s, v) => s + v, 0) / (vals.length || 1);
  const perCategory: Record<string, number> = {};
  for (const cat of new Set(overlaps.map(o => o.category))) {
    perCategory[cat] = avg(overlaps.filter(o => o.category === cat).map(o => o.overlap));
  }
  return {
    avgQuestionGoldenOverlap: avg(overlaps.map(o => o.overlap)),
    minQuestionGoldenOverlap: Math.min(...overlaps.map(o => o.overlap)),
    perCategory,
  };
}

function readBaseline(): GoldenBaseline {
  expect(
    existsSync(BASELINE_PATH),
    `golden baseline missing at ${BASELINE_PATH} — run EVAL_UPDATE_BASELINE=1 npx vitest run src/eval/goldenSummary.test.ts to write it`,
  ).toBe(true);
  const baseline = JSON.parse(readFileSync(BASELINE_PATH, 'utf-8')) as GoldenBaseline;
  expect(baseline.version, 'unexpected golden baseline version').toBe(1);
  return baseline;
}

describe('golden-summary gate (no-LLM eval in CI)', () => {
  it('keeps the schema-bound golden set coherent with its golden answers', () => {
    const actual = computeGoldenAggregates(TEST_QUESTIONS);

    if (UPDATE_BASELINE) {
      const baseline: GoldenBaseline = { version: 1, aggregates: actual };
      writeFileSync(BASELINE_PATH, JSON.stringify(baseline, null, 2) + '\n');
    }

    const baseline = readBaseline();

    // The comparison direction is load-bearing: aggregates must not DROP
    // more than OVERLAP_TOLERANCE below the committed baseline.
    expect(actual.avgQuestionGoldenOverlap).toBeGreaterThanOrEqual(
      baseline.aggregates.avgQuestionGoldenOverlap - OVERLAP_TOLERANCE,
    );
    expect(actual.minQuestionGoldenOverlap).toBeGreaterThanOrEqual(
      baseline.aggregates.minQuestionGoldenOverlap - OVERLAP_TOLERANCE,
    );
    for (const [cat, expected] of Object.entries(baseline.aggregates.perCategory)) {
      expect(actual.perCategory[cat], `${cat} overlap dropped below the baseline`).toBeGreaterThanOrEqual(
        expected - OVERLAP_TOLERANCE,
      );
    }
  });

  it('flags a degraded golden set (answers disjoint from their questions)', () => {
    const baseline = readBaseline();

    // One question paired with a disjoint answer: the min gate trips.
    const oneDegraded: EvalQuestion[] = TEST_QUESTIONS.map((q, i) =>
      i === 0 ? { ...q, goldenAnswer: 'zzz qqq yyy' } : q,
    );
    const oneDegradedAggregates = computeGoldenAggregates(oneDegraded);
    expect(oneDegradedAggregates.minQuestionGoldenOverlap).toBeLessThan(
      baseline.aggregates.minQuestionGoldenOverlap - OVERLAP_TOLERANCE,
    );

    // Every answer disjoint from its question: the avg gate trips.
    const allDegraded: EvalQuestion[] = TEST_QUESTIONS.map(q => ({ ...q, goldenAnswer: 'zzz qqq yyy' }));
    const allDegradedAggregates = computeGoldenAggregates(allDegraded);
    expect(allDegradedAggregates.avgQuestionGoldenOverlap).toBeLessThan(
      baseline.aggregates.avgQuestionGoldenOverlap - OVERLAP_TOLERANCE,
    );
  });
});
