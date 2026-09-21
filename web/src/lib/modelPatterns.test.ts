import { describe, expect, it } from 'vitest';
import {
  EMBEDDING_PATTERNS,
  EMBEDDING_DETECT,
  SIZE_PATTERNS,
  scoreByRules,
} from './modelPatterns';

describe('EMBEDDING_PATTERNS', () => {
  it('ranks nv-embedqa-e5 above llama-nemotron-embed', () => {
    const e5 = scoreByRules('nvidia/nv-embedqa-e5-v5', EMBEDDING_PATTERNS);
    const nemotron = scoreByRules('nvidia/llama-nemotron-embed-v1', EMBEDDING_PATTERNS);
    expect(e5).toBeGreaterThan(nemotron);
  });

  it('returns zero for chat models', () => {
    expect(scoreByRules('meta/llama-3.1-8b-instruct', EMBEDDING_PATTERNS)).toBe(0);
  });
});

describe('EMBEDDING_DETECT', () => {
  it('matches embed/embedqa ids and excludes plain chat ids', () => {
    for (const id of ['nvidia/nv-embedqa-e5-v5', 'snowflake/arctic-embed-l', 'nomic-embed-text']) {
      expect(EMBEDDING_DETECT.some((re) => re.test(id))).toBe(true);
    }
    expect(EMBEDDING_DETECT.some((re) => re.test('meta/llama-3.1-8b-instruct'))).toBe(false);
  });
});

describe('scoreByRules', () => {
  it('sums points from every matching rule', () => {
    const rules = [
      { family: 'embed', pattern: /embed/, points: 10 },
      { family: 'embedqa', pattern: /embedqa/, points: 5 },
      { family: 'nomatch', pattern: /nothing-matches-this/, points: 100 },
    ];
    expect(scoreByRules('x/embedqa-y', rules)).toBe(15);
  });
});

describe('SIZE_PATTERNS', () => {
  it('classifies sizes in the expected buckets', () => {
    const buckets = new Map<string, string>();
    for (const entry of SIZE_PATTERNS) {
      buckets.set(entry.class, entry.patterns.map((re) => re.source).join('|'));
    }
    expect(buckets.get('huge')).toContain('550b');
    expect(buckets.get('large')).toContain('70b');
    expect(buckets.get('medium')).toContain('22b');
    expect(buckets.get('small')).toContain('8b');
    expect(buckets.get('tiny')).toContain('4b');
  });
});
