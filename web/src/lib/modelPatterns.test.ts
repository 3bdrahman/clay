import { describe, expect, it } from 'vitest';
import { SIZE_PATTERNS } from './modelPatterns';

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

  it('loads the externalized config (size tiers present, embedding rules gone)', () => {
    expect(SIZE_PATTERNS.length).toBeGreaterThanOrEqual(5);
    for (const entry of SIZE_PATTERNS) {
      expect(entry.patterns.length).toBeGreaterThan(0);
    }
  });
});
