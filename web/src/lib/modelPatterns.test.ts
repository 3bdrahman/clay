import { describe, expect, it } from 'vitest';
import {
  CHAT_PATTERNS,
  CODE_PATTERNS,
  EMBEDDING_PATTERNS,
  CHAT_DETECT,
  CODE_DETECT,
  SAFETY_DETECT,
  VISION_DETECT,
  EMBEDDING_DETECT,
  SIZE_PATTERNS,
  scoreByRules,
} from './modelPatterns';

describe('CHAT_PATTERNS', () => {
  it('recognises known chat-model families', () => {
    expect(scoreByRules('meta/llama-3.3-70b-instruct', CHAT_PATTERNS)).toBeGreaterThanOrEqual(30);
    expect(scoreByRules('mistralai/mistral-large-2-2407', CHAT_PATTERNS)).toBeGreaterThanOrEqual(28);
    expect(scoreByRules('nvidia/nemotron-3-ultra-550b-a55b', CHAT_PATTERNS)).toBeGreaterThanOrEqual(40);
  });

  it('does not boost embedding or code models', () => {
    expect(scoreByRules('nvidia/nv-embedqa-e5-v5', CHAT_PATTERNS)).toBe(0);
    expect(scoreByRules('mistralai/codestral-22b-instruct', CHAT_PATTERNS)).toBe(0);
  });
});

describe('CODE_PATTERNS', () => {
  it('scores codestral higher than codegemma', () => {
    const codestral = scoreByRules('mistralai/codestral-22b-instruct', CODE_PATTERNS);
    const codegemma = scoreByRules('google/codegemma-7b-instruct', CODE_PATTERNS);
    expect(codestral).toBeGreaterThan(codegemma);
  });

  it('adds size-tier bonus on top of family score', () => {
    const small = scoreByRules('mistralai/codestral-7b-instruct', CODE_PATTERNS);
    const medium = scoreByRules('mistralai/codestral-22b-instruct', CODE_PATTERNS);
    expect(medium).toBeGreaterThan(small);
  });

  it('does not boost embedding models', () => {
    expect(scoreByRules('nvidia/nv-embedcode-1.0', CODE_PATTERNS)).toBe(0);
  });
});

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

describe('class detectors', () => {
  it('EMBEDDING_DETECT matches embed/embedqa ids', () => {
    for (const id of ['nvidia/nv-embedqa-e5-v5', 'snowflake/arctic-embed-l']) {
      expect(EMBEDDING_DETECT.some((re) => re.test(id))).toBe(true);
    }
  });

  it('CODE_DETECT matches code-specialist ids', () => {
    for (const id of ['mistralai/codestral-22b-instruct', 'meta/codellama-70b-instruct']) {
      expect(CODE_DETECT.some((re) => re.test(id))).toBe(true);
    }
  });

  it('SAFETY_DETECT matches guard/safety ids and excludes chat', () => {
    expect(SAFETY_DETECT.some((re) => re.test('meta/llama-guard-3-8b'))).toBe(true);
    expect(SAFETY_DETECT.some((re) => re.test('meta/llama-3.1-8b-instruct'))).toBe(false);
  });

  it('VISION_DETECT matches vision ids and excludes text', () => {
    expect(VISION_DETECT.some((re) => re.test('meta/llama-3.2-11b-vision-instruct'))).toBe(true);
    expect(VISION_DETECT.some((re) => re.test('meta/llama-3.1-8b-instruct'))).toBe(false);
  });

  it('CHAT_DETECT matches chat ids', () => {
    expect(CHAT_DETECT.some((re) => re.test('meta/llama-3.1-8b-instruct'))).toBe(true);
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
