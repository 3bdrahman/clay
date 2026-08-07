import { describe, it, expect } from 'vitest';
import { estimateTokens } from './tokens';

describe('estimateTokens', () => {
  it('returns 0 for empty string', () => {
    expect(estimateTokens('')).toBe(0);
  });

  it('returns 0 for whitespace-only input', () => {
    expect(estimateTokens('   \t\n  ')).toBe(0);
  });

  it('estimates tokens for a single ASCII word', () => {
    // 1 word -> ceil(1 * 1.3) = 2
    expect(estimateTokens('hello')).toBe(2);
  });

  it('estimates tokens for a multi-sentence ASCII text', () => {
    // "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
    // 17 words -> ceil(17 * 1.3) = ceil(22.1) = 23
    const text =
      'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.';
    expect(estimateTokens(text)).toBe(23);
  });

  it('estimates sensibly for CJK text with no spaces', () => {
    // No whitespace separators -> the whole string is one "word" after trim/split.
    // 1 token-unit -> ceil(1 * 1.3) = 2. Sensible: nonzero, finite.
    const text = '这是一个没有空格的中文字符串';
    expect(estimateTokens(text)).toBe(2);
  });

  it('estimates tokens for large text over 1000 words', () => {
    const word = 'word';
    const words = Array.from({ length: 1200 }, () => word).join(' ');
    // 1200 words -> ceil(1200 * 1.3) = ceil(1560) = 1560
    expect(estimateTokens(words)).toBe(1560);
  });
});
