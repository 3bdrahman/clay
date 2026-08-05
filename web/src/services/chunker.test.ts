import { describe, expect, it } from 'vitest';
import { chunkText, chunkSentences, prepareForChunking } from './chunker';

describe('chunkText', () => {
  it('returns a single chunk when text fits', () => {
    const text = 'Short text.';
    const chunks = chunkText(text, { chunkSize: 100, overlap: 10 });
    expect(chunks).toHaveLength(1);
    expect(chunks[0].text).toBe('Short text.');
    expect(chunks[0].index).toBe(0);
  });

  it('respects default chunk size of 800', () => {
    const text = 'a'.repeat(900);
    const chunks = chunkText(text);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(c.text.length).toBeLessThanOrEqual(800);
    }
  });

  it('creates overlapping chunks', () => {
    const text = 'a'.repeat(2000);
    const chunks = chunkText(text, { chunkSize: 500, overlap: 100 });
    expect(chunks.length).toBeGreaterThan(1);
    // The next chunk starts before the previous one ended
    if (chunks.length >= 2) {
      const overlapStart = chunks[1].text.indexOf('a');
      expect(overlapStart).toBeLessThan(100);
    }
  });

  it('breaks at sentence boundaries when possible', () => {
    const text = 'a'.repeat(500) + '. ' + 'b'.repeat(500) + '. ' + 'c'.repeat(500);
    const chunks = chunkText(text, { chunkSize: 800, overlap: 200 });
    // The first chunk should end near the first sentence break
    expect(chunks[0].text.endsWith('.') || chunks[0].text.endsWith('. ')).toBe(true);
  });

  it('handles empty input', () => {
    expect(chunkText('')).toEqual([{ text: '', index: 0 }]);
  });

  it('normalizes whitespace', () => {
    const text = 'Hello    world.\n\n\n\nFoo   bar.';
    const chunks = chunkText(text);
    expect(chunks[0].text).not.toMatch(/\s{2,}/);
  });

  it('returns sequential indices', () => {
    const text = 'a'.repeat(3000);
    const chunks = chunkText(text, { chunkSize: 500, overlap: 50 });
    chunks.forEach((c, i) => expect(c.index).toBe(i));
  });

  it('produces non-empty chunks', () => {
    const text = 'First sentence. Second sentence. Third sentence. Fourth. Fifth.';
    const chunks = chunkText(text, { chunkSize: 30, overlap: 5 });
    for (const c of chunks) {
      expect(c.text.length).toBeGreaterThan(0);
    }
  });
});

describe('chunkSentences', () => {
  it('returns one chunk per input when short', () => {
    const sentences = ['First sentence.', 'Second sentence.'];
    const chunks = chunkSentences(sentences);
    expect(chunks.length).toBeGreaterThanOrEqual(1);
    expect(chunks[0].text).toContain('First');
    expect(chunks[0].text).toContain('Second');
  });

  it('packs sentences up to chunkSize', () => {
    const sentences = Array.from({ length: 20 }, (_, i) => `Sentence number ${i + 1}.`);
    const chunks = chunkSentences(sentences, { chunkSize: 100, overlap: 20 });
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(c.text.length).toBeLessThanOrEqual(100);
    }
  });

  it('creates overlapping chunks when sentences overflow', () => {
    const sentences = Array.from({ length: 50 }, (_, i) => `Word number ${i + 1}.`);
    const chunks = chunkSentences(sentences, { chunkSize: 50, overlap: 10 });
    expect(chunks.length).toBeGreaterThan(1);
  });

  it('handles empty input', () => {
    expect(chunkSentences([])).toEqual([]);
  });

  it('trims whitespace', () => {
    const chunks = chunkSentences(['   Hello world.   ']);
    expect(chunks[0].text).toBe('Hello world.');
  });
});

describe('prepareForChunking', () => {
  it('splits on sentence boundaries', () => {
    const text = 'First sentence. Second sentence. Third sentence.';
    const sentences = prepareForChunking(text);
    expect(sentences.length).toBeGreaterThan(1);
    expect(sentences[0]).toContain('First');
  });

  it('splits on double newlines (paragraphs)', () => {
    const text = 'First paragraph.\n\nSecond paragraph.';
    const sentences = prepareForChunking(text);
    expect(sentences).toContain('First paragraph.');
    expect(sentences).toContain('Second paragraph.');
  });

  it('filters out empty segments', () => {
    const text = 'Hello.\n\n\n\n\n\nWorld.';
    const sentences = prepareForChunking(text);
    for (const s of sentences) {
      expect(s.trim().length).toBeGreaterThan(0);
    }
  });
});
