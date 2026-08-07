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

describe('chunkText V2 — markdown headings', () => {
  it('records the heading in ChunkMetadata when input starts with a markdown heading', () => {
    // Given: a short markdown doc under a single heading
    const text = '# Heading\nBody sentence here.';
    // When: chunked
    const chunks = chunkText(text);
    // Then: the chunk's metadata.heading is the heading text
    expect(chunks.length).toBeGreaterThanOrEqual(1);
    expect(chunks[0]?.metadata?.heading).toBe('Heading');
  });

  it('assigns distinct headings to separate chunks across multiple sections', () => {
    // Given: two markdown sections, each large enough to emit its own chunk
    const text =
      '# H1\n' + 'body1. '.repeat(120) +
      '\n# H2\n' + 'body2. '.repeat(120);
    // When
    const chunks = chunkText(text);
    // Then: at least one chunk carries heading H1, another carries H2
    const headings = chunks.map((c) => c.metadata?.heading);
    expect(headings).toContain('H1');
    expect(headings).toContain('H2');
  });
});

describe('chunkText V2 — token budget', () => {
  it('emits chunks whose tokenCount never exceeds the configured tokenBudget', () => {
    // Given: many clear sentences and a tight token budget
    const sentences = Array.from(
      { length: 40 },
      (_, i) => `Sentence number ${i + 1} ends here.`
    );
    const text = sentences.join(' ');
    const tokenBudget = 25;
    // When
    const chunks = chunkText(text, { tokenBudget });
    // Then: every chunk stays at or under the budget
    for (const c of chunks) {
      expect(c.metadata?.tokenCount).toBeLessThanOrEqual(tokenBudget);
    }
  });
});

describe('chunkText V2 — sentence boundaries', () => {
  it('splits only at sentence boundaries so no chunk ends mid-sentence', () => {
    // Given: multi-sentence input with clear punctuation
    const text = Array.from(
      { length: 20 },
      (_, i) => `This is sentence ${i + 1}.`
    ).join(' ');
    // When
    const chunks = chunkText(text, { tokenBudget: 25 });
    // Then: each chunk's last non-space char is sentence-ending punctuation,
    // unless it is the trailing chunk reaching end-of-input.
    for (const c of chunks) {
      const last = c.text.trim().at(-1);
      expect(last === undefined || /[.!?]/.test(last ?? '')).toBe(true);
    }
  });
});

describe('chunkText V2 — sentence-aware overlap', () => {
  it('starts chunk[i+1] with the last sentence(s) of chunk[i]', () => {
    // Given: enough sentences to force multiple chunks and overlap
    const text = Array.from(
      { length: 30 },
      (_, i) => `Sentence ${i + 1} contents.`
    ).join(' ');
    // When
    const chunks = chunkText(text, { tokenBudget: 25, overlapTokens: 13 });
    // Then: the next chunk's text begins with a trailing sentence of the prior
    // chunk (sentence-aware overlap, not a char count).
    expect(chunks.length).toBeGreaterThan(1);
    const splitSent = (s: string): string[] =>
      s.split(/(?<=[.!?])\s+/).filter(Boolean);
    for (let i = 0; i + 1 < chunks.length; i++) {
      const prevSentences = splitSent(chunks[i].text);
      const nextFirst = splitSent(chunks[i + 1].text)[0];
      if (nextFirst === undefined) continue;
      // Then: chunk[i+1] begins with a trailing sentence of chunk[i]
      // (sentence-aware overlap — one or more trailing sentences carried over).
      const tail = prevSentences.slice(-8);
      expect(tail.includes(nextFirst)).toBe(true);
    }
  });
});

describe('chunkText V2 — full metadata population', () => {
  it('populates every ChunkMetadata field with valid offsets and sequential chunkIndex', () => {
    // Given: a multi-section markdown doc with context
    const text =
      '# Intro\n' + 'Intro body line one. ' + 'Intro body line two.\n' +
      '# Details\n' + 'Details body line one. ' + 'Details body line two.';
    // When
    const chunks = chunkText(
      text,
      {},
      { source: 'doc.md', sourceHash: 'abc123', modelId: 'm1', startPage: 3 }
    );
    // Then: all fields present, offsets valid into source text, sequential index
    const required = [
      'source', 'sourceHash', 'charStart', 'charEnd', 'chunkIndex',
      'tokenCount', 'modelId',
    ] as const;
    for (let i = 0; i < chunks.length; i++) {
      const m = chunks[i]?.metadata;
      expect(m).toBeDefined();
      if (!m) continue;
      for (const key of required) {
        expect(m[key]).not.toBeUndefined();
      }
      expect(m.source).toBe('doc.md');
      expect(m.sourceHash).toBe('abc123');
      expect(m.modelId).toBe('m1');
      expect(m.page).toBe(3);
      expect(m.chunkIndex).toBe(i);
      expect(m.charStart).toBeGreaterThanOrEqual(0);
      expect(m.charEnd).toBeLessThanOrEqual(text.length);
      expect(m.charEnd).toBeGreaterThan(m.charStart);
    }
  });

  it('falls back to empty-string / zero defaults when ctx and opts are omitted', () => {
    // Given: a short text under the default budget
    const text = 'A single short sentence.';
    // When
    const chunks = chunkText(text);
    // Then: metadata present with default-filled values, single chunk
    expect(chunks).toHaveLength(1);
    const m = chunks[0]?.metadata;
    expect(m?.source).toBe('');
    expect(m?.sourceHash).toBe('');
    expect(m?.modelId).toBe('');
    expect(m?.chunkIndex).toBe(0);
    expect(m?.charStart).toBe(0);
    expect(m?.charEnd).toBe(text.length);
    expect(m?.tokenCount).toBeGreaterThan(0);
  });
});

describe('chunkText V2 — empty / single-chunk contracts', () => {
  it('returns a single empty chunk for empty input (back-compat)', () => {
    // Given: empty string
    // When
    const chunks = chunkText('');
    // Then: the binding old contract [{ text: '', index: 0 }] with no metadata key
    expect(chunks).toEqual([{ text: '', index: 0 }]);
  });

  it('keeps chunkIndex === 0 and charEnd === text.length when text fits the budget', () => {
    // Given: text well under the default token budget
    const text = 'One sentence. Two sentence.';
    // When
    const chunks = chunkText(text);
    // Then: single chunk, metadata fully describes the whole text
    expect(chunks).toHaveLength(1);
    expect(chunks[0].metadata?.chunkIndex).toBe(0);
    expect(chunks[0].metadata?.charEnd).toBe(text.length);
  });
});

describe('chunkText V2 — ChunkContext propagation', () => {
  it('propagates source, sourceHash, modelId, and startPage onto metadata.page', () => {
    // Given: a short text and a full context
    const text = 'Context propagation check.';
    const ctx = { source: 'a.md', sourceHash: 'abc', modelId: 'm1', startPage: 3 };
    // When
    const chunks = chunkText(text, {}, ctx);
    // Then: metadata carries the context fields plus page === startPage
    expect(chunks).toHaveLength(1);
    const m = chunks[0]?.metadata;
    expect(m).toBeDefined();
    if (!m) return;
    expect(m.source).toBe('a.md');
    expect(m.sourceHash).toBe('abc');
    expect(m.modelId).toBe('m1');
    expect(m.page).toBe(3);
  });
});

describe('chunkText V2 — backward compatibility', () => {
  it('translates deprecated { chunkSize, overlap } into char-mode budgets and keeps index === chunkIndex', () => {
    // Given: a large single-character input and the OLD-style options
    const text = 'a'.repeat(2000);
    // When: using the deprecated char-based options
    const chunks = chunkText(text, { chunkSize: 500, overlap: 100 });
    // Then: more than one chunk and index matches metadata.chunkIndex throughout
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) expect(c.index).toBe(c.metadata?.chunkIndex);

    // And: the same index === chunkIndex invariant holds for the no-opts no-ctx path
    const text2 = Array.from({ length: 40 }, (_, i) => `Sentence ${i + 1}.`).join(' ');
    for (const c of chunkText(text2)) expect(c.index).toBe(c.metadata?.chunkIndex);
  });
});
