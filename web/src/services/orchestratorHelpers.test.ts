import { describe, it, expect, vi } from 'vitest';
import {
  expandHyDE,
  parallelFanOut,
  parallelGrade,
  formatHeadingCitation,
} from './orchestratorHelpers';
import type { Document } from '../lib/types';
import type { LLMClient } from '../lib/llm';

function makeLLM(content: string): LLMClient {
  return {
    invoke: vi.fn(async () => ({ content })),
    stream: vi.fn(async () => ({ content })),
  };
}

function makeDoc(
  id: string,
  content: string,
  source = 'a.pdf',
  page?: number,
  meta: Record<string, unknown> = {},
): Document {
  const d: Document = { id, content, source, score: 0.5, metadata: meta };
  if (page !== undefined) d.page = page;
  return d;
}

describe('expandHyDE', () => {
  it('returns a hypothetical passage prefixed to the original question', async () => {
    const llm = makeLLM('Hypothetical answer paragraph.');
    const out = await expandHyDE('What is X?', { llm });
    expect(out).toContain('What is X?');
    expect(out).toContain('Hypothetical answer paragraph');
  });

  it('falls back to original question when LLM throws', async () => {
    const llm: LLMClient = {
      invoke: vi.fn(async () => { throw new Error('boom'); }),
      stream: vi.fn(),
    };
    const out = await expandHyDE('What is X?', { llm });
    expect(out).toBe('What is X?');
  });

  it('falls back when LLM returns empty content', async () => {
    const llm = makeLLM('');
    const out = await expandHyDE('What is X?', { llm });
    expect(out).toBe('What is X?');
  });

  it('passes through the model override when provided', async () => {
    const llm = makeLLM('passage');
    await expandHyDE('q', { llm, model: 'custom-model' });
    expect(llm.invoke).toHaveBeenCalledWith(expect.objectContaining({ model: 'custom-model' }));
  });
});

describe('parallelFanOut', () => {
  it('merges results from both query and hypothetical fetches', async () => {
    const fetch = vi.fn(async (q: string, _k: number): Promise<Document[]> => {
      if (q.startsWith('hypo')) return [makeDoc('1', 'passage A about X'), makeDoc('2', 'passage B about Y')];
      return [makeDoc('2', 'passage B about Y'), makeDoc('3', 'unrelated passage C')];
    });
    const out = await parallelFanOut(fetch, 'real query', 'hypo passage A about Z', 4, 2);
    expect(out).toHaveLength(2);
    const ids = out.map(d => d.id);
    expect(ids[0]).toBe('1');
    expect(new Set(ids)).toEqual(new Set(['1', '2']));
  });

  it('deduplicates by id across both fetches', async () => {
    const fetch = vi.fn(async () => [makeDoc('1', 'a'), makeDoc('1', 'a'), makeDoc('2', 'b')]);
    const out = await parallelFanOut(fetch, 'q', 'h', 4, 3);
    expect(out).toHaveLength(2);
  });

  it('runs both fetches concurrently', async () => {
    let resolveA: (() => void) | null = null;
    let resolveB: (() => void) | null = null;
    const fetch = vi.fn(async (q: string): Promise<Document[]> => {
      if (q === 'q') {
        await new Promise<void>(r => { resolveA = r; });
        return [makeDoc('1', 'x')];
      }
      await new Promise<void>(r => { resolveB = r; });
      return [makeDoc('2', 'y')];
    });
    const p = parallelFanOut(fetch, 'q', 'h', 4, 2);
    await new Promise(r => setTimeout(r, 10));
    if (resolveA) resolveA();
    if (resolveB) resolveB();
    const out = await p;
    expect(out).toHaveLength(2);
  });

  it('reranks by hypothetical token overlap', async () => {
    const fetch = vi.fn(async (q: string): Promise<Document[]> => {
      if (q === 'q') return [
        makeDoc('a', 'unrelated content about cats'),
        makeDoc('b', 'context-free noise about dogs'),
      ];
      return [];
    });
    const out = await parallelFanOut(fetch, 'q', 'context-free noise', 4, 1);
    expect(out[0]?.id).toBe('b');
  });

  it('returns rerankK or fewer documents', async () => {
    const fetch = vi.fn(async () => [makeDoc('1', 'a'), makeDoc('2', 'b'), makeDoc('3', 'c')]);
    const out = await parallelFanOut(fetch, 'q', 'h', 4, 2);
    expect(out.length).toBeLessThanOrEqual(2);
  });

  it('handles empty fetches gracefully', async () => {
    const fetch = vi.fn(async () => [] as Document[]);
    const out = await parallelFanOut(fetch, 'q', 'h', 4, 2);
    expect(out).toEqual([]);
  });
});

describe('parallelGrade', () => {
  it('keeps relevant docs and drops irrelevant ones', async () => {
    const docs = [makeDoc('1', 'a'), makeDoc('2', 'b'), makeDoc('3', 'c')];
    const grade = vi.fn(async (d: Document) => d.id === '2' ? 'relevant' : 'irrelevant');
    const out = await parallelGrade(docs, grade);
    expect(out.map(d => d.id)).toEqual(['2']);
  });

  it('keeps docs on grader error', async () => {
    const docs = [makeDoc('1', 'a')];
    const grade = vi.fn(async () => { throw new Error('boom'); });
    const out = await parallelGrade(docs, grade);
    expect(out).toHaveLength(1);
  });

  it('early-exits once earlyExitAt relevant docs are found', async () => {
    const docs = Array.from({ length: 10 }, (_, i) => makeDoc(String(i), 'x'));
    const grade = vi.fn(async () => 'relevant');
    const out = await parallelGrade(docs, grade, { earlyExitAt: 3 });
    expect(out).toHaveLength(3);
  });

  it('runs all grades concurrently (no sequential awaits)', async () => {
    const order: string[] = [];
    const docs = [makeDoc('1', 'a'), makeDoc('2', 'b')];
    const grade = vi.fn(async (d: Document) => {
      order.push(`start:${d.id}`);
      await new Promise(r => setTimeout(r, 20));
      order.push(`end:${d.id}`);
      return 'relevant';
    });
    await parallelGrade(docs, grade);
    expect(order.slice(0, 2).sort()).toEqual(['start:1', 'start:2']);
  });

  it('returns empty array when no docs are relevant', async () => {
    const docs = [makeDoc('1', 'a')];
    const grade = vi.fn(async () => 'irrelevant');
    const out = await parallelGrade(docs, grade);
    expect(out).toEqual([]);
  });
});

describe('formatHeadingCitation', () => {
  it('includes heading prefix when metadata.heading is present', () => {
    const d = makeDoc('1', 'body', 'a.pdf', 1, { heading: 'Intro' });
    const c = formatHeadingCitation(d);
    expect(c.excerpt).toBe('[Intro] body');
    expect(c.source).toBe('a.pdf');
    expect(c.page).toBe(1);
    expect(c.type).toBe('vectorstore');
  });

  it('omits heading prefix when no heading metadata', () => {
    const d = makeDoc('1', 'body', 'a.pdf');
    const c = formatHeadingCitation(d);
    expect(c.excerpt).toBe('body');
  });

  it('truncates long body excerpts to 200 chars', () => {
    const long = 'x'.repeat(500);
    const d = makeDoc('1', long, 'a.pdf');
    const c = formatHeadingCitation(d);
    expect(c.excerpt.length).toBeLessThanOrEqual(200);
  });

  it('omits page field when undefined', () => {
    const d = makeDoc('1', 'body', 'a.pdf', undefined);
    const c = formatHeadingCitation(d);
    expect(c.page).toBeUndefined();
  });

  it('handles source with special characters', () => {
    const d = makeDoc('1', 'body', 'file (1).pdf');
    const c = formatHeadingCitation(d);
    expect(c.source).toBe('file (1).pdf');
  });

  it('returns empty excerpt for empty content', () => {
    const d = makeDoc('1', '', 'a.pdf');
    const c = formatHeadingCitation(d);
    expect(c.excerpt).toBe('');
  });
});

describe('expandHyDE', () => {
  it('handles LLM timeout gracefully', async () => {
    const llm: LLMClient = {
      invoke: vi.fn(async () => { throw new Error('timeout'); }),
      stream: vi.fn(),
    };
    const out = await expandHyDE('What is X?', { llm });
    expect(out).toBe('What is X?');
  });
});
