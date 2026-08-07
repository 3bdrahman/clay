import { describe, expect, it, vi } from 'vitest';
import type { ExtractedPage } from './pdf';

interface PdfItem {
  str: string;
  transform: number[];
  height: number;
  width: number;
  hasEOL?: boolean;
}

interface MarkedItem {
  type: string;
}

type AnyItem = PdfItem | MarkedItem;

interface FakePage {
  numPages: number;
  getPage: (n: number) => Promise<{ getTextContent: () => Promise<{ items: AnyItem[] }> }>;
}

type PageItems = AnyItem[];

function makeItem(str: string, x: number, y: number, height: number): PdfItem {
  return { str, transform: [1, 0, 0, 1, x, y], height, width: str.length };
}

function makeMarked(): MarkedItem {
  return { type: 'beginMarkedContent' };
}

const holder = vi.hoisted((): { fake: FakePage | null } => ({ fake: null }));

vi.mock('pdfjs-dist', () => ({
  GlobalWorkerOptions: { workerSrc: '' },
  getDocument: () => ({ promise: Promise.resolve(holder.fake) }),
}));
vi.mock('pdfjs-dist/build/pdf.worker.mjs?url', () => ({ default: '' }));

async function extract(pages: PageItems[]): Promise<ExtractedPage[]> {
  holder.fake = {
    numPages: pages.length,
    getPage: (n: number) =>
      Promise.resolve({
        getTextContent: () => Promise.resolve({ items: pages[n - 1] ?? [] }),
      }),
  };
  const { extractPdfText } = await import('./pdf');
  return extractPdfText(new ArrayBuffer(0));
}

describe('extractPdfText', () => {
  it('joins a single line and detects a heading when its height is large', async () => {
    const pages = await extract([
      [makeItem('Big Title', 50, 700, 24), makeItem('body line', 50, 650, 10), makeItem('more', 200, 650, 10)],
    ]);
    expect(pages).toHaveLength(1);
    expect(pages[0].text).toBe('## Big Title\nbody line more');
    expect(pages[0].heading).toBe('Big Title');
  });

  it('joins two items on the same y with a space', async () => {
    const pages = await extract([[makeItem('hello', 10, 500, 10), makeItem('world', 80, 500, 10)]]);
    expect(pages[0].text).toBe('hello world');
    expect(pages[0].heading).toBeUndefined();
  });

  it('joins two items on different y with a newline', async () => {
    const pages = await extract([[makeItem('first', 10, 600, 10), makeItem('second', 10, 500, 10)]]);
    expect(pages[0].text).toBe('first\nsecond');
  });

  it('prepends ## for an item whose height is 2x the page default', async () => {
    const pages = await extract([
      [makeItem('Head', 10, 700, 20), makeItem('a', 10, 650, 10), makeItem('b', 10, 600, 10)],
    ]);
    expect(pages[0].text.startsWith('## Head')).toBe(true);
    expect(pages[0].heading).toBe('Head');
  });

  it('prepends # for an item whose height is 3x the page default', async () => {
    const pages = await extract([
      [makeItem('Giant', 10, 700, 30), makeItem('a', 10, 650, 10), makeItem('b', 10, 600, 10)],
    ]);
    expect(pages[0].text.startsWith('# Giant')).toBe(true);
    expect(pages[0].heading).toBe('Giant');
  });

  it('preserves every item when columns are separated by a wide x-gap', async () => {
    const pages = await extract([
      [
        makeItem('left1', 10, 500, 10),
        makeItem('left2', 10, 480, 10),
        makeItem('right1', 400, 500, 10),
        makeItem('right2', 400, 480, 10),
      ],
    ]);
    const text = pages[0].text;
    expect(text).toContain('left1');
    expect(text).toContain('left2');
    expect(text).toContain('right1');
    expect(text).toContain('right2');
  });

  it('skips empty pages (no entry pushed)', async () => {
    const pages = await extract([[]]);
    expect(pages).toHaveLength(0);
  });

  it('skips pages where every item is whitespace-only', async () => {
    const pages = await extract([[makeItem('   ', 10, 500, 10), makeItem('', 20, 500, 10)]]);
    expect(pages).toHaveLength(0);
  });

  it('returns both pages of a multi-page doc, each with its own heading', async () => {
    const pages = await extract([
      [
        makeItem('DocTitle', 10, 700, 24),
        makeItem('body a', 10, 650, 10),
        makeItem('body b', 80, 650, 10),
        makeItem('body c', 10, 600, 10),
      ],
      [
        makeItem('Chapter2', 10, 700, 24),
        makeItem('body d', 10, 650, 10),
        makeItem('body e', 80, 650, 10),
      ],
    ]);
    expect(pages).toHaveLength(2);
    expect(pages[0].heading).toBe('DocTitle');
    expect(pages[1].heading).toBe('Chapter2');
  });

  it('skips items without str (marked content / image xobjects) gracefully', async () => {
    const pages = await extract([[makeMarked(), makeItem('real text', 10, 500, 10), makeMarked()]]);
    expect(pages).toHaveLength(1);
    expect(pages[0].text).toBe('real text');
  });
});
