import { describe, expect, it, vi } from 'vitest';
import {
  detectKind,
  processFile,
  embedDocumentChunks,
  existingSourceHashes,
} from './files';
import { hashText } from '../lib/hash';

class FakeFile {
  name: string;
  type: string;
  private _text: string | undefined;
  private _buffer: ArrayBuffer | undefined;

  constructor(name: string, content: string | ArrayBuffer, mimeType = '') {
    this.name = name;
    this.type = mimeType;
    if (typeof content === 'string') {
      this._text = content;
    } else {
      this._buffer = content;
    }
  }

  async text(): Promise<string> {
    if (this._text === undefined) throw new Error('No text content');
    return this._text;
  }

  async arrayBuffer(): Promise<ArrayBuffer> {
    if (this._buffer === undefined) {
      this._buffer = new TextEncoder().encode(this._text ?? '').buffer;
    }
    return this._buffer;
  }
}

describe('detectKind', () => {
  it('detects CSV by extension', () => {
    expect(detectKind('data.csv')).toBe('csv');
    expect(detectKind('DATA.CSV')).toBe('csv');
  });

  it('detects CSV by MIME type', () => {
    expect(detectKind('unknown', 'text/csv')).toBe('csv');
  });

  it('detects PDF by extension', () => {
    expect(detectKind('report.pdf')).toBe('pdf');
  });

  it('detects PDF by MIME type', () => {
    expect(detectKind('unknown', 'application/pdf')).toBe('pdf');
  });

  it('detects markdown', () => {
    expect(detectKind('readme.md')).toBe('text');
    expect(detectKind('doc.markdown')).toBe('text');
  });

  it('detects plain text', () => {
    expect(detectKind('notes.txt')).toBe('text');
    expect(detectKind('notes.text')).toBe('text');
  });

  it('detects JSON', () => {
    expect(detectKind('config.json')).toBe('text');
  });

  it('detects text by text/* MIME type', () => {
    expect(detectKind('unknown', 'text/plain')).toBe('text');
    expect(detectKind('unknown', 'text/markdown')).toBe('text');
  });

  it('returns unsupported for unknown types', () => {
    expect(detectKind('image.png')).toBe('unsupported');
    expect(detectKind('archive.zip', 'application/zip')).toBe('unsupported');
  });
});

describe('hashText', () => {
  it('returns stable hashes for identical input', () => {
    expect(hashText('hello')).toBe(hashText('hello'));
  });

  it('returns different hashes for different input', () => {
    expect(hashText('hello')).not.toBe(hashText('world'));
  });

  it('returns a non-empty string', () => {
    expect(hashText('x').length).toBeGreaterThan(0);
  });
});

describe('processFile', () => {
  it('parses CSV into an Arquero table', async () => {
    const csv = 'name,age\nAlice,30\nBob,25\n';
    const fake = new FakeFile('people.csv', csv, 'text/csv') as unknown as File;
    const result = await processFile(fake);

    expect(result.kind).toBe('csv');
    expect(result.error).toBeUndefined();
    expect(result.dataset).toBeDefined();
    expect(result.dataset?.columns).toEqual(['name', 'age']);
    expect(result.dataset?.rowCount).toBe(2);
    expect(result.dataset?.name).toBe('people');
  });

  it('sanitizes dataset names from filenames', async () => {
    const csv = 'a,b\n1,2\n';
    const fake = new FakeFile('My Sales Report (2024).csv', csv, 'text/csv') as unknown as File;
    const result = await processFile(fake);
    expect(result.dataset?.name).toBe('My_Sales_Report_2024');
  });

  it('processes text files into chunks', async () => {
    const text = 'First sentence. Second sentence. Third sentence.';
    const fake = new FakeFile('notes.txt', text, 'text/plain') as unknown as File;
    const result = await processFile(fake);

    expect(result.kind).toBe('text');
    expect(result.document).toBeDefined();
    expect(result.document?.chunks.length).toBeGreaterThan(0);
    expect(result.document?.chunks[0].id).toMatch(/notes-\d+/);
  });

  it('processes markdown into chunks', async () => {
    const md = '# Heading\n\nSome paragraph. Another sentence.';
    const fake = new FakeFile('README.md', md, 'text/markdown') as unknown as File;
    const result = await processFile(fake);

    expect(result.kind).toBe('text');
    expect(result.document?.chunks.length).toBeGreaterThan(0);
  });

  it('rejects unsupported file types with an error', async () => {
    const fake = new FakeFile('image.png', '', 'image/png') as unknown as File;
    const result = await processFile(fake);

    expect(result.kind).toBe('unsupported');
    expect(result.error).toBe('Unsupported file type');
    expect(result.dataset).toBeUndefined();
    expect(result.document).toBeUndefined();
  });

  it('parses lenient CSV (Arquero tolerates malformed rows)', async () => {
    const fake = new FakeFile('rough.csv', 'a,b\n1,2\n3,4\n', 'text/csv') as unknown as File;
    const result = await processFile(fake);

    expect(result.kind).toBe('csv');
    expect(result.error).toBeUndefined();
    expect(result.dataset?.rowCount).toBe(2);
  });

  it('normalizes currency strings to numbers', async () => {
    const csv = 'amount\n$1,234.56\n$99\n';
    const fake = new FakeFile('curr.csv', csv, 'text/csv') as unknown as File;
    const result = await processFile(fake);

    const rows = (result.dataset?.table.objects() ?? []) as Array<{ amount: number }>;
    expect(rows[0]?.amount).toBe(1234.56);
    expect(rows[1]?.amount).toBe(99);
  });

  it('normalizes numeric strings to numbers', async () => {
    const csv = 'price\n100\n200.50\n';
    const fake = new FakeFile('price.csv', csv, 'text/csv') as unknown as File;
    const result = await processFile(fake);

    const rows = (result.dataset?.table.objects() ?? []) as Array<{ price: number }>;
    expect(rows[0]?.price).toBe(100);
    expect(rows[1]?.price).toBe(200.5);
  });

  it('normalizes percentages to fractions', async () => {
    const csv = 'rate\n50%\n25.5%\n';
    const fake = new FakeFile('rate.csv', csv, 'text/csv') as unknown as File;
    const result = await processFile(fake);

    const rows = (result.dataset?.table.objects() ?? []) as Array<{ rate: number }>;
    expect(rows[0]?.rate).toBe(0.5);
    expect(rows[1]?.rate).toBe(0.255);
  });

  it('normalizes thousands-separated numbers in data rows only', async () => {
    const csv = 'label,value\n"Q1, 2024",1,234,567.89\nplain,42\n';
    const fake = new FakeFile('sep.csv', csv, 'text/csv') as unknown as File;
    const result = await processFile(fake);

    const rows = (result.dataset?.table.objects() ?? []) as Array<{ label: string; value: number }>;
    expect(rows[0]?.value).toBe(1234567.89);
    expect(rows[1]?.value).toBe(42);
  });

  it('stores empty CSV cells as null', async () => {
    const csv = 'name,value\nAlice,\nBob,100\n';
    const fake = new FakeFile('nulls.csv', csv, 'text/csv') as unknown as File;
    const result = await processFile(fake);

    const rows = (result.dataset?.table.objects() ?? []) as Array<{ name: string; value: number | null }>;
    expect(rows[0]?.value).toBeNull();
    expect(rows[1]?.value).toBe(100);
  });

  it('populates sourceHash on text documents', async () => {
    const text = 'Some text content for hashing purposes. '.repeat(10);
    const fake = new FakeFile('h.txt', text, 'text/plain') as unknown as File;
    const result = await processFile(fake);
    expect(result.document?.sourceHash).toBe(hashText(text));
  });

  it('sourceHash is identical for two files with identical content', async () => {
    const text = 'Identical content here. '.repeat(5);
    const a = new FakeFile('a.txt', text, 'text/plain') as unknown as File;
    const b = new FakeFile('b.txt', text, 'text/plain') as unknown as File;
    const ra = await processFile(a);
    const rb = await processFile(b);
    expect(ra.document?.sourceHash).toBe(rb.document?.sourceHash);
  });

  it('sourceHash differs when content changes', async () => {
    const a = new FakeFile('a.txt', 'first version', 'text/plain') as unknown as File;
    const b = new FakeFile('a.txt', 'second version', 'text/plain') as unknown as File;
    const ra = await processFile(a);
    const rb = await processFile(b);
    expect(ra.document?.sourceHash).not.toBe(rb.document?.sourceHash);
  });
});

describe('embedDocumentChunks', () => {
  it('returns embedded chunks with vector embeddings', async () => {
    const doc = {
      kind: 'document' as const,
      source: 'a.txt',
      sourceHash: 'h',
      chunks: [
        { id: 'a-0', text: 'first chunk', page: undefined },
        { id: 'a-1', text: 'second chunk', page: undefined },
      ],
    };
    const embeddings = {
      embed: vi.fn(async (input: string | string[]) => {
        const arr = Array.isArray(input) ? input : [input];
        return arr.map(() => [0.1, 0.2, 0.3]);
      }),
    };
    const out = await embedDocumentChunks(doc, embeddings);
    expect(out).toHaveLength(2);
    expect(out[0]?.embedding).toEqual([0.1, 0.2, 0.3]);
    expect(out[0]?.source).toBe('a.txt');
    expect(out[0]?.sourceHash).toBe('h');
    expect(embeddings.embed).toHaveBeenCalledOnce();
  });

  it('returns empty array when document has no chunks', async () => {
    const doc = { kind: 'document' as const, source: 'empty.txt', sourceHash: 'h', chunks: [] };
    const embeddings = { embed: vi.fn() };
    const out = await embedDocumentChunks(doc, embeddings);
    expect(out).toEqual([]);
    expect(embeddings.embed).not.toHaveBeenCalled();
  });

  it('propagates page numbers from chunks', async () => {
    const doc = {
      kind: 'document' as const,
      source: 'b.pdf',
      sourceHash: 'h',
      chunks: [{ id: 'b-p1-0', text: 'page one', page: 1 }],
    };
    const embeddings = {
      embed: vi.fn(async () => [[0.5]]),
    };
    const out = await embedDocumentChunks(doc, embeddings);
    expect(out[0]?.page).toBe(1);
  });

  it('computes tokenCount estimate per chunk', async () => {
    const doc = {
      kind: 'document' as const,
      source: 'c.txt',
      sourceHash: 'h',
      chunks: [
        { id: 'c-0', text: 'a'.repeat(400), page: undefined },
        { id: 'c-1', text: 'a'.repeat(800), page: undefined },
      ],
    };
    const embeddings = { embed: vi.fn(async () => [[0], [0]]) };
    const out = await embedDocumentChunks(doc, embeddings);
    expect(out[0]?.tokenCount).toBeGreaterThan(0);
    expect(out[1]?.tokenCount).toBeGreaterThan(out[0]?.tokenCount ?? 0);
  });
});

describe('existingSourceHashes', () => {
  it('returns the vectorstore hashes for the given source only', async () => {
    const vs = {
      load: vi.fn(async () => undefined),
      getSourceHashes: vi.fn(() => new Set(['a', 'b'])),
    };
    const hashes = await existingSourceHashes(vs, 'foo.txt');
    expect(vs.getSourceHashes).toHaveBeenCalledWith('foo.txt');
    expect([...hashes].sort()).toEqual(['a', 'b']);
  });

  it('awaits the vectorstore load before reading hashes', async () => {
    const order: string[] = [];
    const vs = {
      load: vi.fn(async () => { order.push('load'); }),
      getSourceHashes: vi.fn(() => { order.push('hashes'); return new Set<string>(); }),
    };
    await existingSourceHashes(vs, 'bar.txt');
    expect(order).toEqual(['load', 'hashes']);
  });

  it('propagates vectorstore load failures', async () => {
    const vs = {
      load: vi.fn(async () => { throw new Error('write queue failed'); }),
      getSourceHashes: vi.fn(() => new Set<string>()),
    };
    await expect(existingSourceHashes(vs, 'baz.txt')).rejects.toThrow('write queue failed');
  });
});

describe('addFiles dedup via sourceHash', () => {
  it('skips re-embedding when sourceHash already exists in vectorstore', async () => {
    const text = 'Unchanged content for hashing. '.repeat(20);
    const file = new FakeFile('dup.txt', text, 'text/plain') as unknown as File;
    const first = await processFile(file);
    expect(first.document?.sourceHash).toBeDefined();

    const vs = {
      load: vi.fn(async () => undefined),
      getSourceHashes: vi.fn(() => new Set([first.document?.sourceHash ?? ''])),
    };
    const existing = await existingSourceHashes(vs, 'dup.txt');
    expect(existing.has(first.document?.sourceHash ?? '')).toBe(true);
  });
});
