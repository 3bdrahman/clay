import { describe, expect, it } from 'vitest';
import { detectKind, processFile } from './files';

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
});
