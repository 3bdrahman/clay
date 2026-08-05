import * as aq from 'arquero';
import type { ColumnTable } from 'arquero';
import { extractPdfText } from './pdf';
import { chunkText, prepareForChunking } from './chunker';
import type { EmbeddingsClient } from '../lib/embeddings';

export type SupportedKind = 'csv' | 'pdf' | 'text' | 'unsupported';

export interface ProcessedDataset {
  kind: 'dataset';
  name: string;
  table: ColumnTable;
  columns: string[];
  rowCount: number;
}

export interface ProcessedDocument {
  kind: 'document';
  source: string;
  chunks: Array<{ id: string; text: string; page?: number }>;
}

export interface ProcessedFile {
  fileName: string;
  kind: SupportedKind;
  dataset?: ProcessedDataset;
  document?: ProcessedDocument;
  error?: string;
}

export function detectKind(fileName: string, mimeType?: string): SupportedKind {
  const lower = fileName.toLowerCase();
  if (lower.endsWith('.csv') || mimeType === 'text/csv') return 'csv';
  if (lower.endsWith('.pdf') || mimeType === 'application/pdf') return 'pdf';
  if (
    lower.endsWith('.md') ||
    lower.endsWith('.markdown') ||
    lower.endsWith('.txt') ||
    lower.endsWith('.text') ||
    lower.endsWith('.json') ||
    mimeType?.startsWith('text/') === true
  ) {
    return 'text';
  }
  return 'unsupported';
}

function deriveName(fileName: string): string {
  return fileName.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || 'dataset';
}

async function processCsv(file: File): Promise<ProcessedDataset> {
  const text = await file.text();
  const table = aq.fromCSV(text);
  const columns = table.columnNames();
  const rowCount = typeof table.numRows === 'function' ? table.numRows() : 0;
  return {
    kind: 'dataset',
    name: deriveName(file.name),
    table,
    columns,
    rowCount,
  };
}

async function processText(file: File): Promise<ProcessedDocument> {
  const text = await file.text();
  const chunks = chunkText(text);
  return {
    kind: 'document',
    source: file.name,
    chunks: chunks.map((c, i) => ({ id: `${deriveName(file.name)}-${i}`, text: c.text })),
  };
}

async function processPdf(file: File): Promise<ProcessedDocument> {
  const buffer = await file.arrayBuffer();
  const pages = await extractPdfText(buffer);
  const base = deriveName(file.name);
  const chunks: Array<{ id: string; text: string; page?: number }> = [];
  for (const page of pages) {
    const pageChunks = chunkText(page.text);
    for (let i = 0; i < pageChunks.length; i++) {
      chunks.push({
        id: `${base}-p${page.pageNumber}-${i}`,
        text: pageChunks[i].text,
        page: page.pageNumber,
      });
    }
  }
  return { kind: 'document', source: file.name, chunks };
}

export async function processFile(file: File): Promise<ProcessedFile> {
  const kind = detectKind(file.name, file.type);
  const MAX_BYTES = 25 * 1024 * 1024;
  if (file.size > MAX_BYTES) {
    return {
      fileName: file.name,
      kind,
      error: `File too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Max 25MB.`,
    };
  }
  try {
    if (kind === 'csv') {
      const dataset = await processCsv(file);
      return { fileName: file.name, kind, dataset };
    }
    if (kind === 'pdf') {
      const document = await processPdf(file);
      return { fileName: file.name, kind, document };
    }
    if (kind === 'text') {
      const document = await processText(file);
      return { fileName: file.name, kind, document };
    }
    return { fileName: file.name, kind, error: 'Unsupported file type' };
  } catch (e) {
    return {
      fileName: file.name,
      kind,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

export async function embedDocumentChunks(
  document: ProcessedDocument,
  embeddings: EmbeddingsClient,
): Promise<Array<{ id: string; text: string; source: string; page?: number; embedding: number[] }>> {
  if (document.chunks.length === 0) return [];
  const texts = document.chunks.map(c => c.text);
  const vectors = await embeddings.embed(texts, { inputType: 'passage' });
  return document.chunks.map((c, i) => ({
    id: c.id,
    text: c.text,
    source: document.source,
    page: c.page,
    embedding: vectors[i] ?? [],
  }));
}

export { prepareForChunking };
