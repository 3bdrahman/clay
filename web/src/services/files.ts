import * as aq from 'arquero';
import type { ColumnTable } from 'arquero';
import { extractPdfText, type ExtractedPage } from './pdf';
import { chunkText, type Chunk, type ChunkContext } from './chunker';
import type { EmbeddingsClient } from '../lib/embeddings';
import { hashText } from '../lib/hash';

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
  sourceHash: string;
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
  return fileName
    .replace(/\.[^.]+$/, '')
    .replace(/[^a-zA-Z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'dataset';
}

function toChunks(
  baseId: string,
  text: string,
  ctx: ChunkContext,
  page?: number,
): Array<{ id: string; text: string; page?: number }> {
  const pieces: Chunk[] = chunkText(text, {}, { ...ctx, startPage: page });
  return pieces.map((c) => ({
    id: `${baseId}-${c.index}`,
    text: c.text,
    page,
  }));
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
  const sourceHash = hashText(text);
  const ctx: ChunkContext = { source: file.name, sourceHash, modelId: 'chunker-v2' };
  const base = deriveName(file.name);
  return {
    kind: 'document',
    source: file.name,
    sourceHash,
    chunks: toChunks(base, text, ctx),
  };
}

async function processPdf(file: File): Promise<ProcessedDocument> {
  const buffer = await file.arrayBuffer();
  const pages: ExtractedPage[] = await extractPdfText(buffer);
  const fullText = pages.map((p) => p.text).join('\n');
  const sourceHash = hashText(fullText);
  const ctx: ChunkContext = { source: file.name, sourceHash, modelId: 'chunker-v2' };
  const base = deriveName(file.name);
  const chunks: Array<{ id: string; text: string; page?: number }> = [];
  for (const page of pages) {
    chunks.push(...toChunks(`${base}-p${page.pageNumber}`, page.text, ctx, page.pageNumber));
  }
  return { kind: 'document', source: file.name, sourceHash, chunks };
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

export interface EmbeddedChunk {
  id: string;
  text: string;
  source: string;
  sourceHash: string;
  page?: number;
  heading?: string;
  embedding: number[];
  charStart: number;
  charEnd: number;
  chunkIndex: number;
  tokenCount: number;
}

/**
 * Returns the set of sourceHashes currently present in the vectorstore for
 * the given source. Used to short-circuit re-embedding unchanged content.
 */
export async function existingSourceHashes(
  vs: { stats: { entries: number }; similaritySearch: (q: string, k: number) => Promise<Array<{ metadata?: Record<string, unknown> }>> },
  source: string,
): Promise<Set<string>> {
  const hits = await vs.similaritySearch(source, vs.stats.entries || 1);
  const hashes = new Set<string>();
  for (const h of hits) {
    const hash = h.metadata?.['sourceHash'];
    if (typeof hash === 'string' && hash.length > 0) hashes.add(hash);
  }
  return hashes;
}

export async function embedDocumentChunks(
  document: ProcessedDocument,
  embeddings: EmbeddingsClient,
): Promise<EmbeddedChunk[]> {
  if (document.chunks.length === 0) return [];
  const texts = document.chunks.map((c) => c.text);
  const vectors = await embeddings.embed(texts, { inputType: 'passage' });
  return document.chunks.map((c, i) => {
    const vec = vectors[i] ?? [];
    return {
      id: c.id,
      text: c.text,
      source: document.source,
      sourceHash: document.sourceHash,
      page: c.page,
      embedding: vec,
      charStart: 0,
      charEnd: c.text.length,
      chunkIndex: i,
      tokenCount: Math.max(1, Math.ceil(c.text.length / 4)),
    };
  });
}
