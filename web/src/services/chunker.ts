import type { ChunkMetadata } from '../lib/types';
import { estimateTokens } from '../lib/tokens';
import type { Section, SizeFn } from './chunkerPrimitives';
import { packSentences, splitByHeadings, splitBySentence } from './chunkerPrimitives';

export interface ChunkOptions {
  chunkSize?: number;
  overlap?: number;
  tokenBudget?: number;
  overlapTokens?: number;
}

export interface ChunkContext {
  source?: string;
  sourceHash?: string;
  modelId?: string;
  startPage?: number;
}

export interface Chunk {
  readonly text: string;
  readonly index: number;
  readonly metadata?: ChunkMetadata;
}

export const DEFAULT_CHUNK_SIZE = 800;
export const DEFAULT_OVERLAP = 200;
export const DEFAULT_TOKEN_BUDGET = 512;
export const DEFAULT_OVERLAP_TOKENS = 64;
export const MIN_CHUNK_TOKENS = 64;

interface Resolved {
  readonly sizeOf: SizeFn;
  readonly budget: number;
  readonly overlap: number;
}
interface MetaCtx {
  readonly source: string;
  readonly sourceHash: string;
  readonly modelId: string;
  readonly page?: number;
}

function resolveOptions(opts: ChunkOptions, ctx: ChunkContext): Resolved {
  const hasNew = opts.tokenBudget !== undefined || opts.overlapTokens !== undefined;
  const hasOld = opts.chunkSize !== undefined || opts.overlap !== undefined;
  const hasCtx =
    ctx.source !== undefined ||
    ctx.sourceHash !== undefined ||
    ctx.modelId !== undefined ||
    ctx.startPage !== undefined;
  const cap = (b: number, o: number): number => Math.min(o, Math.max(0, Math.floor(b / 2)));
  if ((hasOld && !hasNew) || (!hasNew && !hasCtx)) {
    const b = opts.chunkSize ?? DEFAULT_CHUNK_SIZE;
    const o = opts.overlap ?? DEFAULT_OVERLAP;
    return { sizeOf: (t) => t.length, budget: b, overlap: cap(b, o) };
  }
  const b = opts.tokenBudget ?? DEFAULT_TOKEN_BUDGET;
  const o = opts.overlapTokens ?? DEFAULT_OVERLAP_TOKENS;
  return { sizeOf: estimateTokens, budget: b, overlap: cap(b, o) };
}

interface Emit {
  readonly text: string;
  readonly index: number;
  readonly charStart: number;
}

function buildMetadata(emit: Emit, section: Section, mctx: MetaCtx): ChunkMetadata {
  return {
    source: mctx.source,
    sourceHash: mctx.sourceHash,
    page: mctx.page,
    heading: section.heading,
    charStart: emit.charStart,
    charEnd: emit.charStart + emit.text.length,
    chunkIndex: emit.index,
    tokenCount: estimateTokens(emit.text),
    modelId: mctx.modelId,
  };
}

export function chunkText(text: string, opts: ChunkOptions = {}, ctx: ChunkContext = {}): Chunk[] {
  const resolved = resolveOptions(opts, ctx);
  const mctx: MetaCtx = {
    source: ctx.source ?? '',
    sourceHash: ctx.sourceHash ?? '',
    modelId: ctx.modelId ?? '',
    page: ctx.startPage,
  };

  if (text.trim() === '') {
    return [{ text: '', index: 0 }];
  }

  const sections = splitByHeadings(text);
  const chunks: Chunk[] = [];
  let chunkIndex = 0;

  for (let sIdx = 0; sIdx < sections.length; sIdx++) {
    const section = sections[sIdx];
    const body = section.body.replace(/\s+/g, ' ').trim();
    if (body.length === 0) continue;
    const sentences = splitBySentence(body);
    const packed = packSentences(sentences, resolved);
    const isLastSection = sIdx === sections.length - 1;

    for (const piece of packed) {
      const pieceTokens = estimateTokens(piece);
      if (!isLastSection && pieceTokens < MIN_CHUNK_TOKENS && packed.length === 1) {
        continue;
      }
      const localOffset = body.indexOf(piece);
      const charStart = section.startChar + (localOffset >= 0 ? localOffset : 0);
      chunks.push({
        text: piece,
        index: chunkIndex,
        metadata: buildMetadata({ text: piece, index: chunkIndex, charStart }, section, mctx),
      });
      chunkIndex++;
    }
  }

  if (chunks.length === 0) {
    const clean = text.replace(/\s+/g, ' ').trim();
    return [
      {
        text: clean,
        index: 0,
        metadata: buildMetadata({ text: clean, index: 0, charStart: 0 }, sections[0], mctx),
      },
    ];
  }
  return chunks;
}

export function chunkSentences(sentences: string[], opts: ChunkOptions = {}): Chunk[] {
  if (sentences.length === 0) return [];
  const resolved = resolveOptions(opts, {});
  const packed = packSentences(sentences, resolved);
  const emptyMctx: MetaCtx = {
    source: '',
    sourceHash: '',
    modelId: '',
    page: undefined,
  };
  const emptySection: Section = { heading: undefined, body: '', startChar: 0 };
  return packed.map((p, i) => ({
    text: p,
    index: i,
    metadata: buildMetadata({ text: p, index: i, charStart: 0 }, emptySection, emptyMctx),
  }));
}

export function prepareForChunking(text: string): string[] {
  return splitBySentence(text);
}
