export interface ChunkOptions {
  chunkSize?: number;
  overlap?: number;
}

export interface Chunk {
  text: string;
  index: number;
}

const DEFAULT_CHUNK_SIZE = 800;
const DEFAULT_OVERLAP = 200;

function splitBySentence(text: string): string[] {
  return text.split(/(?<=[.!?])\s+(?=[A-Z])|\n\n+/g).filter(s => s.trim().length > 0);
}

export function chunkText(text: string, opts: ChunkOptions = {}): Chunk[] {
  const chunkSize = opts.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const overlap = opts.overlap ?? DEFAULT_OVERLAP;
  const cleanText = text.replace(/\s+/g, ' ').trim();

  if (cleanText.length <= chunkSize) {
    return [{ text: cleanText, index: 0 }];
  }

  const chunks: Chunk[] = [];
  let start = 0;
  let index = 0;

  while (start < cleanText.length) {
    let end = Math.min(start + chunkSize, cleanText.length);

    if (end < cleanText.length) {
      const lookahead = cleanText.slice(start, end);
      const lastSentence = Math.max(
        lookahead.lastIndexOf('. '),
        lookahead.lastIndexOf('! '),
        lookahead.lastIndexOf('? '),
        lookahead.lastIndexOf('\n'),
      );
      if (lastSentence > chunkSize * 0.5) {
        end = start + lastSentence + 1;
      }
    }

    const piece = cleanText.slice(start, end).trim();
    if (piece.length > 0) chunks.push({ text: piece, index: index++ });

    if (end >= cleanText.length) break;
    start = Math.max(end - overlap, start + 1);
  }

  return chunks;
}

export function chunkSentences(sentences: string[], opts: ChunkOptions = {}): Chunk[] {
  const chunkSize = opts.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const overlap = opts.overlap ?? DEFAULT_OVERLAP;
  const chunks: Chunk[] = [];
  let buffer = '';
  let index = 0;

  for (const sentence of sentences) {
    const candidate = buffer ? `${buffer} ${sentence}` : sentence;
    if (candidate.length > chunkSize && buffer) {
      chunks.push({ text: buffer.trim(), index: index++ });
      const words = buffer.split(/\s+/);
      const keepWords = Math.max(0, words.length - Math.floor(overlap / 6));
      buffer = words.slice(keepWords).join(' ') + ' ' + sentence;
    } else {
      buffer = candidate;
    }
  }

  if (buffer.trim()) chunks.push({ text: buffer.trim(), index: index++ });
  return chunks;
}

export function prepareForChunking(text: string): string[] {
  return splitBySentence(text);
}
