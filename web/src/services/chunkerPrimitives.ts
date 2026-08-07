export const HEADING_RE = /^(#{1,6})\s+(.+)$/gm;
export const SENTENCE_SPLIT_RE = /(?<=[.!?])\s+(?=[A-Z])|\n\n+/g;

export interface Section {
  readonly heading: string | undefined;
  readonly body: string;
  readonly startChar: number;
}

export type SizeFn = (text: string) => number;

export function splitBySentence(text: string): string[] {
  return text.split(SENTENCE_SPLIT_RE).filter((s) => s.trim().length > 0);
}

export function splitByHeadings(text: string): Section[] {
  const matches = [...text.matchAll(HEADING_RE)];
  if (matches.length === 0) {
    return [{ heading: undefined, body: text, startChar: 0 }];
  }
  const sections: Section[] = [];
  const first = matches[0];
  if (first.index !== undefined && first.index > 0) {
    const pre = text.slice(0, first.index);
    if (pre.trim().length > 0) {
      sections.push({ heading: undefined, body: pre, startChar: 0 });
    }
  }
  for (let i = 0; i < matches.length; i++) {
    const m = matches[i];
    if (m.index === undefined) continue;
    const heading = m[2].trim();
    const lineEnd = text.indexOf('\n', m.index);
    const bodyStart = lineEnd === -1 ? text.length : lineEnd + 1;
    const nextStart =
      i + 1 < matches.length && matches[i + 1].index !== undefined
        ? matches[i + 1].index
        : text.length;
    sections.push({
      heading,
      body: text.slice(bodyStart, nextStart),
      startChar: bodyStart,
    });
  }
  return sections;
}

export function hardSplit(sentence: string, budget: number, overlap: number): string[] {
  if (sentence.length <= budget) return [sentence];
  const head = sentence.slice(0, budget);
  const lastPunct = head.search(/[.!?](?:\s|$)/);
  if (lastPunct > budget * 0.5) {
    return [
      sentence.slice(0, lastPunct + 1),
      ...hardSplit(sentence.slice(lastPunct + 1), budget, overlap),
    ];
  }
  const pieces: string[] = [];
  let i = 0;
  while (i < sentence.length) {
    const end = Math.min(i + budget, sentence.length);
    pieces.push(sentence.slice(i, end));
    if (end >= sentence.length) break;
    i = Math.max(end - overlap, i + 1);
  }
  return pieces;
}

export function packSentences(
  sentences: readonly string[],
  resolved: { readonly sizeOf: SizeFn; readonly budget: number; readonly overlap: number }
): string[] {
  const { sizeOf, budget, overlap } = resolved;
  const emitted: string[] = [];
  let buffer: string[] = [];
  let carried: string[] = [];

  const flush = (sizeOf: SizeFn, overlap: number): void => {
    if (buffer.length === 0) return;
    const text = buffer.join(' ').trim();
    if (text.length === 0) {
      buffer = [];
      return;
    }
    emitted.push(text);
    const trailing: string[] = [];
    let ov = 0;
    for (let k = buffer.length - 1; k >= 0 && ov < overlap; k--) {
      const cand = [...trailing, buffer[k]].join(' ');
      if (sizeOf(cand) > overlap && trailing.length > 0) break;
      trailing.unshift(buffer[k]);
      ov = sizeOf(trailing.join(' '));
    }
    carried = trailing;
    buffer = [];
  };

  for (const sentence of sentences) {
    if (sizeOf(sentence) > budget) {
      flush(sizeOf, overlap);
      const pieces = hardSplit(sentence, budget, overlap);
      const firstWithCarry =
        carried.length > 0 ? `${carried.join(' ')} ${pieces[0]}` : pieces[0];
      emitted.push(firstWithCarry);
      for (let idx = 1; idx < pieces.length; idx++) emitted.push(pieces[idx]);
      carried = pieces.length > 0 ? [pieces[pieces.length - 1]] : [];
      continue;
    }
    if (buffer.length > 0) {
      const joined = [...buffer, sentence].join(' ');
      if (sizeOf(joined) > budget) {
        flush(sizeOf, overlap);
        buffer = carried.length > 0 ? [...carried, sentence] : [sentence];
        continue;
      }
      buffer = [...buffer, sentence];
    } else {
      buffer = carried.length > 0 ? [...carried, sentence] : [sentence];
    }
  }
  flush(sizeOf, overlap);
  return emitted;
}
