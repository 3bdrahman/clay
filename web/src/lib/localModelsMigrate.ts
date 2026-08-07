import type { LocalModelPicks } from './types';

export interface LegacyLocalModelPicks {
  routing?: string;
  codeGen?: string;
  answer?: string;
  eval?: string;
  embedding?: string;
  chat?: string;
  embeddings?: string;
}

/**
 * One-way migration from the legacy 5-slot LocalModelPicks shape to the
 * collapsed 2-slot chat+embeddings shape. Prefer the existing `answer` field
 * for `chat` (the user's main answer model), then `routing`, then the first
 * non-empty of `codeGen`/`eval`. `embedding` becomes `embeddings`. If `chat`
 * already exists (already-migrated state), leave both fields untouched.
 */
export function migrateLegacyLocalModels(
  raw: Partial<LegacyLocalModelPicks> | undefined,
): LocalModelPicks {
  if (raw === undefined) return { chat: '', embeddings: '' };
  if (typeof raw.chat === 'string' && typeof raw.embeddings === 'string') {
    return { chat: raw.chat, embeddings: raw.embeddings };
  }
  const firstNonEmpty = (xs: Array<string | undefined>): string =>
    xs.find(x => x && x.trim()) ?? '';
  const chat =
    firstNonEmpty([raw.answer, raw.routing, raw.codeGen, raw.eval]) ||
    (raw.chat ?? '');
  const embeddings = raw.embedding ?? raw.embeddings ?? '';
  return { chat, embeddings };
}
