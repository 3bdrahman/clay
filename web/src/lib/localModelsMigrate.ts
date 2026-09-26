import type { LocalModelPicks, PickedModelsOverride } from './types';

export function migrateChatSelection(raw: unknown): PickedModelsOverride {
  if (typeof raw !== 'object' || raw === null) return { chatModel: '' };
  const chatModel = 'chatModel' in raw && typeof raw.chatModel === 'string'
    ? raw.chatModel
    : [
      'answer' in raw ? raw.answer : undefined,
      'routing' in raw ? raw.routing : undefined,
      'codeGen' in raw ? raw.codeGen : undefined,
      'eval' in raw ? raw.eval : undefined,
    ].find(value => typeof value === 'string' && value.trim());
  return {
    chatModel: typeof chatModel === 'string' ? chatModel : '',
  };
}

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
 * chat-only shape. Prefer the existing `answer` field for `chat` (the user's
 * main answer model), then `routing`, then the first non-empty of
 * `codeGen`/`eval`. Legacy embedding picks are dropped: embeddings are local
 * (transformers.js) and never user-selected.
 */
export function migrateLegacyLocalModels(
  raw: Partial<LegacyLocalModelPicks> | undefined,
): LocalModelPicks {
  if (raw === undefined) return { chat: '' };
  const firstNonEmpty = (xs: Array<string | undefined>): string =>
    xs.find(x => x && x.trim()) ?? '';
  const chat =
    firstNonEmpty([raw.answer, raw.routing, raw.codeGen, raw.eval]) ||
    (raw.chat ?? '');
  return { chat };
}
