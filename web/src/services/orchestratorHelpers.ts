/**
 * Helpers for the workflow orchestrator: HyDE expansion, parallel fan-out
 * retrieval (topK=8 → rerank to 4), parallel grading with early-exit, and
 * heading-aware citation formatting.
 */
import type { Citation, Document } from '../lib/types';

export interface HyDEOptions {
  llm: { invoke: (req: { system?: string; messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>; jsonMode?: boolean; temperature?: number; model?: string }) => Promise<{ content: string }> };
  model?: string;
}

const HYDE_INSTRUCTIONS =
  'You generate hypothetical documents that would answer a question. ' +
  'Write a short (3-5 sentence) paragraph as if it were a passage from ' +
  'a relevant document. No preamble.';

const HYDE_PROMPT =
  'Question: {question}\n\n' +
  'Write a hypothetical passage (no preamble) that directly answers this question.';

/**
 * HyDE (Hypothetical Document Embeddings): generate a hypothetical passage
 * that would answer the question, then use it for dense retrieval.
 * Falls back to the original question if generation fails.
 */
export async function expandHyDE(question: string, opts: HyDEOptions): Promise<string> {
  try {
    const resp = await opts.llm.invoke({
      system: HYDE_INSTRUCTIONS,
      messages: [{ role: 'user', content: HYDE_PROMPT.replace('{question}', question) }],
      temperature: 0,
      ...(opts.model !== undefined ? { model: opts.model } : {}),
    });
    const text = (resp.content || '').trim();
    if (text.length > 0) return `${question}\n\n${text}`;
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[orchestratorHelpers] expandHyDE failed (falling back to original question):', e);
    }
  }
  return question;
}

/**
 * Score a candidate document against a hypothetical passage using
 * length-normalized token overlap (BM25-lite reranker, zero deps).
 * Higher score = more relevant.
 */
function rerankScore(docText: string, hypothetical: string): number {
  const docTokens = new Set(docText.toLowerCase().split(/\W+/).filter(Boolean));
  const hypTokens = hypothetical.toLowerCase().split(/\W+/).filter(Boolean);
  if (hypTokens.length === 0 || docTokens.size === 0) return 0;
  let hits = 0;
  for (const t of hypTokens) if (docTokens.has(t)) hits++;
  return hits / hypTokens.length;
}

/**
 * Parallel fan-out retrieval: run vector search over both the original
 * query and the HyDE hypothetical, then rerank the union down to `rerankK`
 * with length-normalized token overlap.
 */
export async function parallelFanOut(
  fetch: (query: string, k: number) => Promise<Document[]>,
  question: string,
  hypothetical: string,
  initialK: number,
  rerankK: number,
): Promise<Document[]> {
  const [primary, secondary] = await Promise.all([
    fetch(question, initialK),
    fetch(hypothetical, initialK),
  ]);
  const seen = new Set<string>();
  const merged: Document[] = [];
  for (const d of [...primary, ...secondary]) {
    if (seen.has(d.id)) continue;
    seen.add(d.id);
    merged.push(d);
  }
  const scored = merged.map(d => ({ d, s: rerankScore(d.content, hypothetical) || (d.score ?? 0) }));
  scored.sort((a, b) => b.s - a.s);
  return scored.slice(0, rerankK).map(x => x.d);
}

export interface GradeFn {
  (doc: Document): Promise<'relevant' | 'irrelevant' | 'keep-on-error'>;
}

export interface ParallelGradeOptions {
  earlyExitAt?: number;
  signal?: AbortSignal;
}

/**
 * Grade documents in parallel with an early-exit once `earlyExitAt`
 * relevant docs have been found.
 */
export async function parallelGrade(
  docs: Document[],
  grade: GradeFn,
  opts: ParallelGradeOptions = {},
): Promise<Document[]> {
  const earlyExitAt = opts.earlyExitAt ?? docs.length;
  const settled = await Promise.all(
    docs.map(async (d) => {
      if (opts.signal?.aborted) return { d, outcome: 'irrelevant' as const };
      try {
        const outcome = await grade(d);
        return { d, outcome };
      } catch (e) {
        if (import.meta.env.DEV) {
          console.warn('[orchestratorHelpers] parallelGrade: grade() failed (marking "keep-on-error"):', e);
        }
        return { d, outcome: 'keep-on-error' as const };
      }
    }),
  );
  const relevant: Document[] = [];
  for (const { d, outcome } of settled) {
    if (outcome === 'relevant' || outcome === 'keep-on-error') {
      relevant.push(d);
      if (relevant.length >= earlyExitAt) break;
    }
  }
  return relevant;
}

/**
 * Format a heading-aware citation excerpt. If the chunk's metadata has a
 * heading, the excerpt is prefixed with `[heading]`.
 */
export function formatHeadingCitation(d: Document): Citation {
  const heading = (d.metadata?.['heading'] as string | undefined) ?? undefined;
  const excerpt = heading
    ? `[${heading}] ${d.content.slice(0, 200)}`
    : d.content.slice(0, 200);
  return {
    source: d.source,
    ...(d.page !== undefined ? { page: d.page } : {}),
    excerpt,
    type: 'vectorstore',
  };
}
