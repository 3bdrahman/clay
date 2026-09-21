// Answer composer — produces factual answers from real retrieved context
// when no LLM API is configured. Extracts relevant sentences from documents,
// formats real data analysis results, and assembles citations from real sources.

import type { Document, DataAnalysisResult, WebResult, WorkflowState } from '../lib/types';

const STOP_WORDS = new Set([
  'a','an','and','are','as','at','be','by','for','from','has','have','he','her','his',
  'i','in','is','it','its','of','on','or','our','she','that','the','their','them',
  'they','this','to','was','we','were','what','when','where','which','who','why',
  'will','with','you','your','about','can','could','did','do','does','doing','had',
  'if','just','like','me','my','no','not','so','some','than','then','there','these',
  'those','too','us','very','would',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2 && !STOP_WORDS.has(w));
}

function scoreSentence(sentence: string, queryWords: Set<string>): number {
  const words = tokenize(sentence);
  if (words.length === 0) return 0;
  let hits = 0;
  for (const w of words) {
    if (queryWords.has(w)) hits++;
  }
  return hits / Math.sqrt(words.length);
}

function splitSentences(text: string): string[] {
  return text
    .replace(/\s+/g, ' ')
    .split(/(?<=[.!?])\s+(?=[A-Z(])/)
    .map(s => s.trim())
    .filter(s => s.length > 20 && s.length < 400);
}

export function extractRelevantSentences(
  documents: Document[],
  question: string,
  maxSentences: number,
): Array<{ text: string; source: string; page?: number; score: number }> {
  const queryWords = new Set(tokenize(question));
  const candidates: Array<{ text: string; source: string; page?: number; score: number }> = [];

  for (const doc of documents) {
    for (const sentence of splitSentences(doc.content)) {
      const score = scoreSentence(sentence, queryWords);
      if (score > 0) {
        candidates.push({
          text: sentence,
          source: doc.source,
          page: doc.page,
          score,
        });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  return candidates.slice(0, maxSentences);
}

export function formatDataResult(analysis: DataAnalysisResult): string {
  const r = analysis.result;
  if (r == null) return '';

  if (Array.isArray(r)) {
    if (r.length === 0) return 'The analysis returned no rows.';
    const cols = Object.keys(r[0] as Record<string, unknown>);
    const header = cols.join(' | ');
    const rows = r.slice(0, 8).map(row =>
      cols.map(c => String((row as Record<string, unknown>)[c] ?? '')).join(' | ')
    );
    return `| ${header} |\n| ${cols.map(() => '---').join(' | ')} |\n${rows.map(r => `| ${r} |`).join('\n')}`;
  }

  if (typeof r === 'object') {
    const entries = Object.entries(r as Record<string, unknown>);
    return entries
      .map(([k, v]) => {
        if (Array.isArray(v)) {
          return `**${k}:** ${v.length} item${v.length === 1 ? '' : 's'} — ${v
            .slice(0, 5)
            .map(x => (typeof x === 'object' && x ? JSON.stringify(x) : String(x)))
            .join(', ')}`;
        }
        if (typeof v === 'number') {
          const formatted = Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
          return `**${k}:** ${formatted}`;
        }
        return `**${k}:** ${String(v)}`;
      })
      .join('\n');
  }

  return `Result: ${String(r)}`;
}

export function composeVectorstoreAnswer(
  question: string,
  documents: Document[],
): string {
  if (documents.length === 0) {
    return "I couldn't find relevant documents to answer your question.";
  }

  const top = documents[0];
  const sentences = extractRelevantSentences(documents, question, 4);

  const intro = sentences.length > 0
    ? sentences[0].text
    : top.content.slice(0, 300).replace(/\s+/g, ' ').trim();

  const supporting = sentences.slice(1);

  const parts: string[] = [intro];
  if (supporting.length > 0) {
    parts.push('\n\nAdditional relevant information:');
    for (const s of supporting) {
      parts.push(`- ${s.text}`);
    }
  }

  return parts.join('\n');
}

export function composeDataAnswer(
  _question: string,
  analysis: DataAnalysisResult,
): string {
  const formatted = formatDataResult(analysis);
  if (!formatted) {
    return analysis.explanation || 'The analysis produced no results.';
  }
  return `${formatted}\n\n_${analysis.explanation || 'Computed from internal datasets.'}_`;
}

export function composeWebAnswer(
  _question: string,
  results: WebResult[],
): string {
  if (results.length === 0) {
    return "I couldn't find current web results for this question.";
  }
  const top = results[0];
  const others = results.slice(1);
  const parts = [top.content.slice(0, 500).trim()];
  if (others.length > 0) {
    parts.push('\n\nRelated sources:');
    for (const r of others) {
      parts.push(`- **${r.title}**: ${r.content.slice(0, 150).trim()}…`);
    }
  }
  return parts.join('\n');
}

export function composeAnswerFromState(state: WorkflowState, question: string): string {
  const parts: string[] = [];

  if (state.dataAnalysis && state.dataAnalysis.resultType !== 'error') {
    parts.push(composeDataAnswer(question, state.dataAnalysis));
  }

  if (state.documents.length > 0) {
    parts.push(composeVectorstoreAnswer(question, state.documents));
  }

  if (state.webResults.length > 0) {
    parts.push(composeWebAnswer(question, state.webResults));
  }

  if (parts.length === 0) {
    return "I couldn't find relevant information to answer your question.";
  }

  return parts.join('\n\n---\n\n');
}

export function buildCitationsFromState(state: WorkflowState): string {
  const sources: string[] = [];
  let idx = 1;

  for (const d of state.documents) {
    const excerpt = d.content.replace(/\s+/g, ' ').slice(0, 160).trim();
    sources.push(`[${idx}] ${d.source}${d.page ? `, p. ${d.page}` : ''} — ${excerpt}…`);
    idx++;
  }

  if (state.dataAnalysis && state.dataAnalysis.resultType !== 'error') {
    sources.push(`[${idx}] Data Analysis (${state.dataAnalysis.code.split('\n')[0].slice(0, 60)}…)`);
    idx++;
  }

  for (const w of state.webResults) {
    sources.push(`[${idx}] ${w.title}${w.url ? ` — ${w.url}` : ''}`);
    idx++;
  }

  return sources.join('\n');
}
