import type { Document, ChunkMetadata } from './types';
import type { EmbeddingsClient } from './embeddings';
import type { HnswIndex } from './hnsw';
import { classifyError } from './errors';
import { cosineUnit, mmrSelect, type ScoredCandidate } from './vectorSearchMath';

export interface VectorEntry {
  id: string;
  text: string;
  embedding: Float32Array;
  metadata: ChunkMetadata;
}

export interface SearchPipelineConfig {
  topK: number;
  scoreThreshold: number;
  useMMR: boolean;
  mmrLambda: number;
  annThreshold: number;
  annEfSearch: number;
  denseTopKMultiplier: number;
}

export interface SearchPipelineDeps {
  memory: Map<string, VectorEntry>;
  knownDimension: number | null;
  annIndex: HnswIndex | null;
  ensureAnnIndex: () => HnswIndex | null;
  embeddings: EmbeddingsClient;
  config: SearchPipelineConfig;
}

function bruteForceCandidates(
  queryEmbedding: Float32Array,
  poolK: number,
  memory: Map<string, VectorEntry>
): ScoredCandidate[] {
  const scored: ScoredCandidate[] = [];
  for (const e of memory.values()) {
    scored.push({ id: e.id, score: cosineUnit(queryEmbedding, e.embedding) });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, poolK);
}

export async function searchPipeline(
  query: string,
  deps: SearchPipelineDeps
): Promise<Document[]> {
  const { memory, ensureAnnIndex, embeddings, config } = deps;
  const topK = config.topK;
  if (memory.size === 0) return [];
  let queryEmbeddingRaw: number[][];
  try {
    queryEmbeddingRaw = await embeddings.embed(query);
  } catch (e) {
    throw classifyError(e, 'embeddings', 'similaritySearch');
  }
  const queryRaw = queryEmbeddingRaw[0];
  if (!queryRaw) return [];
  const queryEmbedding = Float32Array.from(queryRaw);

  // Candidate pool: the ANN index at or above the engage threshold
  // (sub-linear, approximate recall), brute-force cosine below it (exact).
  const poolK = Math.max(topK * config.denseTopKMultiplier, 6);
  let denseCandidates: ScoredCandidate[];
  if (memory.size >= config.annThreshold) {
    const index = ensureAnnIndex();
    denseCandidates = index
      ? index
          .search(queryEmbedding, poolK, Math.max(poolK, config.annEfSearch))
          .map(f => ({ id: f.label, score: f.score }))
      : bruteForceCandidates(queryEmbedding, poolK, memory);
  } else {
    denseCandidates = bruteForceCandidates(queryEmbedding, poolK, memory);
  }

  const filtered = denseCandidates.filter((c) => c.score >= config.scoreThreshold);

  let chosen: string[];
  if (config.useMMR) {
    const embMap = new Map<string, Float32Array>();
    for (const id of new Set(filtered.map((f) => f.id))) {
      const e = memory.get(id);
      if (e) embMap.set(id, e.embedding);
    }
    chosen = mmrSelect(filtered, embMap, topK, config.mmrLambda);
  } else {
    chosen = filtered.slice(0, topK).map((c) => c.id);
  }

  const scoreLookup = new Map(denseCandidates.map((m) => [m.id, m.score]));
  const results: Document[] = [];
  for (const id of chosen) {
    const e = memory.get(id);
    if (!e) continue;
    const score = scoreLookup.get(id) ?? 0;
    const doc: Document = {
      id: e.id,
      content: e.text,
      source: e.metadata.source,
      page: e.metadata.page,
      score,
      metadata: e.metadata as unknown as Record<string, unknown>,
    };
    results.push(doc);
  }
  return results;
}