import type { ChunkMetadata } from './types';

export interface VectorEntry {
  id: string;
  text: string;
  embedding: Float32Array;
  metadata: ChunkMetadata;
}

export function getSourceHashes(
  source: string,
  memory: Map<string, VectorEntry>
): Set<string> {
  const hashes = new Set<string>();
  for (const entry of memory.values()) {
    if (entry.metadata.source === source && entry.metadata.sourceHash) {
      hashes.add(entry.metadata.sourceHash);
    }
  }
  return hashes;
}

export function listSources(
  memory: Map<string, VectorEntry>
): Array<{ source: string; entryCount: number }> {
  const counts = new Map<string, number>();
  for (const entry of memory.values()) {
    counts.set(entry.metadata.source, (counts.get(entry.metadata.source) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([source, entryCount]) => ({ source, entryCount }))
    .sort((a, b) => a.source.localeCompare(b.source));
}