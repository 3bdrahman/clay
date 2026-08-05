import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createVectorStore } from './vectorstore';
import type { EmbeddingsClient } from './embeddings';

const mockEmbeddings: EmbeddingsClient = {
  embed: vi.fn(async (input: string | string[]) => {
    const arr = Array.isArray(input) ? input : [input];
    return arr.map(() => new Array(1024).fill(0).map(() => Math.random()));
  }),
};

describe('createVectorStore', () => {
  let vectorstore: ReturnType<typeof createVectorStore>;

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    vectorstore = createVectorStore(mockEmbeddings);
  });

  it('starts empty', async () => {
    await vectorstore.load();
    expect(vectorstore.stats.entries).toBe(0);
  });

  it('adds entries and caches to localStorage', async () => {
    await vectorstore.load();
    const entries = [
      { id: '1', text: 'hello world', source: 'test.txt', embedding: new Array(1024).fill(0.1) },
      { id: '2', text: 'foo bar', source: 'test.txt', embedding: new Array(1024).fill(0.2) },
    ];
    vectorstore.addEntries(entries);
    expect(vectorstore.stats.entries).toBe(2);

    // Verify localStorage cache
    const cached = JSON.parse(localStorage.getItem('clay-vector-entries-v1') ?? '[]');
    expect(cached).toHaveLength(2);
  });

  it('persists across load calls', async () => {
    await vectorstore.load();
    vectorstore.addEntries([
      { id: '1', text: 'persistent', source: 'test.txt', embedding: new Array(1024).fill(0.5) },
    ]);

    // Create new vectorstore instance (simulates reload)
    const vs2 = createVectorStore(mockEmbeddings);
    await vs2.load();
    expect(vs2.stats.entries).toBe(1);
  });

  it('similaritySearch returns empty when no entries', async () => {
    await vectorstore.load();
    const results = await vectorstore.similaritySearch('query', 4);
    expect(results).toEqual([]);
  });

  it('similaritySearch returns top-K by cosine similarity', async () => {
    await vectorstore.load();
    const base = new Array(1024).fill(0);
    base[0] = 1;
    const orth = new Array(1024).fill(0);
    orth[1] = 1;

    vectorstore.addEntries([
      { id: '1', text: 'parallel', source: 'a', embedding: base },
      { id: '2', text: 'orthogonal', source: 'b', embedding: orth },
    ]);

    // Query parallel to first entry
    (mockEmbeddings.embed as ReturnType<typeof vi.fn>).mockResolvedValueOnce([base]);
    const results = await vectorstore.similaritySearch('query', 2);
    expect(results).toHaveLength(2);
    expect(results[0].id).toBe('1');
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });

  it('removeBySource removes entries and updates cache', async () => {
    await vectorstore.load();
    vectorstore.addEntries([
      { id: '1', text: 'a', source: 'doc1', embedding: new Array(1024).fill(0.1) },
      { id: '2', text: 'b', source: 'doc2', embedding: new Array(1024).fill(0.2) },
      { id: '3', text: 'c', source: 'doc1', embedding: new Array(1024).fill(0.3) },
    ]);
    expect(vectorstore.stats.entries).toBe(3);

    const removed = vectorstore.removeBySource('doc1');
    expect(removed).toBe(2);
    expect(vectorstore.stats.entries).toBe(1);

    const cached = JSON.parse(localStorage.getItem('clay-vector-entries-v1') ?? '[]');
    expect(cached).toHaveLength(1);
    expect(cached[0].source).toBe('doc2');
  });

  it('clear removes all entries and clears cache', async () => {
    await vectorstore.load();
    vectorstore.addEntries([
      { id: '1', text: 'a', source: 'doc1', embedding: new Array(1024).fill(0.1) },
    ]);
    expect(vectorstore.stats.entries).toBe(1);

    vectorstore.clear();
    expect(vectorstore.stats.entries).toBe(0);
    expect(localStorage.getItem('clay-vector-entries-v1')).toBeNull();
  });

  it('deduplicates by id on addEntries', async () => {
    await vectorstore.load();
    const entry = { id: 'dup', text: 'x', source: 's', embedding: new Array(1024).fill(0.1) };
    vectorstore.addEntries([entry]);
    vectorstore.addEntries([entry]);
    expect(vectorstore.stats.entries).toBe(1);
  });

  it('handles localStorage corruption gracefully', async () => {
    localStorage.setItem('clay-vector-entries-v1', 'not json');
    await vectorstore.load();
    expect(vectorstore.stats.entries).toBe(0);
  });
});