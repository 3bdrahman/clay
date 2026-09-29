// hnsw — in-repo Hierarchical Navigable Small World index over unit-norm
// vectors. Internal distance is cosine distance (1 - dot); results report
// the dot score so callers see the same values a brute-force scan produces.
//
// Design notes:
// - Seeded xorshift PRNG for level assignment: the construction needs
//   randomness, and a seeded PRNG keeps builds reproducible and tests
//   deterministic (Math.random() would make the graph irreproducible).
// - Internal sequential node ids with a label→id map: re-indexing a label
//   (the vectorstore replaces entries on re-add) marks the old node deleted
//   and allocates a fresh one, so queries route to the new vector and the
//   stale node never surfaces.
// - Deletions are filters, not graph surgery: markDeleted adds to a set and
//   search skips deleted nodes — removals never rebuild the graph.
// - The candidate and result pools are ascending-by-distance sorted arrays;
//   efConstruction bounds their size, so linear insertion is cheap and a
//   hand-rolled binary heap is unnecessary.

export interface HnswConfig {
  dimensions: number;
  /** Connections per node at layers above 0; layer 0 uses 2M. */
  M?: number;
  /** Candidate pool size during insert. */
  efConstruction?: number;
  /** PRNG seed for level assignment. */
  seed?: number;
}

export interface HnswSearchResult {
  label: string;
  score: number;
}

export interface HnswIndex {
  insert(label: string, vector: Float32Array): void;
  search(query: Float32Array, k: number, efSearch?: number): HnswSearchResult[];
  markDeleted(label: string): void;
  readonly size: number;
}

const DEFAULT_M = 8;
const DEFAULT_EF_CONSTRUCTION = 64;
// Floor for bare search() calls (no efSearch given): ef = k alone gives poor
// recall on high-dimensional data; 64 is the hnswlib-scale accuracy default.
const DEFAULT_EF_SEARCH = 64;

interface Node {
  label: string;
  vector: Float32Array;
  level: number;
  neighbors: number[][];
}

interface ScoredNode {
  id: number;
  score: number;
}

function makePrng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

function insertAscending(list: ScoredNode[], item: ScoredNode): void {
  let i = list.length;
  while (i > 0 && list[i - 1]!.score > item.score) i--;
  list.splice(i, 0, item);
}

export function createHnswIndex(config: HnswConfig): HnswIndex {
  const M = config.M ?? DEFAULT_M;
  const M0 = M * 2;
  const efConstruction = config.efConstruction ?? DEFAULT_EF_CONSTRUCTION;
  const levelFactor = 1 / Math.log(M);
  const prng = makePrng(config.seed ?? 1);

  const nodes = new Map<number, Node>();
  const labelToId = new Map<string, number>();
  const deleted = new Set<number>();
  let nextId = 0;
  let entryPointId: number | null = null;
  let maxLevel = 0;

  function randomLevel(): number {
    const u = Math.max(prng(), Number.EPSILON);
    return Math.floor(-Math.log(u) * levelFactor);
  }

  function distance(query: Float32Array, vector: Float32Array): number {
    let dot = 0;
    const len = Math.min(query.length, vector.length);
    for (let i = 0; i < len; i++) dot += query[i] * vector[i];
    return 1 - dot;
  }

  function searchLayer(query: Float32Array, entryPoints: number[], ef: number, layer: number): ScoredNode[] {
    // Deleted nodes are traversed for navigation but excluded from the
    // result pool, so the pool always fills with live candidates and the
    // search self-compensates for deletions instead of returning fewer
    // results than asked.
    const visited = new Set<number>(entryPoints);
    const candidates: ScoredNode[] = [];
    const results: ScoredNode[] = [];
    for (const id of entryPoints) {
      const node = nodes.get(id);
      if (!node) continue;
      const score = distance(query, node.vector);
      insertAscending(candidates, { id, score });
      if (!deleted.has(id)) insertAscending(results, { id, score });
    }
    while (candidates.length > 0) {
      const closest = candidates.shift()!;
      const furthest = results[results.length - 1];
      if (results.length >= ef && closest.score > furthest.score) break;
      const node = nodes.get(closest.id);
      if (!node) continue;
      const layerNeighbors = node.neighbors[layer] ?? [];
      for (const nbId of layerNeighbors) {
        if (visited.has(nbId)) continue;
        visited.add(nbId);
        const nbNode = nodes.get(nbId);
        if (!nbNode) continue;
        const score = distance(query, nbNode.vector);
        insertAscending(candidates, { id: nbId, score });
        if (deleted.has(nbId)) continue;
        const currentFurthest = results[results.length - 1];
        if (results.length < ef || score < currentFurthest.score) {
          insertAscending(results, { id: nbId, score });
          if (results.length > ef) results.pop();
        }
      }
    }
    return results;
  }

  function selectNeighbors(candidates: ScoredNode[], m: number): ScoredNode[] {
    // The full heuristic (Algorithm 4): keep a candidate only when it is
    // closer to the query than to any already-kept neighbor; backfill with
    // the pruned closest when fewer than m survive.
    const kept: ScoredNode[] = [];
    const pruned: ScoredNode[] = [];
    for (const c of candidates) {
      if (kept.length >= m) {
        pruned.push(c);
        continue;
      }
      const node = nodes.get(c.id);
      if (!node) continue;
      let closerToKept = false;
      for (const k of kept) {
        const keptNode = nodes.get(k.id);
        if (!keptNode) continue;
        if (distance(node.vector, keptNode.vector) < c.score) {
          closerToKept = true;
          break;
        }
      }
      if (closerToKept) pruned.push(c);
      else kept.push(c);
    }
    while (kept.length < m && pruned.length > 0) {
      const backfill = pruned.shift()!;
      kept.push(backfill);
    }
    return kept;
  }

  function pruneNeighbors(node: Node, layer: number, maxM: number): void {
    const dists = node.neighbors[layer].map(id => {
      const nbNode = nodes.get(id);
      return { id, score: nbNode ? distance(node.vector, nbNode.vector) : Number.POSITIVE_INFINITY };
    });
    dists.sort((a, b) => a.score - b.score);
    node.neighbors[layer] = dists.slice(0, maxM).map(d => d.id);
  }

  function insert(label: string, vector: Float32Array): void {
    const existingId = labelToId.get(label);
    if (existingId !== undefined) {
      deleted.add(existingId);
    }
    const id = nextId++;
    const level = randomLevel();
    nodes.set(id, { label, vector, level, neighbors: Array.from({ length: level + 1 }, () => [] as number[]) });
    labelToId.set(label, id);

    if (entryPointId === null) {
      entryPointId = id;
      maxLevel = level;
      return;
    }

    let ep = [entryPointId];
    for (let lc = maxLevel; lc > level; lc--) {
      const found = searchLayer(vector, ep, 1, lc);
      if (found.length === 0) break;
      ep = [found[0]!.id];
    }

    for (let lc = Math.min(level, maxLevel); lc >= 0; lc--) {
      const candidates = searchLayer(vector, ep, efConstruction, lc);
      if (candidates.length === 0) continue;
      const selected = selectNeighbors(candidates, M);
      const node = nodes.get(id)!;
      for (const s of selected) {
        node.neighbors[lc].push(s.id);
        const nbNode = nodes.get(s.id)!;
        nbNode.neighbors[lc].push(id);
        const maxM = lc === 0 ? M0 : M;
        if (nbNode.neighbors[lc].length > maxM) {
          pruneNeighbors(nbNode, lc, maxM);
        }
      }
      ep = candidates.map(c => c.id);
    }

    if (level > maxLevel) {
      maxLevel = level;
      entryPointId = id;
    }
  }

  function search(query: Float32Array, k: number, efSearch?: number): HnswSearchResult[] {
    if (entryPointId === null || nodes.size === 0) return [];
    const ef = Math.max(k, efSearch ?? DEFAULT_EF_SEARCH);
    let ep = [entryPointId];
    for (let lc = maxLevel; lc >= 1; lc--) {
      const found = searchLayer(query, ep, 1, lc);
      if (found.length === 0) break;
      ep = [found[0]!.id];
    }
    const results = searchLayer(query, ep, ef, 0);
    const out: HnswSearchResult[] = [];
    for (const r of results) {
      if (out.length >= k) break;
      const node = nodes.get(r.id)!;
      out.push({ label: node.label, score: 1 - r.score });
    }
    return out;
  }

  function markDeleted(label: string): void {
    const id = labelToId.get(label);
    if (id !== undefined) deleted.add(id);
  }

  return {
    insert,
    search,
    markDeleted,
    get size() {
      return nodes.size - deleted.size;
    },
  };
}
