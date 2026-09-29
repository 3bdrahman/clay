/**
 * Pure statistical helpers for data analysis tools.
 * No dependencies on analysis context or tool infrastructure.
 */

export function quantile(sortedValues: number[], p: number): number {
  const n = sortedValues.length;
  if (n === 0) return NaN;
  const idx = p * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedValues[lo]!;
  return sortedValues[lo]! + (sortedValues[hi]! - sortedValues[lo]!) * (idx - lo);
}

export function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2 || n !== ys.length) return 0;
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += xs[i]!;
    sumY += ys[i]!;
  }
  const meanX = sumX / n;
  const meanY = sumY / n;
  let num = 0;
  let denX = 0;
  let denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i]! - meanX;
    const dy = ys[i]! - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  if (denX === 0 || denY === 0) return 0;
  return num / Math.sqrt(denX * denY);
}

function rankWithAverageTies(values: number[]): number[] {
  const n = values.length;
  const indexed = values.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array<number>(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n && indexed[j]!.v === indexed[i]!.v) j++;
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) {
      ranks[indexed[k]!.i] = avgRank;
    }
    i = j;
  }
  return ranks;
}

export function spearman(xs: number[], ys: number[]): number {
  if (xs.length !== ys.length) return 0;
  const rx = rankWithAverageTies(xs);
  const ry = rankWithAverageTies(ys);
  return pearson(rx, ry);
}

export function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

export function coerceNumber(v: unknown): number | undefined {
  if (isFiniteNumber(v)) return v;
  if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v);
  return undefined;
}