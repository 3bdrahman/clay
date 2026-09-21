/**
 * Model classification patterns used by the embedding auto-picker and the
 * model size-tier display.
 *
 * Patterns are loaded from `modelPatterns.config.json` at build time (via Vite import)
 * with hardcoded defaults as fallback. This allows updating model preferences without
 * code changes by replacing the JSON config file.
 *
 * Each pattern set is a list of `{ pattern, points }` entries that contribute
 * to a model's score. The pattern matcher is intentionally regex-based so we
 * can match model identifiers the catalog hasn't seen yet by family suffix.
 */

import modelPatternsConfig from './modelPatterns.config.json?raw';

export interface ScoreRule {
  /** Human-readable description, surfaced in test failure messages. */
  readonly family: string;
  /** Regex matched against the lowercased model id. */
  readonly pattern: RegExp;
  /** Score contribution when the pattern matches. */
  readonly points: number;
}

export interface ModelPatternsConfig {
  version: number;
  embeddingPatterns: Array<{ family: string; pattern: string; points: number }>;
  embeddingDetect: string[];
  sizePatterns: Array<{ class: 'huge' | 'large' | 'medium' | 'small' | 'tiny'; patterns: string[] }>;
}

// Hardcoded defaults matching the config file - used as fallback if config fails to load
const DEFAULT_EMBEDDING_PATTERNS: readonly ScoreRule[] = [
  { family: 'nv-embedqa-e5',           pattern: /nv-embedqa-e5/,                           points: 50 },
  { family: 'nv-embedqa-mistral',      pattern: /nv-embedqa-mistral/,                      points: 35 },
  { family: 'embedqa',                 pattern: /embedqa/,                                 points: 30 },
  { family: 'nv-embedcode',            pattern: /nv-embedcode/,                            points: 25 },
  { family: 'nv-embed-v1',             pattern: /nv-embed-v1/,                             points: 20 },
  { family: 'llama-nemotron-embed',    pattern: /llama-nemotron-embed/,                    points: 25 },
  { family: 'nemotron-3-embed',        pattern: /nemotron-3-embed/,                        points: 22 },
  { family: 'nemoretriever',           pattern: /nemoretriever/,                           points: 18 },
  { family: 'arctic-embed',            pattern: /arctic-embed/,                            points: 15 },
  { family: 'bge-m3',                  pattern: /bge-m3/,                                  points: 12 },
  { family: 'embed-qa-4',              pattern: /embed-qa-4/,                              points: 5  },
] as const;

const DEFAULT_EMBEDDING_DETECT: readonly RegExp[] = [/embed|embedqa/i];

const DEFAULT_SIZE_PATTERNS: ReadonlyArray<{ class: 'huge' | 'large' | 'medium' | 'small' | 'tiny'; patterns: RegExp[] }> = [
  { class: 'huge',   patterns: [/ultra|550b|340b|253b|122b/] },
  { class: 'large',  patterns: [/120b|90b|72b|70b|^.*large/] },
  { class: 'medium', patterns: [/49b|51b|34b|30b|22b|15b|14b|13b|12b|11b/] },
  { class: 'small',  patterns: [/8b|7b|nano/] },
  { class: 'tiny',   patterns: [/mini|4b|3b|2b|1b/] },
] as const;

function parseConfig(json: string): ModelPatternsConfig | null {
  try {
    const parsed = JSON.parse(json);
    // Validate every field buildPatterns consumes - a config missing any of
    // them must fall back to the hardcoded defaults above.
    if (
      !parsed.version ||
      !Array.isArray(parsed.embeddingPatterns) ||
      !Array.isArray(parsed.embeddingDetect) ||
      !Array.isArray(parsed.sizePatterns)
    ) {
      return null;
    }
    return parsed as ModelPatternsConfig;
  } catch {
    return null;
  }
}

function buildPatterns(config: ModelPatternsConfig | null) {
  const useDefaults = !config;

  const embeddingRules: readonly ScoreRule[] = useDefaults
    ? DEFAULT_EMBEDDING_PATTERNS
    : config.embeddingPatterns.map(r => ({ family: r.family, pattern: new RegExp(r.pattern), points: r.points }));

  const embeddingDetect: readonly RegExp[] = useDefaults
    ? DEFAULT_EMBEDDING_DETECT
    : config.embeddingDetect.map(p => new RegExp(p, 'i'));

  const sizePatterns: ReadonlyArray<{ class: 'huge' | 'large' | 'medium' | 'small' | 'tiny'; patterns: RegExp[] }> = useDefaults
    ? DEFAULT_SIZE_PATTERNS
    : config.sizePatterns.map(s => ({ class: s.class, patterns: s.patterns.map(p => new RegExp(p)) }));

  return {
    embeddingRules,
    embeddingDetect,
    sizePatterns,
  };
}

// Load and parse config at module initialization
const config = parseConfig(modelPatternsConfig);
const patterns = buildPatterns(config);

export const EMBEDDING_PATTERNS: readonly ScoreRule[] = patterns.embeddingRules;
export const EMBEDDING_DETECT: readonly RegExp[] = patterns.embeddingDetect;
export const SIZE_PATTERNS: ReadonlyArray<{ class: 'huge' | 'large' | 'medium' | 'small' | 'tiny'; patterns: RegExp[] }> = patterns.sizePatterns;

/** Apply every rule in `rules` to a lowercased model id and sum the points. */
export function scoreByRules(idLower: string, rules: readonly ScoreRule[]): number {
  let score = 0;
  for (const rule of rules) {
    if (rule.pattern.test(idLower)) score += rule.points;
  }
  return score;
}
