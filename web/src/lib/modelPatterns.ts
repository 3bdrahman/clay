/**
 * Model size-tier patterns used by the model class display.
 * Patterns are loaded from `modelPatterns.config.json` at build time (via Vite
 * import) with hardcoded defaults as fallback. This allows updating the size
 * tiers without code changes by replacing the JSON config file.
 */

import modelPatternsConfig from './modelPatterns.config.json?raw';

type SizeTier = 'huge' | 'large' | 'medium' | 'small' | 'tiny';

export interface ModelPatternsConfig {
  version: number;
  sizePatterns: Array<{ class: SizeTier; patterns: string[] }>;
}

const DEFAULT_SIZE_PATTERNS: ReadonlyArray<{ class: SizeTier; patterns: RegExp[] }> = [
  { class: 'huge',   patterns: [/ultra|550b|340b|253b|122b/] },
  { class: 'large',  patterns: [/120b|90b|72b|70b|^.*large/] },
  { class: 'medium', patterns: [/49b|51b|34b|30b|22b|15b|14b|13b|12b|11b/] },
  { class: 'small',  patterns: [/8b|7b|nano/] },
  { class: 'tiny',   patterns: [/mini|4b|3b|2b|1b/] },
] as const;

function parseConfig(json: string): ModelPatternsConfig | null {
  try {
    const parsed = JSON.parse(json) as Partial<ModelPatternsConfig> | null;
    if (!parsed?.version || !Array.isArray(parsed.sizePatterns)) return null;
    return parsed as ModelPatternsConfig;
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[modelPatterns] failed to parse config; using default size tiers', e);
    }
    return null;
  }
}

function buildSizePatterns(config: ModelPatternsConfig | null): ReadonlyArray<{ class: SizeTier; patterns: RegExp[] }> {
  if (!config) return DEFAULT_SIZE_PATTERNS;
  return config.sizePatterns.map(s => ({ class: s.class, patterns: s.patterns.map(p => new RegExp(p)) }));
}

const config = parseConfig(modelPatternsConfig);

export const SIZE_PATTERNS: ReadonlyArray<{ class: SizeTier; patterns: RegExp[] }> = buildSizePatterns(config);
