/**
 * Model classification patterns used by the dynamic model picker.
 *
 * Each pattern set is a list of `{ pattern, points }` entries that contribute
 * to a model's score when picked for a task. Extracted from the legacy
 * monolithic `models.ts` so the heuristics are:
 *
 *   1. Inspectable — easy to see at a glance which model families win.
 *   2. Editable   — adding a new preferred model family is a one-line change.
 *   3. Testable   — each rule is a pure function of the model ID.
 *
 * The pattern matcher is intentionally regex-based so we can match model
 * identifiers the catalog hasn't seen yet (e.g. new NIM releases) by family
 * suffix or size class.
 */

export interface ScoreRule {
  /** Human-readable description, surfaced in test failure messages. */
  readonly family: string;
  /** Regex matched against the lowercased model id. */
  readonly pattern: RegExp;
  /** Score contribution when the pattern matches. */
  readonly points: number;
}

export const CHAT_PATTERNS: readonly ScoreRule[] = [
  { family: 'meta-llama-3.3',          pattern: /^meta\/llama-3\.3-/,                     points: 30 },
  { family: 'meta-llama-3.1-70b-8b',   pattern: /^meta\/llama-3\.1-(70b|8b)/,              points: 25 },
  { family: 'mistral-large-2',         pattern: /^mistralai\/mistral-large-2/,             points: 28 },
  { family: 'mistral-7b',              pattern: /^mistralai\/mistral-7b/,                  points: 22 },
  { family: 'nemotron-3-super-ultra',  pattern: /^nvidia\/nemotron-3-(super|ultra)/,       points: 40 },
  { family: 'nemotron-4',              pattern: /^nvidia\/nemotron-4-/,                    points: 35 },
  { family: 'llama-3.1-nemotron-70b',  pattern: /^nvidia\/llama-3\.1-nemotron-(70b|ultra|super)/, points: 30 },
  { family: 'llama-3.1-nemotron-nano', pattern: /^nvidia\/llama-3\.1-nemotron-nano/,       points: 22 },
  { family: 'gpt-oss',                 pattern: /^openai\/gpt-oss-/,                       points: 30 },
  { family: 'palmyra',                 pattern: /^writer\/palmyra/,                        points: 20 },
  { family: 'stepfun',                 pattern: /^stepfun-ai\/step-/,                      points: 18 },
  { family: 'kimi',                    pattern: /^moonshotai\/kimi-/,                      points: 25 },
  { family: 'glm',                     pattern: /^z-ai\/glm-/,                             points: 22 },
  { family: 'deepseek-v',              pattern: /^deepseek-ai\/deepseek-v/,                points: 28 },
  { family: 'gemma-3',                 pattern: /^google\/gemma-3-(12b|4b)/,               points: 18 },
  { family: 'gemma-4',                 pattern: /^google\/gemma-4-/,                       points: 22 },
  { family: 'granite-3.0',             pattern: /^ibm\/granite-3\.0-/,                     points: 15 },
  { family: 'laguna',                  pattern: /^poolside\/laguna/,                       points: 18 },
  { family: 'zamba',                   pattern: /^zyphra\/zamba/,                          points: 12 },
] as const;

export const CODE_PATTERNS: readonly ScoreRule[] = [
  { family: 'codestral-22b',           pattern: /codestral-22b/,                           points: 50 },
  { family: 'codestral',               pattern: /codestral/,                               points: 45 },
  { family: 'codellama-70b',           pattern: /codellama-70b/,                           points: 35 },
  { family: 'codellama',               pattern: /codellama/,                               points: 30 },
  { family: 'codegemma',               pattern: /codegemma/,                               points: 25 },
  { family: 'deepseek-coder',          pattern: /deepseek-coder/,                          points: 28 },
  { family: 'granite-code',            pattern: /granite.*code/,                           points: 25 },
  { family: 'starcoder2',              pattern: /starcoder2/,                              points: 20 },
  { family: 'nemotron-code',           pattern: /nemotron.*code/,                          points: 22 },
  { family: 'size-8b',                 pattern: /8b/,                                      points: 3  },
  { family: 'size-15b',                pattern: /15b/,                                     points: 5  },
  { family: 'size-22b',                pattern: /22b/,                                     points: 8  },
  { family: 'size-34b',                pattern: /34b/,                                     points: 10 },
  { family: 'size-70b',                pattern: /70b/,                                     points: 12 },
] as const;

export const EMBEDDING_PATTERNS: readonly ScoreRule[] = [
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

export const EMBEDDING_DETECT: readonly RegExp[] = [/embed|embedqa/i];
export const CODE_DETECT: readonly RegExp[] = [
  /codestral|codellama|codegemma|granite.*code|deepseek-coder|nemotron.*code|starcoder|embedcode/,
];
export const SAFETY_DETECT: readonly RegExp[] = [
  /guard|safety|content-safety|topic-control|reward|parse|translate|detector|calibration|neva|vila|ai-synthetic|cosmo/,
];
export const VISION_DETECT: readonly RegExp[] = [
  /vision|vl$|clip|video|diffusion|deplot|recurrent|cosmos/,
];
export const CHAT_DETECT: readonly RegExp[] = [
  /instruct|chat|^.*\/gpt-|it$|nemotron|moe|reasoning|creative|magistral|laguna|kimi|step-|glm|inkling|palmyra|sea-lion|yi-|zamba|granite|gemma/,
];

export const SIZE_PATTERNS: ReadonlyArray<{ class: 'huge' | 'large' | 'medium' | 'small' | 'tiny'; patterns: RegExp[] }> = [
  { class: 'huge',   patterns: [/ultra|550b|340b|253b|122b/] },
  { class: 'large',  patterns: [/120b|90b|72b|70b|^.*large/] },
  { class: 'medium', patterns: [/49b|51b|34b|30b|22b|15b|14b|13b|12b|11b/] },
  { class: 'small',  patterns: [/8b|7b|nano/] },
  { class: 'tiny',   patterns: [/mini|4b|3b|2b|1b/] },
] as const;

/** Apply every rule in `rules` to a lowercased model id and sum the points. */
export function scoreByRules(idLower: string, rules: readonly ScoreRule[]): number {
  let score = 0;
  for (const rule of rules) {
    if (rule.pattern.test(idLower)) score += rule.points;
  }
  return score;
}
