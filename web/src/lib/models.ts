import { getProviderConfig, type ProviderKind } from './providers';
import type { ModelInfo, Settings, LocalModelPicks } from './types';
import {
  ProviderUnreachableError,
  InvalidApiKeyError,
  RateLimitError,
  ModelCatalogEmptyError,
  ModelNotFoundError,
  classifyError,
} from './errors';
import {
  CHAT_PATTERNS,
  CODE_PATTERNS,
  EMBEDDING_PATTERNS,
  EMBEDDING_DETECT,
  CODE_DETECT,
  SAFETY_DETECT,
  VISION_DETECT,
  CHAT_DETECT,
  SIZE_PATTERNS,
  scoreByRules,
} from './modelPatterns';

export type { ModelInfo };

export type ModelClass = 'tiny' | 'small' | 'medium' | 'large' | 'huge';

export type TaskRole = 'routing' | 'codeGen' | 'answer' | 'eval';

export interface PickedModels {
  routing: string | undefined;
  codeGen: string | undefined;
  answer: string | undefined;
  eval: string | undefined;
  embedding: string | undefined;
}

/**
 * Fetch the model catalog from any supported provider.
 * @param provider - Provider kind (openrouter, groq, together, local)
 * @param apiKey - API key for providers that require it
 * @param baseUrl - Optional custom base URL (for local)
 * @returns Array of model info objects with id, ownedBy, created
 * @throws ModelCatalogEmptyError if catalog is empty
 * @throws InvalidApiKeyError if API key is invalid (401/403)
 * @throws ProviderUnreachableError if network error or server unavailable
 * @throws RateLimitError if rate limited (429)
 */
export async function listModels(
  provider: ProviderKind,
  apiKey: string,
  baseUrl?: string,
): Promise<ModelInfo[]> {
  const config = getProviderConfig(provider);
  const url = (baseUrl || config.baseUrl).replace(/\/+$/, '');

  if (!url) {
    throw new ProviderUnreachableError(provider, undefined, { isTimeout: false });
  }

  if (config.requiresApiKey && !apiKey) {
    throw new InvalidApiKeyError(config.displayName, 401);
  }

  const headers: Record<string, string> = {};
  if (config.defaultHeaders) {
    Object.assign(headers, config.defaultHeaders);
  }
  if (apiKey && config.requiresApiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  const modelsEndpoint = config.modelsEndpoint;
  let resp: Response;
  try {
    resp = await fetch(`${url}${modelsEndpoint}`, { headers });
  } catch (e) {
    throw classifyError(e, config.displayName, 'listModels');
  }

  if (!resp.ok) {
    if (resp.status === 401 || resp.status === 403) {
      throw new InvalidApiKeyError(config.displayName, resp.status as 401 | 403);
    }
    if (resp.status === 429) {
      const retryAfter = resp.headers.get('retry-after');
      const retryAfterMs = retryAfter ? parseInt(retryAfter, 10) * 1000 : undefined;
      throw new RateLimitError(config.displayName, retryAfterMs);
    }
    throw new ProviderUnreachableError(config.displayName, new Error(`${resp.status} ${resp.statusText}`), {
      retryable: resp.status >= 500,
    });
  }

  const json = await resp.json();

  // Standard OpenAI-compatible format: { data: [{ id, object, created, owned_by }] }
  const data = json.data || [];

  if (data.length === 0) {
    throw new ModelCatalogEmptyError(config.displayName);
  }

  return data.map((m: { id: string; object?: string; created?: number; owned_by?: string }) => ({
    id: m.id,
    ownedBy: m.owned_by ?? '',
    created: m.created ?? 0,
  }));
}

// Legacy exports for backward compatibility
export async function listLocalCatalog(baseUrl: string, apiKey: string): Promise<ModelInfo[]> {
  return listModels('local', apiKey, baseUrl);
}

/**
 * Normalize user-provided local model picks into PickedModels.
 * The user-facing LocalModelPicks has only 2 slots (chat, embeddings) — the
 * single chat model fans out into all 4 chat-style roles (routing, codeGen,
 * answer, eval) that the orchestrator/analyzer/eval consume. Empty strings
 * become undefined.
 */
export function pickLocalModels(picks: LocalModelPicks): PickedModels {
  const def = (s: string) => s.trim() || undefined;
  const chat = def(picks.chat);
  const embedding = def(picks.embeddings);
  return {
    routing: chat,
    codeGen: chat,
    answer: chat,
    eval: chat,
    embedding,
  };
}

export interface ResolvedModels {
  catalog: ModelInfo[];
  picked: PickedModels;
  warnings: string[];
}

/**
 * Resolve the final model set for the current session.
 * For API providers with catalogs (NIM, OpenRouter, Groq, Together, OpenAI, Anthropic): auto-picks best models.
 * For local/Ollama: validates user picks against catalog.
 * @throws ModelNotFoundError if picked model not in catalog
 */
export function resolveModels(
  settings: Settings,
  catalog: ModelInfo[],
): ResolvedModels {
  const provider = settings.provider;

  // Local uses user-selected models
  if (provider === 'local') {
    const picked = pickLocalModels(settings.localModels);
    const warnings: string[] = [];
    if (settings.localCatalog.length === 0) {
      warnings.push(
        'Local catalog is empty. Click Discover in Settings to fetch models from the server.',
      );
    } else {
      const catalogIds = new Set(settings.localCatalog.map(m => m.id));
      const userSlots = [
        { key: 'chat', model: picked.routing },
        { key: 'embeddings', model: picked.embedding },
      ] as const;
      for (const { model } of userSlots) {
        if (model && !catalogIds.has(model)) {
          throw new ModelNotFoundError(model, Array.from(catalogIds));
        }
      }
    }
    return { catalog: settings.localCatalog, picked, warnings };
  }

  // All other providers use auto-pick from their catalog
  return { catalog, picked: pickBestModels(catalog), warnings: [] };
}

function isEmbedding(id: string): boolean {
  const lower = id.toLowerCase();
  return EMBEDDING_DETECT.some((re) => re.test(lower));
}

function isCodeSpecialist(id: string): boolean {
  const lower = id.toLowerCase();
  return CODE_DETECT.some((re) => re.test(lower));
}

function isSafetyOrGuard(id: string): boolean {
  const lower = id.toLowerCase();
  return SAFETY_DETECT.some((re) => re.test(lower));
}

function isVision(id: string): boolean {
  const lower = id.toLowerCase();
  return VISION_DETECT.some((re) => re.test(lower));
}

function isGeneralChat(id: string): boolean {
  if (isEmbedding(id)) return false;
  if (isCodeSpecialist(id)) return false;
  if (isSafetyOrGuard(id)) return false;
  if (isVision(id)) return false;
  const lower = id.toLowerCase();
  return CHAT_DETECT.some((re) => re.test(lower));
}

function inferClass(id: string): ModelClass {
  const lower = id.toLowerCase();
  for (const entry of SIZE_PATTERNS) {
    if (entry.patterns.some((re) => re.test(lower))) return entry.class;
  }
  return 'medium';
}

function scoreGeneralChat(model: ModelInfo): number {
  return scoreByRules(model.id.toLowerCase(), CHAT_PATTERNS);
}

function scoreCodeSpecialist(model: ModelInfo): number {
  return scoreByRules(model.id.toLowerCase(), CODE_PATTERNS);
}

function scoreEmbedding(model: ModelInfo): number {
  return scoreByRules(model.id.toLowerCase(), EMBEDDING_PATTERNS);
}

function pickHighest(
  models: ModelInfo[],
  score: (m: ModelInfo) => number,
  classes: ModelClass[],
): ModelInfo | undefined {
  const candidates = models.filter(m => classes.includes(inferClass(m.id)));
  if (candidates.length === 0) return undefined;
  let bestId: string | undefined;
  let bestScore = -Infinity;
  for (const m of candidates) {
    const s = score(m);
    if (s > bestScore) {
      bestScore = s;
      bestId = m.id;
    }
  }
  return candidates.find(m => m.id === bestId);
}

/**
 * Heuristically pick the best model for each task from the NIM catalog.
 * @param models - Full model catalog from NIM
 * @returns PickedModels with routing, codeGen, answer, eval, embedding
 * Returns undefined for all roles if catalog is empty.
 */
export function pickBestModels(models: ModelInfo[]): PickedModels {
  // Handle empty catalog gracefully
  if (models.length === 0) {
    return {
      routing: undefined,
      codeGen: undefined,
      answer: undefined,
      eval: undefined,
      embedding: undefined,
    };
  }

  const chats = models.filter(m => isGeneralChat(m.id));
  const codes = models.filter(m => isCodeSpecialist(m.id));
  const embeddings = models.filter(m => isEmbedding(m.id));

  const small = chats.filter(m => inferClass(m.id) === 'small');
  const tiny = chats.filter(m => inferClass(m.id) === 'tiny');
  const large = chats.filter(m => inferClass(m.id) === 'large');
  const huge = chats.filter(m => inferClass(m.id) === 'huge');

  const routing = pickHighest(small, scoreGeneralChat, ['small', 'tiny'])
    ?? pickHighest(tiny, scoreGeneralChat, ['tiny', 'small'])
    ?? small[0]
    ?? tiny[0]
    ?? chats[0];

  const evalCandidates = small.filter(m => m.id !== routing?.id);
  const evalModel = pickHighest(evalCandidates, scoreGeneralChat, ['small', 'tiny'])
    ?? pickHighest(small, scoreGeneralChat, ['small', 'tiny'])
    ?? evalCandidates[0]
    ?? small[0]
    ?? chats[0];

  const codeGen = pickHighest(codes, scoreCodeSpecialist, ['medium', 'large', 'huge', 'small'])
    ?? codes[0];

  const answer = pickHighest(huge, scoreGeneralChat, ['huge', 'large'])
    ?? pickHighest(large, scoreGeneralChat, ['large', 'huge'])
    ?? huge[0]
    ?? large[0]
    ?? chats[0];

  let embedding: ModelInfo | undefined;
  let bestEmbScore = -Infinity;
  for (const m of embeddings) {
    const s = scoreEmbedding(m);
    if (s > bestEmbScore) {
      bestEmbScore = s;
      embedding = m;
    }
  }

  // Validate all required models were found - warn but don't throw for incomplete catalogs
  const warnings: string[] = [];
  if (!routing) warnings.push('routing model not found');
  if (!evalModel) warnings.push('eval model not found');
  if (!codeGen) warnings.push('codeGen model not found');
  if (!answer) warnings.push('answer model not found');
  if (!embedding) warnings.push('embedding model not found');
  if (warnings.length > 0) {
    if (import.meta.env.DEV) console.warn('[models] Incomplete model catalog:', warnings.join(', '));
  }

  return {
    routing: routing?.id,
    codeGen: codeGen?.id,
    answer: answer?.id,
    eval: evalModel?.id,
    embedding: embedding?.id,
  };
}

/**
 * Get human-readable description of a model class.
 */
export function describeModelClass(cls: ModelClass): string {
  switch (cls) {
    case 'tiny': return 'tiny';
    case 'small': return 'small';
    case 'medium': return 'medium';
    case 'large': return 'large';
    case 'huge': return 'huge';
  }
}

/**
 * Infer the model class (size tier) from a model ID string.
 */
export function modelClass(id: string): ModelClass {
  return inferClass(id);
}