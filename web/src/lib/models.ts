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
  EMBEDDING_PATTERNS,
  EMBEDDING_DETECT,
  SIZE_PATTERNS,
  scoreByRules,
} from './modelPatterns';

export type { ModelInfo };

export type ModelClass = 'tiny' | 'small' | 'medium' | 'large' | 'huge';

export interface PickedModels {
  chat: string | undefined;
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

export async function listLocalCatalog(baseUrl: string, apiKey: string): Promise<ModelInfo[]> {
  return listModels('local', apiKey, baseUrl);
}

/**
 * Normalize user-provided local model picks into PickedModels.
 * The user-facing LocalModelPicks has 2 slots (chat, embeddings) — the
 * single chat model is used for all chat-style roles (routing, codeGen,
 * answer, eval) that the orchestrator/analyzer/eval consume. Empty strings
 * become undefined.
 */
export function pickLocalModels(picks: LocalModelPicks): PickedModels {
  const def = (s: string) => s.trim() || undefined;
  const chat = def(picks.chat);
  const embedding = def(picks.embeddings);
  return {
    chat,
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
 * For API providers, use the explicit chat selection; only embeddings retain catalog defaults.
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
        { key: 'chat', model: picked.chat },
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

  return {
    catalog,
    picked: {
      chat: settings.pickedModelsOverride.chatModel?.trim() || undefined,
      embedding: settings.pickedModelsOverride.embedding?.trim() || pickBestEmbedding(catalog),
    },
    warnings: [],
  };
}

function isEmbedding(id: string): boolean {
  const lower = id.toLowerCase();
  return EMBEDDING_DETECT.some((re) => re.test(lower));
}

function inferClass(id: string): ModelClass {
  const lower = id.toLowerCase();
  for (const entry of SIZE_PATTERNS) {
    if (entry.patterns.some((re) => re.test(lower))) return entry.class;
  }
  return 'medium';
}

function scoreEmbedding(model: ModelInfo): number {
  return scoreByRules(model.id.toLowerCase(), EMBEDDING_PATTERNS);
}

/**
 * Auto-pick the best embedding model from the catalog by pattern score.
 * The embedding slot is the only auto-picked model — the chat model is always
 * an explicit user choice (BYOK cost control: never silently bill a premium
 * chat model the user never selected).
 * @returns the best-scoring embedding id, or undefined when the catalog has none
 */
export function pickBestEmbedding(models: ModelInfo[]): string | undefined {
  let best: ModelInfo | undefined;
  let bestScore = -Infinity;
  for (const m of models) {
    if (!isEmbedding(m.id)) continue;
    const s = scoreEmbedding(m);
    if (s > bestScore) {
      bestScore = s;
      best = m;
    }
  }
  if (best === undefined && models.length > 0 && import.meta.env.DEV) {
    console.warn(`[models] No embedding model found among ${models.length} catalog models`);
  }
  return best?.id;
}

export function modelClass(id: string): ModelClass {
  return inferClass(id);
}