import { getProviderConfig, type ProviderKind } from './providers';
import type { ModelInfo, Settings, LocalModelPicks } from './types';
import { getCloudModelOptions } from './modelPolicy';
import {
  ProviderUnreachableError,
  ProviderTimeoutError,
  RagError,
  InvalidApiKeyError,
  RateLimitError,
  ModelCatalogEmptyError,
  ModelNotFoundError,
  classifyError,
} from './errors';
import { SIZE_PATTERNS } from './modelPatterns';
import { getLocalRequestOptions, localConnectionError, normalizeLocalServerUrl } from './localEndpoint';

const MODEL_DISCOVERY_TIMEOUT_MS = 15_000;

function invalidCatalog(providerLabel: string): ProviderUnreachableError {
  return new ProviderUnreachableError(providerLabel, undefined, {
    message: `${providerLabel} returned an invalid model catalog. Use the server’s OpenAI-compatible API base URL (usually ending in /v1).`,
    retryable: false,
  });
}

async function fetchCatalog(provider: ProviderKind, baseUrl: string, headers: Record<string, string>, signal?: AbortSignal): Promise<unknown> {
  const config = getProviderConfig(provider);
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  signal?.addEventListener('abort', onAbort, { once: true });
  if (signal?.aborted) controller.abort();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, MODEL_DISCOVERY_TIMEOUT_MS);
  try {
    if (signal?.aborted) throw new DOMException('Model discovery cancelled', 'AbortError');
    const resp = await fetch(`${baseUrl}${config.modelsEndpoint}`, {
      ...(provider === 'local' ? getLocalRequestOptions(baseUrl) : {}),
      headers,
      signal: controller.signal,
    });
    if (!resp.ok) {
      if (resp.status === 401 || resp.status === 403) throw new InvalidApiKeyError(config.displayName, resp.status as 401 | 403);
      if (resp.status === 429) {
        const retryAfter = resp.headers.get('retry-after');
        throw new RateLimitError(config.displayName, retryAfter ? parseInt(retryAfter, 10) * 1000 : undefined);
      }
      throw new ProviderUnreachableError(config.displayName, undefined, {
        message: `Model discovery returned HTTP ${resp.status}. Check the OpenAI-compatible API base URL and that the server exposes /models.`,
        retryable: resp.status >= 500,
      });
    }
    return await resp.json();
  } catch (error) {
    if (timedOut) throw new ProviderTimeoutError(config.displayName, MODEL_DISCOVERY_TIMEOUT_MS);
    if (signal?.aborted) throw new ProviderUnreachableError(config.displayName, undefined, { message: 'Model discovery was cancelled.', retryable: false });
    if (error instanceof RagError) throw error;
    if (typeof error === 'object' && error !== null && 'name' in error && error.name === 'SyntaxError') throw invalidCatalog(config.displayName);
    if (provider === 'local') throw localConnectionError(baseUrl, error);
    throw classifyError(error, config.displayName, 'listModels');
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

export type { ModelInfo };

export type ModelClass = 'tiny' | 'small' | 'medium' | 'large' | 'huge';

export interface PickedModels {
  chat: string | undefined;
}

/**
 * Fetch the model catalog from any supported provider.
 * @param provider - Provider kind (openrouter, local)
 * @param apiKey - API key for providers that require it
 * @param baseUrl - Resolved OpenRouter endpoint or local server
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
  signal?: AbortSignal,
): Promise<ModelInfo[]> {
  const config = getProviderConfig(provider);
  const url = provider === 'local'
    ? normalizeLocalServerUrl(baseUrl ?? config.baseUrl)
    : (baseUrl || config.baseUrl).replace(/\/+$/, '');

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

  const json = await fetchCatalog(provider, url, headers, signal);
  if (!json || typeof json !== 'object' || !('data' in json) || !Array.isArray(json.data)) {
    throw invalidCatalog(config.displayName);
  }
  const data = json.data;
  if (data.some(model => !model || typeof model !== 'object' || typeof model.id !== 'string' || !model.id.trim())) {
    throw invalidCatalog(config.displayName);
  }

  if (data.length === 0) {
    throw new ModelCatalogEmptyError(config.displayName);
  }

  return data.map((m: {
    id: string;
    object?: string;
    created?: number;
    owned_by?: string;
    pricing?: Record<string, string | number | null>;
    supported_parameters?: string[];
  }) => {
    const pricing = m.pricing
      ? Object.fromEntries(
        Object.entries(m.pricing)
          .filter((entry): entry is [string, string | number] => entry[1] !== null)
          .map(([key, value]) => [key, String(value)]),
      )
      : undefined;
    return {
      id: m.id,
      ownedBy: m.owned_by ?? '',
      created: m.created ?? 0,
      ...(pricing ? { pricing } : {}),
      ...(m.supported_parameters ? { supportedParameters: m.supported_parameters } : {}),
    };
  });
}

export async function listLocalCatalog(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<ModelInfo[]> {
  return listModels('local', apiKey, normalizeLocalServerUrl(baseUrl), signal);
}

/**
 * Normalize the user-provided local chat pick into PickedModels. The single
 * chat model drives every chat-style role (routing, codeGen, answer, eval);
 * embeddings are local (transformers.js) and never user-selected. Empty
 * strings become undefined.
 */
export function pickLocalModels(picks: LocalModelPicks): PickedModels {
  return { chat: picks.chat.trim() || undefined };
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

  // Local uses the user-selected chat model
  if (provider === 'local') {
    const picked = pickLocalModels(settings.localModels);
    const warnings: string[] = [];
    if (settings.localCatalog.length === 0) {
      warnings.push(
        'Local catalog is empty. Click Discover in Settings to fetch models from the server.',
      );
    } else {
      const catalogIds = new Set(settings.localCatalog.map(m => m.id));
      if (picked.chat && !catalogIds.has(picked.chat)) {
        throw new ModelNotFoundError(picked.chat, Array.from(catalogIds));
      }
    }
    return { catalog: settings.localCatalog, picked, warnings };
  }

  const cloudCatalog = getCloudModelOptions(provider, catalog);
  const selected = settings.pickedModelsOverride.chatModel?.trim() || undefined;
  const warnings: string[] = [];

  if (selected) {
    const selectedModel = cloudCatalog.find((model) => model.id === selected);
    if (!selectedModel) {
      warnings.push(
        `${selected} is not available for ${provider}. Choose one of the approved free models in Settings.`,
      );
      return { catalog: cloudCatalog, picked: { chat: undefined }, warnings };
    }
    return { catalog: cloudCatalog, picked: { chat: selectedModel.id }, warnings };
  }

  const recommended = cloudCatalog[0]?.id;
  if (!recommended) {
    warnings.push(`No approved ${provider} models are available in the fetched catalog.`);
  }
  return {
    catalog: cloudCatalog,
    picked: { chat: recommended },
    warnings,
  };
}

export function modelClass(id: string): ModelClass {
  return inferClass(id);
}

function inferClass(id: string): ModelClass {
  const lower = id.toLowerCase();
  for (const entry of SIZE_PATTERNS) {
    if (entry.patterns.some((re) => re.test(lower))) return entry.class;
  }
  return 'medium';
}
