// Provider registry - all providers work client-side without proxy (except Ollama CORS)
// OpenRouter is default - works everywhere with generous free tier

import type { ProviderKind } from './types';
import type { Settings } from './types';

export type { ProviderKind };

/** Settings field holding the API key for a cloud provider. Undefined for 'local'. */
export type ProviderApiKeyField = 'openrouterApiKey' | 'groqApiKey' | 'togetherApiKey';

export interface ProviderConfig {
  kind: ProviderKind;
  displayName: string;
  baseUrl: string;
  modelsEndpoint: string;
  apiKeyHint: string;
  freeTier: boolean;
  requiresApiKey: boolean;
  defaultHeaders?: Record<string, string>;
  apiKeyUrl: string;
}

// Attribution referer sent to OpenRouter. Can be overridden at build time with
// VITE_OPENROUTER_REFERER; defaults to the current origin at runtime, and to the
// GitHub Pages deployment when no window exists (SSR/build).
const OPENROUTER_REFERER_FALLBACK = 'https://3bdrahman.github.io/clay/';

function getOpenRouterReferer(): string {
  const override = import.meta.env.VITE_OPENROUTER_REFERER?.trim();
  if (override) return override;
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }
  return OPENROUTER_REFERER_FALLBACK;
}

export const LOCAL_DEFAULT_BASE_URL = 'http://localhost:11434/v1';

export const LOCAL_PROVIDER_HINT =
  'Any OpenAI-compatible endpoint — LM Studio, vLLM, llama.cpp server, Jan, GPT4All.';

export const OLLAMA_CORS_HINT =
  'If using Ollama, browser CORS blocks requests unless OLLAMA_ORIGINS is set. ' +
  'Run: OLLAMA_ORIGINS="*" ollama serve  (or add your deployed origin to OLLAMA_ORIGINS).';

export function isOllamaUrl(url: string): boolean {
  try {
    const u = new URL(url.trim());
    return u.port === '11434' || u.hostname.endsWith('.ollama') || u.pathname.includes('ollama');
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[providers] isOllamaUrl: malformed URL (returning false):', e);
    }
    return false;
  }
}

export const PROVIDER_REGISTRY: Record<ProviderKind, ProviderConfig> = {
  openrouter: {
    kind: 'openrouter',
    displayName: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    modelsEndpoint: '/models',
    apiKeyHint: 'sk-or-v1-...',
    freeTier: true,
    requiresApiKey: true,
    defaultHeaders: {
      'X-Title': 'Clay RAG',
    },
    apiKeyUrl: 'https://openrouter.ai/keys',
  },
  groq: {
    kind: 'groq',
    displayName: 'Groq',
    baseUrl: 'https://api.groq.com/openai/v1',
    modelsEndpoint: '/models',
    apiKeyHint: 'gsk_...',
    freeTier: true,
    requiresApiKey: true,
    apiKeyUrl: 'https://console.groq.com/keys',
  },
  together: {
    kind: 'together',
    displayName: 'Together AI',
    baseUrl: 'https://api.together.xyz/v1',
    modelsEndpoint: '/models',
    apiKeyHint: '...',
    freeTier: true,
    requiresApiKey: true,
    apiKeyUrl: 'https://api.together.xyz/settings/api-keys',
  },
  local: {
    kind: 'local',
    displayName: 'Local (OpenAI-compatible)',
    baseUrl: LOCAL_DEFAULT_BASE_URL,
    modelsEndpoint: '/models',
    apiKeyHint: 'optional',
    freeTier: true,
    requiresApiKey: false,
    apiKeyUrl: '',
  },
} as const;

export function getProviderConfig(kind: ProviderKind): ProviderConfig {
  return PROVIDER_REGISTRY[kind];
}

export function getProviderApiKeyField(kind: ProviderKind): ProviderApiKeyField | undefined {
  switch (kind) {
    case 'openrouter': return 'openrouterApiKey';
    case 'groq': return 'groqApiKey';
    case 'together': return 'togetherApiKey';
    case 'local': return undefined;
  }
}

export interface ProviderEndpoint {
  baseUrl: string;
  apiKey: string;
  providerLabel: string;
  defaultHeaders?: Record<string, string>;
}

export function resolveProviderEndpoint(settings: Settings): ProviderEndpoint {
  const config = PROVIDER_REGISTRY[settings.provider];
  const apiKeyField = getProviderApiKeyField(settings.provider);

  if (settings.provider === 'local') {
    return {
      baseUrl: settings.localServerUrl.trim(),
      apiKey: '',
      providerLabel: config.displayName,
    };
  }

  const apiKey = (apiKeyField !== undefined ? settings[apiKeyField] : '') || settings.apiKey || '';

  const defaultHeaders = { ...config.defaultHeaders };
  if (settings.provider === 'openrouter') {
    defaultHeaders['HTTP-Referer'] = getOpenRouterReferer();
  }

  return {
    baseUrl: config.baseUrl,
    apiKey,
    providerLabel: config.displayName,
    defaultHeaders,
  };
}