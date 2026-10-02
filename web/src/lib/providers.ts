// Provider configuration and browser endpoint resolution.

import type { ProviderKind } from './types';
import type { Settings } from './types';

export type { ProviderKind };

/** Settings field holding the API key for a cloud provider. Undefined for 'local'. */
export type ProviderApiKeyField = 'openrouterApiKey' | 'nimApiKey';

export interface ProviderConfig {
  kind: ProviderKind;
  displayName: string;
  baseUrl: string;
  modelsEndpoint: string;
  apiKeyHint: string;
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
export const NIM_API_BASE_URL = 'https://integrate.api.nvidia.com/v1';
export const NIM_DEV_BASE_URL = '/nim-api/v1';

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
    requiresApiKey: true,
    defaultHeaders: {
      'X-Title': 'Clay RAG',
    },
    apiKeyUrl: 'https://openrouter.ai/keys',
  },
  nim: {
    kind: 'nim',
    displayName: 'NVIDIA NIM',
    baseUrl: NIM_API_BASE_URL,
    modelsEndpoint: '/models',
    apiKeyHint: 'nvapi-...',
    requiresApiKey: true,
    apiKeyUrl: 'https://build.nvidia.com/settings/api-keys',
  },
  local: {
    kind: 'local',
    displayName: 'Local (OpenAI-compatible)',
    baseUrl: LOCAL_DEFAULT_BASE_URL,
    modelsEndpoint: '/models',
    apiKeyHint: 'optional',
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
    case 'nim': return 'nimApiKey';
    case 'local': return undefined;
  }
}

export interface ProviderEndpoint {
  baseUrl: string;
  apiKey: string;
  providerLabel: string;
  defaultHeaders?: Record<string, string>;
  /** An endpoint can be constructed for local data services while chat setup is incomplete. */
  configurationError?: string;
}

function resolveNimBaseUrl(value: string): { baseUrl: string; configurationError?: string } {
  const unavailable = (configurationError: string) => ({ baseUrl: NIM_API_BASE_URL, configurationError });
  if (!value) {
    return unavailable('NVIDIA NIM needs a browser-accessible relay. Enter your relay URL ending in /v1 below.');
  }
  const relative = value.startsWith('/') && !value.startsWith('//');
  let url: URL;
  try {
    url = new URL(value, relative ? 'http://localhost' : undefined);
  } catch {
    return unavailable('Enter a valid NIM relay URL ending in /v1.');
  }
  const isLoopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && isLoopback)) {
    return unavailable('Use HTTPS for the NIM relay, or HTTP for a server on localhost.');
  }
  if (url.username || url.password || url.search || url.hash) {
    return unavailable('The NIM relay URL must not contain credentials, query parameters, or a fragment. Enter the key separately.');
  }
  if (url.hostname === new URL(NIM_API_BASE_URL).hostname) {
    return unavailable('NVIDIA blocks direct browser requests. Use a relay you control instead of the NVIDIA API URL.');
  }
  const path = url.pathname.replace(/\/+$/, '');
  if (!path.endsWith('/v1')) {
    return unavailable('The NIM relay base URL must include /v1, for example https://your-relay.workers.dev/v1.');
  }
  return { baseUrl: `${relative ? '' : url.origin}${path}` };
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

  const apiKey = apiKeyField !== undefined ? settings[apiKeyField].trim() : '';

  if (settings.provider === 'nim') {
    const configuredBase = settings.nimBaseUrl?.trim()
      || import.meta.env.VITE_NIM_BASE_URL?.trim()
      || (import.meta.env.DEV ? NIM_DEV_BASE_URL : '');
    return { ...resolveNimBaseUrl(configuredBase), apiKey, providerLabel: config.displayName };
  }

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
