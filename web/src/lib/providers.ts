// Provider registry with all supported providers
// Each provider config includes base URL, API key hints, free tier info, and model discovery endpoint

import type { ProviderKind } from './types';

export type { ProviderKind };

export interface ProviderConfig {
  kind: ProviderKind;
  displayName: string;
  baseUrl: string;
  modelsEndpoint: string;
  apiKeyEnvVar: string;
  apiKeyHint: string;
  freeTier: boolean;
  requiresApiKey: boolean;
  defaultHeaders?: Record<string, string>;
  apiKeyUrl: string;
  description: string;
}

// NIM only serves CORS headers for build.nvidia.com. To make the 100%
// client-side app work in dev, vite.config.ts proxies /nim-api → NIM.
// On Netlify/Cloudflare, a Worker proxy handles CORS.
// In production on other hosts, set VITE_NIM_BASE_URL to your edge proxy.
const NIM_FALLBACK_BASE_URL = 'https://integrate.api.nvidia.com/v1';

function resolveCloudflareProxy(): string {
  const envUrl = (import.meta.env.VITE_CLOUDFLARE_PROXY_URL as string | undefined)?.trim();
  if (envUrl) return envUrl;
  return 'https://clay-nim-proxy.mixed-account.workers.dev';
}

function isBuildNvidia(): boolean {
  try {
    if (typeof window === 'undefined') return false;
    return window.location.hostname === 'build.nvidia.com';
  } catch {
    return false;
  }
}

function isNetlifyOrCloudflare(): boolean {
  try {
    if (typeof window === 'undefined') return false;
    const h = window.location.hostname;
    return h.endsWith('.netlify.app') ||
           h.endsWith('.cloudflare.workers.dev') ||
           h === 'clay-rag.netlify.app' ||
           h === '3bdrahman.github.io';
  } catch {
    return false;
  }
}

function resolveNimBaseUrl(): string {
  const envUrl = (import.meta.env.VITE_NIM_BASE_URL as string | undefined)?.trim();
  if (envUrl) return envUrl;

  const isLocal = typeof window !== 'undefined' &&
                  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

  if (import.meta.env.DEV || isLocal) return '/nim-api';
  if (isBuildNvidia()) return NIM_FALLBACK_BASE_URL;
  if (isNetlifyOrCloudflare()) return resolveCloudflareProxy();
  return NIM_FALLBACK_BASE_URL;
}

export const NIM_BASE_URL: string = resolveNimBaseUrl();

export const LOCAL_DEFAULT_BASE_URL = 'http://localhost:11434/v1';

export const LOCAL_PROVIDER_HINT =
  'Any OpenAI-compatible endpoint — Ollama, LM Studio, vLLM, llama.cpp server.';

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
  nim: {
    kind: 'nim',
    displayName: 'NVIDIA NIM',
    baseUrl: NIM_BASE_URL,
    modelsEndpoint: '/models',
    apiKeyEnvVar: 'VITE_NIM_API_KEY',
    apiKeyHint: 'nvapi-...',
    freeTier: true,
    requiresApiKey: true,
    apiKeyUrl: 'https://build.nvidia.com/settings/api-keys',
    description: 'NVIDIA hosted models with free tier. Auto-picks best model per task from live catalog.',
  },
  openrouter: {
    kind: 'openrouter',
    displayName: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    modelsEndpoint: '/models',
    apiKeyEnvVar: 'VITE_OPENROUTER_API_KEY',
    apiKeyHint: 'sk-or-v1-...',
    freeTier: true,
    requiresApiKey: true,
    defaultHeaders: {
      'HTTP-Referer': 'https://clay-rag.netlify.app',
      'X-Title': 'Clay RAG',
    },
    apiKeyUrl: 'https://openrouter.ai/keys',
    description: 'Access 300+ models from all providers. Free tier includes many models. Auto-selects best free model.',
  },
  groq: {
    kind: 'groq',
    displayName: 'Groq',
    baseUrl: 'https://api.groq.com/openai/v1',
    modelsEndpoint: '/models',
    apiKeyEnvVar: 'VITE_GROQ_API_KEY',
    apiKeyHint: 'gsk_...',
    freeTier: true,
    requiresApiKey: true,
    apiKeyUrl: 'https://console.groq.com/keys',
    description: 'Ultra-fast inference on Llama, Mixtral, Gemma. Generous free tier. Best for speed.',
  },
  together: {
    kind: 'together',
    displayName: 'Together AI',
    baseUrl: 'https://api.together.xyz/v1',
    modelsEndpoint: '/models',
    apiKeyEnvVar: 'VITE_TOGETHER_API_KEY',
    apiKeyHint: '...',
    freeTier: true,
    requiresApiKey: true,
    apiKeyUrl: 'https://api.together.xyz/settings/api-keys',
    description: 'Open models with fast inference. Free credits on signup. Good model variety.',
  },
  local: {
    kind: 'local',
    displayName: 'Local (OpenAI-compatible)',
    baseUrl: LOCAL_DEFAULT_BASE_URL,
    modelsEndpoint: '/models',
    apiKeyEnvVar: '',
    apiKeyHint: 'optional',
    freeTier: true,
    requiresApiKey: false,
    apiKeyUrl: '',
    description: 'Any OpenAI-compatible server (LM Studio, vLLM, llama.cpp, Jan, GPT4All).',
  },
} as const;

export function getProviderConfig(kind: ProviderKind): ProviderConfig {
  return PROVIDER_REGISTRY[kind];
}

export function getProviderDisplayName(kind: ProviderKind): string {
  return PROVIDER_REGISTRY[kind].displayName;
}

export function getProvidersWithFreeTier(): ProviderKind[] {
  return Object.entries(PROVIDER_REGISTRY)
    .filter(([, config]) => config.freeTier)
    .map(([kind]) => kind as ProviderKind);
}

export function getProviderApiKeyField(kind: ProviderKind): string {
  const fieldMap: Record<ProviderKind, string> = {
    nim: 'nimApiKey',
    openrouter: 'openrouterApiKey',
    groq: 'groqApiKey',
    together: 'togetherApiKey',
    local: '',
  };
  return fieldMap[kind];
}

import type { Settings } from './types';

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

  const apiKey = (settings as unknown as Record<string, string>)[apiKeyField] || settings.apiKey || '';

  return {
    baseUrl: config.baseUrl,
    apiKey,
    providerLabel: config.displayName,
    defaultHeaders: config.defaultHeaders,
  };
}