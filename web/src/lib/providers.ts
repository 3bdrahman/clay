// Provider registry - all providers work client-side without proxy (except Ollama CORS)
// OpenRouter is default - works everywhere with generous free tier

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
    apiKeyEnvVar: 'VITE_OPENROUTER_API_KEY',
    apiKeyHint: 'sk-or-v1-...',
    freeTier: true,
    requiresApiKey: true,
    defaultHeaders: {
      'HTTP-Referer': 'https://clay-rag.netlify.app',
      'X-Title': 'Clay RAG',
    },
    apiKeyUrl: 'https://openrouter.ai/keys',
    description: '300+ models from all providers. Generous free tier. Best all-around choice.',
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
  nim: {
    kind: 'nim',
    displayName: 'NVIDIA NIM',
    baseUrl: 'https://integrate.api.nvidia.com/v1',
    modelsEndpoint: '/models',
    apiKeyEnvVar: 'VITE_NIM_API_KEY',
    apiKeyHint: 'nvapi-...',
    freeTier: true,
    requiresApiKey: true,
    apiKeyUrl: 'https://build.nvidia.com/settings/api-keys',
    description: 'NVIDIA hosted models. Only works from build.nvidia.com or via proxy. Not recommended for production deployments.',
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
    openrouter: 'openrouterApiKey',
    groq: 'groqApiKey',
    together: 'togetherApiKey',
    nim: 'nimApiKey',
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