// Provider base URLs

// NIM only serves CORS headers for build.nvidia.com. To make the 100%
// client-side app work in dev, vite.config.ts proxies /nim-api → NIM.
// On Netlify/Cloudflare, a Worker proxy handles CORS.
// In production on other hosts, set VITE_NIM_BASE_URL to your edge proxy.
const NIM_FALLBACK_BASE_URL = 'https://integrate.api.nvidia.com/v1';

// Cloudflare Worker proxy URL for NIM CORS bypass in production.
// Configure via VITE_CLOUDFLARE_PROXY_URL at build time.
// Falls back to a documented default for the official clay-rag.netlify.app deployment.
function resolveCloudflareProxy(): string {
  const envUrl = (import.meta.env.VITE_CLOUDFLARE_PROXY_URL as string | undefined)?.trim();
  if (envUrl) return envUrl;
  // Default to the official Clay proxy for the primary deployment.
  // Override at build time for custom deployments.
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

export const NIM_FREE_KEY_URL = 'https://build.nvidia.com/settings/api-keys';
export const NIM_KEY_HINT = 'nvapi-...';

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

import type { Settings } from './types';

export interface ProviderEndpoint {
  baseUrl: string;
  apiKey: string;
  providerLabel: string;
}

export function resolveProviderEndpoint(settings: Settings): ProviderEndpoint {
  if (settings.provider === 'local') {
    return {
      baseUrl: settings.localServerUrl.trim(),
      apiKey: '',
      providerLabel: 'Local server',
    };
  }
  return {
    baseUrl: NIM_BASE_URL,
    apiKey: settings.apiKey,
    providerLabel: 'NVIDIA NIM',
  };
}
