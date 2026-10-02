import { describe, it, expect, afterEach, vi } from 'vitest';
import { resolveProviderEndpoint, LOCAL_DEFAULT_BASE_URL, PROVIDER_REGISTRY } from './providers';
import type { Settings } from './types';

const baseSettings: Settings = {
  provider: 'openrouter',
  openrouterApiKey: '',
  nimApiKey: '',
  apiKey: '',
  webSearchProvider: 'duckduckgo',
  serperApiKey: '',
  temperature: 0,
  maxRetries: 3,
  theme: 'system',
  localServerUrl: LOCAL_DEFAULT_BASE_URL,
  localModels: { chat: '' },
  localCatalog: [],
  localCatalogFetchedAt: 0,
  pickedModelsOverride: {
    chatModel: '',
  },
};

describe('resolveProviderEndpoint', () => {
  afterEach(() => vi.unstubAllEnvs());

  it('offers exactly OpenRouter, NVIDIA NIM, and local', () => {
    expect(Object.keys(PROVIDER_REGISTRY)).toEqual(['openrouter', 'nim', 'local']);
  });

  it('returns local server URL and empty key for local provider', () => {
    const out = resolveProviderEndpoint({
      ...baseSettings,
      provider: 'local',
      localServerUrl: 'http://localhost:1234/v1',
    });
    expect(out.baseUrl).toBe('http://localhost:1234/v1');
    expect(out.apiKey).toBe('');
    expect(out.providerLabel).toBe('Local (OpenAI-compatible)');
  });

  it('trims whitespace from the local server URL', () => {
    const out = resolveProviderEndpoint({
      ...baseSettings,
      provider: 'local',
      localServerUrl: '   http://localhost:11434/v1   ',
    });
    expect(out.baseUrl).toBe('http://localhost:11434/v1');
  });

  it('local provider does not require an apiKey (returns empty string even when legacy apiKey is set)', () => {
    const out = resolveProviderEndpoint({
      ...baseSettings,
      provider: 'local',
      apiKey: 'this-should-be-ignored',
    });
    expect(out.apiKey).toBe('');
  });

  it('returns OpenRouter base URL and openrouterApiKey', () => {
    const out = resolveProviderEndpoint({ ...baseSettings, provider: 'openrouter', openrouterApiKey: 'sk-or-v1-x' });
    expect(out.baseUrl).toBe('https://openrouter.ai/api/v1');
    expect(out.apiKey).toBe('sk-or-v1-x');
    expect(out.providerLabel).toBe('OpenRouter');
  });

  it('routes NIM through its configured gateway with only its own API key', () => {
    const out = resolveProviderEndpoint({
      ...baseSettings, provider: 'nim', nimApiKey: 'nvapi-test', openrouterApiKey: 'sk-or-other',
      nimBaseUrl: ' https://my-nim.example/v1/ ',
    });
    expect(out.baseUrl).toBe('https://my-nim.example/v1');
    expect(out.apiKey).toBe('nvapi-test');
    expect(out.providerLabel).toBe('NVIDIA NIM');
    expect(out.configurationError).toBeUndefined();
  });

  it('never sends an ambiguous legacy key to a cloud provider', () => {
    expect(resolveProviderEndpoint({ ...baseSettings, apiKey: 'old-provider-key' }).apiKey).toBe('');
  });

  it('uses the real same-origin NIM proxy during development', () => {
    vi.stubEnv('DEV', true);
    vi.stubEnv('VITE_NIM_BASE_URL', '');
    const out = resolveProviderEndpoint({ ...baseSettings, provider: 'nim' });
    expect(out.baseUrl).toBe('/nim-api/v1');
    expect(out.configurationError).toBeUndefined();
  });

  it('requires a relay in production instead of advertising a blocked direct request', () => {
    vi.stubEnv('DEV', false);
    vi.stubEnv('VITE_NIM_BASE_URL', '');
    const out = resolveProviderEndpoint({ ...baseSettings, provider: 'nim', nimApiKey: 'nvapi-test' });
    expect(out.configurationError).toMatch(/relay/i);
  });

  it('supports a deployment-configured NIM relay without embedding a key', () => {
    vi.stubEnv('DEV', false);
    vi.stubEnv('VITE_NIM_BASE_URL', 'https://clay-nim.workers.dev/v1/');
    const out = resolveProviderEndpoint({ ...baseSettings, provider: 'nim' });
    expect(out.baseUrl).toBe('https://clay-nim.workers.dev/v1');
    expect(out.apiKey).toBe('');
    expect(out.configurationError).toBeUndefined();
  });

  it.each([
    'https://integrate.api.nvidia.com/v1',
    'javascript:alert(1)',
    'https://user:password@relay.example/v1',
    'https://relay.example/v1?key=secret',
    'https://relay.example/v1#fragment',
    'http://remote.example/v1',
    'https://relay.example',
  ])('rejects an unusable or unsafe NIM base URL: %s', nimBaseUrl => {
    const out = resolveProviderEndpoint({ ...baseSettings, provider: 'nim', nimBaseUrl });
    expect(out.configurationError).toBeTruthy();
  });
});
