import { describe, it, expect, afterEach, vi } from 'vitest';
import { resolveProviderEndpoint, LOCAL_DEFAULT_BASE_URL, PROVIDER_REGISTRY } from './providers';
import type { Settings } from './types';

const baseSettings: Settings = {
  provider: 'openrouter',
  openrouterApiKey: '',
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

  it('offers exactly OpenRouter and local', () => {
    expect(Object.keys(PROVIDER_REGISTRY)).toEqual(['openrouter', 'local']);
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

  it('never sends an ambiguous legacy key to a cloud provider', () => {
    expect(resolveProviderEndpoint({ ...baseSettings, apiKey: 'old-provider-key' }).apiKey).toBe('');
  });

});
