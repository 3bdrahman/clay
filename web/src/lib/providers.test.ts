import { describe, it, expect } from 'vitest';
import { resolveProviderEndpoint } from './providers';
import { LOCAL_DEFAULT_BASE_URL, NIM_BASE_URL } from './providers';
import type { Settings } from './types';

const baseSettings: Settings = {
  provider: 'nim',
  apiKey: '',
  embeddingApiKey: '',
  webSearchProvider: 'duckduckgo',
  serperApiKey: '',
  temperature: 0,
  maxRetries: 3,
  theme: 'system',
  localServerUrl: LOCAL_DEFAULT_BASE_URL,
  localModels: { chat: '', embeddings: '' },
  localCatalog: [],
  localCatalogFetchedAt: 0,
};

describe('resolveProviderEndpoint', () => {
  it('returns NIM base URL and apiKey for nim provider', () => {
    const out = resolveProviderEndpoint({ ...baseSettings, provider: 'nim', apiKey: 'nvapi-x' });
    expect(out.baseUrl).toBe(NIM_BASE_URL);
    expect(out.apiKey).toBe('nvapi-x');
    expect(out.providerLabel).toBe('NVIDIA NIM');
  });

  it('returns local server URL and empty key for local provider', () => {
    const out = resolveProviderEndpoint({
      ...baseSettings,
      provider: 'local',
      localServerUrl: 'http://localhost:1234/v1',
    });
    expect(out.baseUrl).toBe('http://localhost:1234/v1');
    expect(out.apiKey).toBe('');
    expect(out.providerLabel).toBe('Local server');
  });

  it('trims whitespace from the local server URL', () => {
    const out = resolveProviderEndpoint({
      ...baseSettings,
      provider: 'local',
      localServerUrl: '   http://localhost:11434/v1   ',
    });
    expect(out.baseUrl).toBe('http://localhost:11434/v1');
  });

  it('local provider does not require an apiKey (returns empty string even when set)', () => {
    const out = resolveProviderEndpoint({
      ...baseSettings,
      provider: 'local',
      apiKey: 'this-should-be-ignored',
    });
    expect(out.apiKey).toBe('');
  });
});
