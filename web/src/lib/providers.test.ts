import { describe, it, expect } from 'vitest';
import { resolveProviderEndpoint, LOCAL_DEFAULT_BASE_URL } from './providers';
import type { Settings } from './types';

const baseSettings: Settings = {
  provider: 'nim',
  nimApiKey: '',
  openrouterApiKey: '',
  groqApiKey: '',
  togetherApiKey: '',
  openaiApiKey: '',
  anthropicApiKey: '',
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
  pickedModelsOverride: {
    routing: '',
    codeGen: '',
    answer: '',
    eval: '',
    embedding: '',
  },
};

describe('resolveProviderEndpoint', () => {
  it('returns NIM base URL and nimApiKey for nim provider', () => {
    const out = resolveProviderEndpoint({ ...baseSettings, provider: 'nim', nimApiKey: 'nvapi-x' });
    expect(out.baseUrl).toBe('https://integrate.api.nvidia.com/v1');
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

  it('returns Groq base URL and groqApiKey', () => {
    const out = resolveProviderEndpoint({ ...baseSettings, provider: 'groq', groqApiKey: 'gsk_x' });
    expect(out.baseUrl).toBe('https://api.groq.com/openai/v1');
    expect(out.apiKey).toBe('gsk_x');
    expect(out.providerLabel).toBe('Groq');
  });

  it('returns Together base URL and togetherApiKey', () => {
    const out = resolveProviderEndpoint({ ...baseSettings, provider: 'together', togetherApiKey: 'together_x' });
    expect(out.baseUrl).toBe('https://api.together.xyz/v1');
    expect(out.apiKey).toBe('together_x');
    expect(out.providerLabel).toBe('Together AI');
  });
});
