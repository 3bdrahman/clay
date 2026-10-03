import { beforeEach, expect, it } from 'vitest';
import { validateSettings, getSettingsStatus } from './config';
import { useAppStore } from '../store';
import { NoProviderError } from './errors';

beforeEach(() => useAppStore.getState().resetAll());

it('requires the named OpenRouter key instead of an ambiguous legacy key', () => {
  const settings = { ...useAppStore.getState().settings, apiKey: 'legacy-other-provider-key' };
  expect(getSettingsStatus(settings).hasApiKey).toBe(false);
  expect(() => validateSettings(settings, { throwOnError: true })).toThrow(NoProviderError);
  expect(() => validateSettings(settings, { throwOnError: true })).toThrow(/openrouter/i);
});

it('accepts an OpenRouter key without any relay configuration', () => {
  const settings = { ...useAppStore.getState().settings, openrouterApiKey: 'sk-or-own-key' };
  expect(validateSettings(settings).valid).toBe(true);
  expect(getSettingsStatus(settings)).toMatchObject({ configured: true, hasApiKey: true, provider: 'OpenRouter' });
});

it('accepts a selected installed local model without a cloud API key', () => {
  const settings = {
    ...useAppStore.getState().settings,
    provider: 'local' as const,
    localModels: { chat: 'my-installed-model' },
    localCatalog: [{ id: 'my-installed-model', ownedBy: 'local', created: 0 }],
  };
  expect(validateSettings(settings).valid).toBe(true);
  expect(getSettingsStatus(settings).configured).toBe(true);
});
