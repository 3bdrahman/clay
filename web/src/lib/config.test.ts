import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { validateSettings, getSettingsStatus } from './config';
import { useAppStore } from '../store';
import { NoProviderError, ProviderUnreachableError } from './errors';

beforeEach(() => useAppStore.getState().resetAll());
afterEach(() => vi.unstubAllEnvs());

it('requires the selected NIM key instead of an OpenRouter or legacy key', () => {
  const settings = { ...useAppStore.getState().settings, provider: 'nim' as const, openrouterApiKey: 'other-key', apiKey: 'legacy', nimBaseUrl: 'https://relay.example/v1' };
  expect(getSettingsStatus(settings).hasApiKey).toBe(false);
  expect(() => validateSettings(settings, { throwOnError: true })).toThrow(NoProviderError);
  expect(() => validateSettings(settings, { throwOnError: true })).toThrow(/nim/i);
});

it('reports a missing production relay as an actionable typed provider error', () => {
  vi.stubEnv('DEV', false);
  vi.stubEnv('VITE_NIM_BASE_URL', '');
  const settings = { ...useAppStore.getState().settings, provider: 'nim' as const, nimApiKey: 'nvapi-test' };
  expect(validateSettings(settings).valid).toBe(false);
  expect(() => validateSettings(settings, { throwOnError: true })).toThrow(ProviderUnreachableError);
  expect(() => validateSettings(settings, { throwOnError: true })).toThrow(/relay/i);
});

it('accepts a configured NIM relay and its matching provider key', () => {
  const settings = { ...useAppStore.getState().settings, provider: 'nim' as const, nimApiKey: 'nvapi-test', nimBaseUrl: 'https://relay.example/v1' };
  expect(validateSettings(settings).valid).toBe(true);
  expect(getSettingsStatus(settings)).toMatchObject({ configured: true, hasApiKey: true, provider: 'NVIDIA NIM' });
});
