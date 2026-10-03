import { beforeEach, describe, expect, it } from 'vitest';
import { useAppStore } from './store';
import { resolveProviderEndpoint } from './lib/providers';

type State = ReturnType<typeof useAppStore.getState>;

const retiredSettings = {
  provider: 'nim',
  apiKey: 'legacy-nvidia-key',
  nimApiKey: 'named-nvidia-key',
  nimBaseUrl: 'https://retired-relay.example/v1',
  nimProxyUrl: 'https://older-relay.example',
  pickedModelsOverride: { chatModel: 'nvidia/nemotron-3-super-120b-a12b' },
  webSearchProvider: 'duckduckgo',
  serperApiKey: 'saved-serper-key',
  localServerUrl: 'http://localhost:1234/v1',
  localModels: { chat: 'my-installed-model' },
};

function expectRetiredSettingsRemoved(state: State) {
  expect(state.settings.provider).toBe('openrouter');
  expect(state.settings.apiKey).toBe('');
  expect(state.settings).not.toHaveProperty('nimApiKey');
  expect(state.settings).not.toHaveProperty('nimBaseUrl');
  expect(state.settings).not.toHaveProperty('nimProxyUrl');
  expect(state.settings.pickedModelsOverride).toEqual({ chatModel: '' });
  expect(state.settings.webSearchProvider).toBe('mwmbl');
  expect(state.settings.serperApiKey).toBe('saved-serper-key');
  expect(state.availableModels).toEqual([]);
  expect(state.modelsFetchedAt).toBe(0);
  expect(state.modelsLoading).toBe(false);
  expect(state.modelsError).toBeNull();
}

beforeEach(() => useAppStore.getState().resetAll());

describe('retired provider persistence', () => {
  it.each([0, 6, 7, 8])('discards NVIDIA credentials and model state from saved version %s', async version => {
    const migrate = useAppStore.persist.getOptions().migrate!;
    const state = await migrate({
      settings: retiredSettings,
      availableModels: [{ id: 'retired-model', ownedBy: 'nvidia', created: 0 }],
      modelsFetchedAt: 123,
      modelsLoading: true,
      modelsError: 'retired provider error',
    }, version) as State;

    expectRetiredSettingsRemoved(state);
    expect(resolveProviderEndpoint(state.settings)).toMatchObject({
      baseUrl: 'https://openrouter.ai/api/v1',
      apiKey: '',
    });
    expect(state.settings.localServerUrl).toBe('http://localhost:1234/v1');
    expect(state.settings.localModels.chat).toBe('my-installed-model');
  });

  it('preserves a separately saved OpenRouter key and conversations during migration', async () => {
    const conversation = {
      id: 'saved-conversation', title: 'Saved analysis', createdAt: 1, updatedAt: 2,
      messages: [{ id: 'saved-message', role: 'user', content: 'Keep my data', timestamp: 1 }],
    };
    const migrate = useAppStore.persist.getOptions().migrate!;
    const state = await migrate({
      settings: { ...retiredSettings, openrouterApiKey: 'sk-or-own-key' },
      conversations: [conversation],
      activeConversationId: conversation.id,
    }, 8) as State;

    expectRetiredSettingsRemoved(state);
    expect(resolveProviderEndpoint(state.settings).apiKey).toBe('sk-or-own-key');
    expect(state.conversations).toEqual([conversation]);
    expect(state.activeConversationId).toBe(conversation.id);
  });

  it('applies the same credential and catalog isolation during same-version merging', () => {
    const merge = useAppStore.persist.getOptions().merge!;
    const state = merge({
      settings: retiredSettings,
      availableModels: [{ id: 'retired-model', ownedBy: 'nvidia', created: 0 }],
      modelsFetchedAt: 123,
      modelsLoading: true,
      modelsError: 'retired provider error',
    }, useAppStore.getState());

    expectRetiredSettingsRemoved(state);
    expect(resolveProviderEndpoint(state.settings).apiKey).toBe('');
  });

  it('removes inactive retired credentials without changing the active Local configuration', () => {
    const merge = useAppStore.persist.getOptions().merge!;
    const state = merge({
      settings: { ...retiredSettings, provider: 'local', openrouterApiKey: 'sk-or-own-key' },
    }, useAppStore.getState());

    expect(state.settings.provider).toBe('local');
    expect(state.settings.openrouterApiKey).toBe('sk-or-own-key');
    expect(state.settings.localModels.chat).toBe('my-installed-model');
    expect(state.settings).not.toHaveProperty('nimApiKey');
    expect(state.settings).not.toHaveProperty('nimBaseUrl');
    expect(resolveProviderEndpoint(state.settings)).toMatchObject({
      baseUrl: 'http://localhost:1234/v1', apiKey: '',
    });
  });

  it('preserves an existing OpenRouter selection while removing inactive NVIDIA credentials', () => {
    const merge = useAppStore.persist.getOptions().merge!;
    const selected = 'nvidia/nemotron-3-super-120b-a12b:free';
    const state = merge({
      settings: {
        ...retiredSettings,
        provider: 'openrouter',
        openrouterApiKey: 'sk-or-own-key',
        pickedModelsOverride: { chatModel: selected },
      },
    }, useAppStore.getState());

    expect(state.settings.pickedModelsOverride).toEqual({ chatModel: selected });
    expect(resolveProviderEndpoint(state.settings).apiKey).toBe('sk-or-own-key');
    expect(state.settings).not.toHaveProperty('nimApiKey');
    expect(state.settings).not.toHaveProperty('nimBaseUrl');
  });

  it('rewrites saved version 7 without obsolete keys when the browser rehydrates', async () => {
    const storage = useAppStore.persist.getOptions().storage!;
    await storage.setItem('clay-settings-v1', {
      version: 8,
      state: { settings: retiredSettings } as unknown as State,
    });

    await useAppStore.persist.rehydrate();

    expectRetiredSettingsRemoved(useAppStore.getState());
    const saved = await storage.getItem('clay-settings-v1');
    expect(saved?.version).toBe(useAppStore.persist.getOptions().version);
    expect(saved?.state.settings.provider).toBe('openrouter');
    expect(saved?.state.settings.webSearchProvider).toBe('mwmbl');
    expect(saved?.state.settings).not.toHaveProperty('nimApiKey');
    expect(saved?.state.settings).not.toHaveProperty('nimBaseUrl');
    expect(saved?.state.settings).not.toHaveProperty('nimProxyUrl');
  });
});
