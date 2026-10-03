import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore, sanitizeProvider } from './store';
import { LOCAL_DEFAULT_BASE_URL, PROVIDER_REGISTRY } from './lib/providers';
import type { LocalModelPicks } from './lib/types';

describe('useAppStore.updateSettings', () => {
  beforeEach(() => {
    useAppStore.setState({
      settings: {
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
        pickedModelsOverride: { chatModel: '' },
      },
    });
  });

  it('clears localCatalog when switching from local to openrouter', () => {
    useAppStore.setState({
      settings: {
        ...useAppStore.getState().settings,
        provider: 'local',
        localCatalog: [{ id: 'llama3.1:8b', ownedBy: 'ollama', created: 0 }],
        localCatalogFetchedAt: 12345,
      },
    });
    useAppStore.getState().updateSettings({ provider: 'openrouter' });
    const after = useAppStore.getState().settings;
    expect(after.provider).toBe('openrouter');
    expect(after.localCatalog).toEqual([]);
    expect(after.localCatalogFetchedAt).toBe(0);
  });

  it('does not clear localCatalog when staying on local', () => {
    useAppStore.setState({
      settings: {
        ...useAppStore.getState().settings,
        provider: 'local',
        localCatalog: [{ id: 'llama3.1:8b', ownedBy: 'ollama', created: 0 }],
        localCatalogFetchedAt: 12345,
      },
    });
    useAppStore.getState().updateSettings({ openrouterApiKey: '' });
    const after = useAppStore.getState().settings;
    expect(after.provider).toBe('local');
    expect(after.localCatalog.length).toBe(1);
    expect(after.localCatalogFetchedAt).toBe(12345);
  });

  it('updates localServerUrl without clearing the catalog', () => {
    useAppStore.setState({
      settings: {
        ...useAppStore.getState().settings,
        provider: 'local',
        localCatalog: [{ id: 'llama3.1:8b', ownedBy: 'ollama', created: 0 }],
      },
    });
    useAppStore.getState().updateSettings({ localServerUrl: 'http://localhost:1234/v1' });
    const after = useAppStore.getState().settings;
    expect(after.localServerUrl).toBe('http://localhost:1234/v1');
    expect(after.localCatalog.length).toBe(1);
  });

  it('clears cloud model state when switching to Local while preserving the OpenRouter key and local picks', () => {
    useAppStore.setState({
      settings: {
        ...useAppStore.getState().settings,
        provider: 'openrouter',
        openrouterApiKey: 'openrouter-key',
        pickedModelsOverride: { chatModel: 'openrouter/model' },
        localModels: { chat: 'llama3.1:8b' },
      },
      availableModels: [{ id: 'openrouter/model', ownedBy: 'openrouter', created: 1 }],
      modelsFetchedAt: 12345,
      modelsError: 'previous provider failed',
      modelsLoading: true,
    });

    useAppStore.getState().updateSettings({ provider: 'local' });

    const state = useAppStore.getState();
    expect(state.settings.provider).toBe('local');
    expect(state.settings.openrouterApiKey).toBe('openrouter-key');
    expect(state.settings.localModels).toEqual({ chat: 'llama3.1:8b' });
    expect(state.settings.pickedModelsOverride).toEqual({ chatModel: '' });
    expect(state.availableModels).toEqual([]);
    expect(state.modelsFetchedAt).toBe(0);
    expect(state.modelsError).toBeNull();
    expect(state.modelsLoading).toBe(false);
  });

  it('preserves provider-scoped model state when inactive cloud settings change', () => {
    useAppStore.setState({
      settings: {
        ...useAppStore.getState().settings,
        provider: 'local',
        pickedModelsOverride: { chatModel: 'openrouter/model' },
      },
      availableModels: [{ id: 'openrouter/model', ownedBy: 'openrouter', created: 1 }],
      modelsFetchedAt: 12345,
      modelsError: 'kept until next fetch',
      modelsLoading: true,
    });

    useAppStore.getState().updateSettings({ openrouterApiKey: 'inactive-new-key' });

    const state = useAppStore.getState();
    expect(state.settings.openrouterApiKey).toBe('inactive-new-key');
    expect(state.settings.pickedModelsOverride).toEqual({ chatModel: 'openrouter/model' });
    expect(state.availableModels).toEqual([{ id: 'openrouter/model', ownedBy: 'openrouter', created: 1 }]);
    expect(state.modelsFetchedAt).toBe(12345);
    expect(state.modelsError).toBe('kept until next fetch');
    expect(state.modelsLoading).toBe(true);
  });

  it('invalidates cached models while preserving selection when the active cloud API key changes', () => {
    useAppStore.setState({
      settings: {
        ...useAppStore.getState().settings,
        provider: 'openrouter',
        openrouterApiKey: 'old-openrouter-key',
        pickedModelsOverride: { chatModel: 'openrouter/model' },
      },
      availableModels: [{ id: 'openrouter/model', ownedBy: 'openrouter', created: 1 }],
      modelsFetchedAt: 12345,
      modelsError: 'previous key failed',
      modelsLoading: true,
    });

    useAppStore.getState().updateSettings({ openrouterApiKey: 'new-openrouter-key' });

    const state = useAppStore.getState();
    expect(state.settings.openrouterApiKey).toBe('new-openrouter-key');
    expect(state.settings.pickedModelsOverride).toEqual({ chatModel: 'openrouter/model' });
    expect(state.availableModels).toEqual([]);
    expect(state.modelsFetchedAt).toBe(0);
    expect(state.modelsError).toBeNull();
    expect(state.modelsLoading).toBe(false);
  });

});

describe('store persist migrate — LocalModelPicks legacy shape → chat-only', () => {
  const migrate = () => useAppStore.persist.getOptions().migrate;

  it('migrates a persisted 5-field localModels into the chat-only shape, dropping embedding picks', () => {
    const persistedOld = {
      settings: {
        provider: 'local',
        localServerUrl: 'http://localhost:11434/v1',
        localModels: {
          routing: 'r',
          codeGen: 'c',
          answer: 'a',
          eval: 'e',
          embedding: 'emb',
        },
      },
    };
    const out = migrate()?.(persistedOld, 4) as { settings: { localModels: LocalModelPicks } };
    expect(out.settings.localModels).toEqual({ chat: 'a' });
  });

  it('prefers answer for chat carryover, then routing, then codeGen/eval, then first non-empty', () => {
    const migrateFn = migrate();
    expect(migrateFn).toBeDefined();

    const noAnswer = migrateFn?.(
      { settings: { localModels: { routing: 'r', codeGen: '', answer: '', eval: '', embedding: 'emb' } } },
      4,
    ) as { settings: { localModels: LocalModelPicks } };
    expect(noAnswer.settings.localModels.chat).toBe('r');

    const noAnswerNoRouting = migrateFn?.(
      { settings: { localModels: { routing: '', codeGen: 'cg', answer: '', eval: 'ev', embedding: 'emb' } } },
      4,
    ) as { settings: { localModels: LocalModelPicks } };
    expect(noAnswerNoRouting.settings.localModels.chat).toBe('cg');

    const onlyEval = migrateFn?.(
      { settings: { localModels: { routing: '', codeGen: '', answer: '', eval: 'ev', embedding: 'emb' } } },
      4,
    ) as { settings: { localModels: LocalModelPicks } };
    expect(onlyEval.settings.localModels.chat).toBe('ev');
  });

  it('drops legacy embedding picks even from already-migrated localModels', () => {
    const persisted = {
      settings: {
        provider: 'local',
        localModels: { chat: 'llama3.1:8b', embeddings: 'nomic-embed-text' },
      },
    };
    const out = migrate()?.(persisted, 5) as { settings: { localModels: LocalModelPicks } };
    expect(out.settings.localModels).toEqual({ chat: 'llama3.1:8b' });
  });

  it('falls back to openrouter when persisted provider is no longer registered', () => {
    const out = migrate()?.(
      { settings: { provider: 'removed-provider' as never, openrouterApiKey: 'legacy-key' } },
      5,
    ) as { settings: { provider: string; openrouterApiKey: string } };
    expect(out.settings.provider).toBe('openrouter');
    expect(out.settings.openrouterApiKey).toBe('legacy-key');
  });

  it('migrates removed Groq state to the default provider without leaking legacy key or model selection', () => {
    const out = migrate()?.(
      {
        settings: {
          provider: 'groq' as never,
          apiKey: 'legacy-groq-key',
          groqApiKey: 'named-groq-key',
          openrouterApiKey: '',
          nimApiKey: '',
          pickedModelsOverride: { chatModel: 'llama-3.3-70b-versatile' },
        },
      },
      6,
    ) as { settings: Record<string, unknown> };
    expect(out.settings.provider).toBe('openrouter');
    expect(out.settings.openrouterApiKey).toBe('');
    expect(out.settings.nimApiKey).toBeUndefined();
    expect(out.settings.apiKey).toBe('');
    expect(out.settings.groqApiKey).toBeUndefined();
    expect(out.settings.pickedModelsOverride).toEqual({ chatModel: '' });
  });

  it('migrates explicitly matching legacy OpenRouter keys without overwriting a named cleared key', () => {
    const migratedLegacy = migrate()?.(
      { settings: { provider: 'openrouter', apiKey: 'legacy-openrouter-key' } },
      6,
    ) as { settings: Record<string, unknown> };
    expect(migratedLegacy.settings.provider).toBe('openrouter');
    expect(migratedLegacy.settings.openrouterApiKey).toBe('legacy-openrouter-key');
    expect(migratedLegacy.settings.apiKey).toBe('');

    const preservedNamedClear = migrate()?.(
      { settings: { provider: 'openrouter', apiKey: 'legacy-openrouter-key', openrouterApiKey: '' } },
      6,
    ) as { settings: Record<string, unknown> };
    expect(preservedNamedClear.settings.openrouterApiKey).toBe('');
    expect(preservedNamedClear.settings.apiKey).toBe('');
  });

  it('falls back to openrouter when persisted provider is garbage', () => {
    const out = migrate()?.(
      { settings: { provider: 'does-not-exist' as never } },
      5,
    ) as { settings: { provider: string } };
    expect(out.settings.provider).toBe('openrouter');
  });
});

describe('persisted chat selection', () => {
  it('migrates an old answer selection on same-version rehydration', async () => {
    const current = useAppStore.getState();
    const saved = localStorage.getItem('clay-settings-v1');
    try {
      localStorage.setItem('clay-settings-v1', JSON.stringify({
        version: useAppStore.persist.getOptions().version,
        state: {
          settings: {
            ...current.settings,
            pickedModelsOverride: { answer: 'chosen-answer', routing: 'old-router', embedding: 'embed' },
          },
        },
      }));
      await useAppStore.persist.rehydrate();
      expect(useAppStore.getState().settings.pickedModelsOverride).toEqual({
        chatModel: 'chosen-answer',
      });
    } finally {
      useAppStore.setState(current);
      if (saved === null) localStorage.removeItem('clay-settings-v1');
      else localStorage.setItem('clay-settings-v1', saved);
    }
  });

  it('preserves an explicitly cleared chat selection instead of restoring legacy picks', () => {
    const merge = useAppStore.persist.getOptions().merge;
    const result = merge?.({
      settings: { pickedModelsOverride: { chatModel: '', answer: 'legacy', embedding: 'embed' } },
    }, useAppStore.getState());
    expect(result?.settings.pickedModelsOverride).toEqual({ chatModel: '' });
  });

  it('drops removed Groq credentials during same-version rehydration without moving them to active providers', () => {
    const merge = useAppStore.persist.getOptions().merge;
    const result = merge?.({
      settings: {
        provider: 'groq' as never,
        apiKey: 'legacy-groq-key',
        groqApiKey: 'named-groq-key',
        openrouterApiKey: '',
        nimApiKey: '',
        pickedModelsOverride: { chatModel: 'groq-model' },
      },
    }, useAppStore.getState()) as { settings: Record<string, unknown> } | undefined;
    expect(result?.settings.provider).toBe('openrouter');
    expect(result?.settings.openrouterApiKey).toBe('');
    expect(result?.settings.nimApiKey).toBeUndefined();
    expect(result?.settings.apiKey).toBe('');
    expect(result?.settings.groqApiKey).toBeUndefined();
    expect(result?.settings.pickedModelsOverride).toEqual({ chatModel: '' });
  });
});

describe('sanitizeProvider', () => {
  it('returns openrouter for unknown provider', () => {
    expect(sanitizeProvider('unknown')).toBe('openrouter');
  });

  it('returns openrouter for non-string input', () => {
    expect(sanitizeProvider(null)).toBe('openrouter');
    expect(sanitizeProvider(undefined)).toBe('openrouter');
    expect(sanitizeProvider(123)).toBe('openrouter');
  });

  it('returns the provider when it is registered', () => {
    expect(sanitizeProvider('openrouter')).toBe('openrouter');
    expect(sanitizeProvider('local')).toBe('local');
  });

  it('does not accept provider names inherited from the registry prototype', () => {
    const originalPrototype = Object.getPrototypeOf(PROVIDER_REGISTRY);
    try {
      Object.setPrototypeOf(PROVIDER_REGISTRY, { malicious: { kind: 'malicious' } });
      expect(sanitizeProvider('malicious')).toBe('openrouter');
    } finally {
      Object.setPrototypeOf(PROVIDER_REGISTRY, originalPrototype);
    }
  });
});

describe('useAppStore.updateMessage routing', () => {
  beforeEach(() => {
    useAppStore.setState({ conversations: [], activeConversationId: null });
  });

  it('routes the update to the conversation containing the message id, not the active one', () => {
    const convA = {
      id: 'conv-a',
      title: 'A',
      messages: [{ id: 'msg-1', role: 'assistant' as const, content: '', timestamp: 0 }],
      createdAt: 0,
      updatedAt: 0,
    };
    const convB = { id: 'conv-b', title: 'B', messages: [], createdAt: 0, updatedAt: 0 };
    useAppStore.setState({ conversations: [convB, convA], activeConversationId: 'conv-b' });

    useAppStore.getState().updateMessage('msg-1', (m) => ({ ...m, content: 'final answer' }));

    const state = useAppStore.getState();
    expect(state.conversations.find(c => c.id === 'conv-a')?.messages[0]?.content).toBe('final answer');
  });

  it('is a no-op when no conversation contains the message id', () => {
    const convA = { id: 'conv-a', title: 'A', messages: [{ id: 'msg-1', role: 'assistant' as const, content: 'kept', timestamp: 0 }], createdAt: 0, updatedAt: 0 };
    useAppStore.setState({ conversations: [convA], activeConversationId: 'conv-a' });

    useAppStore.getState().updateMessage('msg-missing', (m) => ({ ...m, content: 'changed' }));

    const state = useAppStore.getState();
    expect(state.conversations[0]?.messages[0]?.content).toBe('kept');
  });
});
