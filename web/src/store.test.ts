import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from './store';
import { LOCAL_DEFAULT_BASE_URL } from './lib/providers';
import type { LocalModelPicks } from './lib/types';

describe('useAppStore.updateSettings', () => {
  beforeEach(() => {
    useAppStore.setState({
      settings: {
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
      },
    });
  });

  it('clears localCatalog when switching from local to nim', () => {
    useAppStore.setState({
      settings: {
        ...useAppStore.getState().settings,
        provider: 'local',
        localCatalog: [{ id: 'llama3.1:8b', ownedBy: 'ollama', created: 0 }],
        localCatalogFetchedAt: 12345,
      },
    });
    useAppStore.getState().updateSettings({ provider: 'nim' });
    const after = useAppStore.getState().settings;
    expect(after.provider).toBe('nim');
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
    useAppStore.getState().updateSettings({ apiKey: '' });
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
});

describe('store persist migrate — LocalModelPicks 5-field → 2-field', () => {
  const migrate = () => useAppStore.persist.getOptions().migrate;

  it('migrates a persisted 5-field localModels into the 2-field chat+embeddings shape', () => {
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
    expect(out.settings.localModels).toEqual({ chat: 'a', embeddings: 'emb' });
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

  it('leaves a loaded localModels untouched when it already has the 2-field chat shape', () => {
    const persisted = {
      settings: {
        provider: 'local',
        localModels: { chat: 'llama3.1:8b', embeddings: 'nomic-embed-text' },
      },
    };
    const out = migrate()?.(persisted, 5) as { settings: { localModels: LocalModelPicks } };
    expect(out.settings.localModels).toEqual({ chat: 'llama3.1:8b', embeddings: 'nomic-embed-text' });
  });

  it('does not discard persisted localModels.embeddings when the old shape lacks chat', () => {
    const out = migrate()?.(
      { settings: { localModels: { routing: 'r', codeGen: '', answer: '', eval: '', embedding: 'nomic' } } },
      4,
    ) as { settings: { localModels: LocalModelPicks } };
    expect(out.settings.localModels.embeddings).toBe('nomic');
    expect(out.settings.localModels.chat).toBe('r');
  });
});
