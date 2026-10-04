import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useFetchCloudModels, useFetchLocalModels } from './useModelCatalog';
import { useAppStore } from '../store';
import { listLocalCatalog, listModels } from '../lib/models';
import type { ModelInfo, Settings } from '../lib/types';

vi.mock('../lib/models', () => ({ listLocalCatalog: vi.fn(), listModels: vi.fn() }));

let fetchModels: ReturnType<typeof useFetchCloudModels>;
let fetchLocalModels: ReturnType<typeof useFetchLocalModels>;
let root: ReturnType<typeof createRoot>;
let container: HTMLDivElement;

function Harness() {
  const state = useAppStore();
  fetchModels = useFetchCloudModels({
    settings: state.settings, availableModels: state.availableModels,
    modelsFetchedAt: state.modelsFetchedAt,
    setModels: state.setModels, setModelsLoading: state.setModelsLoading,
    setModelsError: state.setModelsError, setLocalCatalog: state.setLocalCatalog,
    getSettings: () => useAppStore.getState().settings,
  });
  fetchLocalModels = useFetchLocalModels({
    settings: state.settings, availableModels: state.availableModels,
    modelsFetchedAt: state.modelsFetchedAt,
    setModels: state.setModels, setModelsLoading: state.setModelsLoading,
    setModelsError: state.setModelsError, setLocalCatalog: state.setLocalCatalog,
    getSettings: () => useAppStore.getState().settings,
  });
  return null;
}

async function configure(patch: Partial<Settings>) {
  await act(async () => useAppStore.getState().updateSettings(patch));
}

beforeEach(async () => {
  vi.clearAllMocks();
  useAppStore.getState().resetAll();
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => root.render(<Harness />));
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.unstubAllEnvs();
});

describe('cloud catalog provider boundary', () => {
  it('discovers OpenRouter models at the configured provider endpoint used by chat', async () => {
    await configure({ openrouterApiKey: 'sk-or-test' });
    vi.mocked(listModels).mockResolvedValue([{ id: 'openrouter-chat', ownedBy: 'openrouter', created: 0 }]);
    await act(async () => { await fetchModels('sk-or-test'); });
    expect(listModels).toHaveBeenCalledWith('openrouter', 'sk-or-test', 'https://openrouter.ai/api/v1');
    expect(useAppStore.getState().availableModels[0].id).toBe('openrouter-chat');
  });

  it('does not reuse a cached catalog when the key changes', async () => {
    vi.mocked(listModels).mockResolvedValue([{ id: 'first-model', ownedBy: 'test', created: 0 }]);
    await configure({ openrouterApiKey: 'first-key' });
    await act(async () => { await fetchModels('first-key'); });
    await configure({ openrouterApiKey: 'second-key' });
    await act(async () => { await fetchModels('second-key'); });
    expect(listModels).toHaveBeenCalledTimes(2);
    expect(listModels).toHaveBeenLastCalledWith('openrouter', 'second-key', 'https://openrouter.ai/api/v1');
  });

  it('discards an old key response that finishes after the key changes', async () => {
    let finishOld!: (models: ModelInfo[]) => void;
    vi.mocked(listModels).mockImplementationOnce(() => new Promise(resolve => { finishOld = resolve; }));
    await configure({ openrouterApiKey: 'old-key' });
    let pending!: Promise<ModelInfo[]>;
    await act(async () => { pending = fetchModels('old-key'); });
    await configure({ openrouterApiKey: 'new-key' });
    const currentCatalog = [{ id: 'openrouter-current', ownedBy: 'openrouter', created: 0 }];
    vi.mocked(listModels).mockResolvedValueOnce(currentCatalog);
    await act(async () => { await fetchModels('new-key'); });
    await act(async () => { finishOld([{ id: 'old-model', ownedBy: 'other', created: 0 }]); await pending; });
    expect(useAppStore.getState().availableModels).toEqual(currentCatalog);
    expect(useAppStore.getState().modelsError).toBeNull();
    expect(useAppStore.getState().modelsLoading).toBe(false);
  });

  it('discards an old cloud response after switching to Local', async () => {
    let finishOld!: (models: ModelInfo[]) => void;
    vi.mocked(listModels).mockImplementationOnce(() => new Promise(resolve => { finishOld = resolve; }));
    await configure({ openrouterApiKey: 'cloud-key' });
    let pending!: Promise<ModelInfo[]>;
    await act(async () => { pending = fetchModels('cloud-key'); });

    const localCatalog = [{ id: 'llama3:8b', ownedBy: 'ollama', created: 0 }];
    await configure({
      provider: 'local',
      localModels: { chat: 'llama3:8b' },
      localCatalog,
      localCatalogFetchedAt: Date.now(),
    });
    useAppStore.getState().setModelsLoading(false);

    await act(async () => { finishOld([{ id: 'old-cloud-model', ownedBy: 'other', created: 0 }]); await pending; });

    expect(useAppStore.getState().settings.provider).toBe('local');
    expect(useAppStore.getState().settings.localCatalog).toEqual(localCatalog);
    expect(useAppStore.getState().availableModels).toEqual([]);
    expect(useAppStore.getState().modelsError).toBeNull();
    expect(useAppStore.getState().modelsLoading).toBe(false);
  });
});

describe('local catalog endpoint boundary', () => {
  it('clears stale cached models after a failed current-server refresh', async () => {
    await configure({
      provider: 'local', localServerUrl: 'http://localhost:11434/v1',
      localCatalog: [{ id: 'stale-model', ownedBy: 'local', created: 0 }],
      localCatalogBaseUrl: 'http://localhost:11434/v1', localCatalogFetchedAt: Date.now(),
    });
    vi.mocked(listLocalCatalog).mockRejectedValueOnce(new Error('Current server unreachable'));
    await act(async () => { await fetchLocalModels('http://localhost:11434/v1', true); });
    expect(useAppStore.getState().settings.localCatalog).toEqual([]);
    expect(useAppStore.getState().settings.localCatalogBaseUrl).toBe('');
    expect(useAppStore.getState().modelsError).toBe('Current server unreachable');
    expect(useAppStore.getState().modelsLoading).toBe(false);
  });

  it('does not let an obsolete local callback overwrite cloud errors', async () => {
    await configure({ provider: 'local', localServerUrl: 'invalid-old-url' });
    const previousCallback = fetchLocalModels;
    await configure({ provider: 'openrouter' });
    useAppStore.getState().setModelsError('Current cloud error');
    await act(async () => { await previousCallback('invalid-old-url', true); });
    expect(useAppStore.getState().modelsError).toBe('Current cloud error');
    expect(listLocalCatalog).not.toHaveBeenCalled();
  });

  it('reuses a cached local catalog only for the same normalized server URL', async () => {
    const cached = [{ id: 'llama3:8b', ownedBy: 'ollama', created: 0 }];
    await configure({
      provider: 'local',
      localServerUrl: 'http://localhost:11434/v1/',
      localCatalog: cached,
      localCatalogBaseUrl: 'http://localhost:11434/v1',
      localCatalogFetchedAt: Date.now(),
    });

    let result!: ModelInfo[];
    await act(async () => { result = await fetchLocalModels('http://localhost:11434'); });

    expect(result).toEqual(cached);
    expect(listLocalCatalog).not.toHaveBeenCalled();
  });

  it('does not use an old local catalog cache for a different server', async () => {
    await configure({
      provider: 'local',
      localServerUrl: 'http://localhost:1234/v1',
      localCatalog: [{ id: 'old-model', ownedBy: 'ollama', created: 0 }],
      localCatalogBaseUrl: 'http://localhost:11434/v1',
      localCatalogFetchedAt: Date.now(),
    });
    const nextCatalog = [{ id: 'lm-studio-model', ownedBy: 'lmstudio', created: 0 }];
    vi.mocked(listLocalCatalog).mockResolvedValue(nextCatalog);

    let result!: ModelInfo[];
    await act(async () => { result = await fetchLocalModels('http://localhost:1234/v1'); });

    expect(listLocalCatalog).toHaveBeenCalledWith('http://localhost:1234/v1', '');
    expect(result).toEqual(nextCatalog);
    expect(useAppStore.getState().settings.localCatalog).toEqual(nextCatalog);
    expect(useAppStore.getState().settings.localCatalogBaseUrl).toBe('http://localhost:1234/v1');
  });

  it('does not make a network request for an invalid local URL', async () => {
    await configure({ provider: 'local', localServerUrl: 'not-a-url' });

    await act(async () => { await fetchLocalModels('not-a-url', true); });

    expect(listLocalCatalog).not.toHaveBeenCalled();
    expect(useAppStore.getState().modelsError).toContain('complete HTTP or HTTPS');
    expect(useAppStore.getState().modelsLoading).toBe(false);
  });

  it('discards an old local response after the local URL changes', async () => {
    let finishOld!: (models: ModelInfo[]) => void;
    vi.mocked(listLocalCatalog).mockImplementationOnce(() => new Promise(resolve => { finishOld = resolve; }));
    await configure({ provider: 'local', localServerUrl: 'http://localhost:11434/v1' });
    let pending!: Promise<ModelInfo[]>;
    await act(async () => { pending = fetchLocalModels('http://localhost:11434/v1', true); });

    await configure({ provider: 'local', localServerUrl: 'http://localhost:1234/v1' });
    const currentCatalog = [{ id: 'current-local', ownedBy: 'lmstudio', created: 0 }];
    vi.mocked(listLocalCatalog).mockResolvedValueOnce(currentCatalog);
    await act(async () => { await fetchLocalModels('http://localhost:1234/v1', true); });
    await act(async () => { finishOld([{ id: 'stale-local', ownedBy: 'ollama', created: 0 }]); await pending; });

    expect(useAppStore.getState().settings.localCatalog).toEqual(currentCatalog);
    expect(useAppStore.getState().settings.localCatalogBaseUrl).toBe('http://localhost:1234/v1');
    expect(useAppStore.getState().modelsError).toBeNull();
    expect(useAppStore.getState().modelsLoading).toBe(false);
  });

  it('discards an old local error after switching providers', async () => {
    let failOld!: (error: Error) => void;
    vi.mocked(listLocalCatalog).mockImplementationOnce(() => new Promise((_resolve, reject) => { failOld = reject; }));
    await configure({ provider: 'local', localServerUrl: 'http://localhost:11434/v1' });
    let pending!: Promise<ModelInfo[]>;
    await act(async () => { pending = fetchLocalModels('http://localhost:11434/v1', true); });

    await configure({ provider: 'openrouter' });
    useAppStore.getState().setModelsLoading(false);
    await act(async () => { failOld(new Error('stale local failure')); await pending; });

    expect(useAppStore.getState().settings.provider).toBe('openrouter');
    expect(useAppStore.getState().modelsError).toBeNull();
    expect(useAppStore.getState().modelsLoading).toBe(false);
  });
});
