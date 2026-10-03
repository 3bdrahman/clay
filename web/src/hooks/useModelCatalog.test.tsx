import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useFetchCloudModels } from './useModelCatalog';
import { useAppStore } from '../store';
import { listModels } from '../lib/models';
import type { ModelInfo, Settings } from '../lib/types';

vi.mock('../lib/models', () => ({ listModels: vi.fn() }));

let fetchModels: ReturnType<typeof useFetchCloudModels>;
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
