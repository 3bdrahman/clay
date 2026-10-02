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
  it('discovers NIM models at the same configured gateway used by chat', async () => {
    await configure({ provider: 'nim', nimApiKey: 'nvapi-test', nimBaseUrl: 'https://relay.example/v1' });
    vi.mocked(listModels).mockResolvedValue([{ id: 'nim-chat', ownedBy: 'nvidia', created: 0 }]);
    await act(async () => { await fetchModels('nvapi-test'); });
    expect(listModels).toHaveBeenCalledWith('nim', 'nvapi-test', 'https://relay.example/v1');
    expect(useAppStore.getState().availableModels[0].id).toBe('nim-chat');
  });

  it('does not call an upstream when the production relay is missing', async () => {
    vi.stubEnv('DEV', false);
    vi.stubEnv('VITE_NIM_BASE_URL', '');
    await configure({ provider: 'nim', nimApiKey: 'nvapi-test' });
    await act(async () => { await fetchModels('nvapi-test'); });
    expect(listModels).not.toHaveBeenCalled();
    expect(useAppStore.getState().modelsError).toMatch(/relay/i);
  });

  it('cannot reuse a cached catalog across providers even when key strings match', async () => {
    vi.mocked(listModels).mockResolvedValue([{ id: 'first-model', ownedBy: 'test', created: 0 }]);
    await configure({ openrouterApiKey: 'same-key' });
    await act(async () => { await fetchModels('same-key'); });
    await configure({ provider: 'nim', nimApiKey: 'same-key', nimBaseUrl: 'https://relay.example/v1' });
    await act(async () => { await fetchModels('same-key'); });
    expect(listModels).toHaveBeenCalledTimes(2);
    expect(listModels).toHaveBeenLastCalledWith('nim', 'same-key', 'https://relay.example/v1');
  });

  it('discards an old provider response that finishes after a switch', async () => {
    let finishOld!: (models: ModelInfo[]) => void;
    vi.mocked(listModels).mockImplementationOnce(() => new Promise(resolve => { finishOld = resolve; }));
    await configure({ openrouterApiKey: 'old-key' });
    let pending!: Promise<ModelInfo[]>;
    await act(async () => { pending = fetchModels('old-key'); });
    await configure({ provider: 'nim', nimApiKey: 'new-key', nimBaseUrl: 'https://relay.example/v1' });
    const nimCatalog = [{ id: 'nim-current', ownedBy: 'nvidia', created: 0 }];
    vi.mocked(listModels).mockResolvedValueOnce(nimCatalog);
    await act(async () => { await fetchModels('new-key'); });
    await act(async () => { finishOld([{ id: 'old-model', ownedBy: 'other', created: 0 }]); await pending; });
    expect(useAppStore.getState().availableModels).toEqual(nimCatalog);
    expect(useAppStore.getState().modelsError).toBeNull();
    expect(useAppStore.getState().modelsLoading).toBe(false);
  });
});
