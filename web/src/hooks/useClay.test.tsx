import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import type { ReactNode } from 'react';
import { useClay, type ClayServices } from './useClay';
import { useAppStore } from '../store';

const originalFetch = globalThis.fetch;

;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type HookValue = ReturnType<typeof useClay>;
type HookResult = { current: HookValue };

function renderHook(hook: () => HookValue): { result: HookResult; unmount: () => void } {
  const result: HookResult = { current: {} as HookValue };
  let root: ReturnType<typeof createRoot> | null = null;

  function Probe() {
    result.current = hook();
    return null as ReactNode;
  }

  act(() => {
    const container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    root.render(<Probe />);
  });

  return {
    result,
    unmount: () => {
      act(() => root?.unmount());
    },
  };
}

async function flush(n = 3) {
  for (let i = 0; i < n; i++) {
    await act(async () => {
      await new Promise<void>(r => setTimeout(r, 10));
    });
  }
}

const baseSettings = {
  provider: 'nim' as const,
  apiKey: '',
  embeddingApiKey: '',
  webSearchProvider: 'duckduckgo' as const,
  serperApiKey: '',
  temperature: 0,
  maxRetries: 3,
  theme: 'system' as const,
  localServerUrl: 'http://localhost:11434/v1',
  localModels: {
    routing: '',
    codeGen: '',
    answer: '',
    eval: '',
    embedding: '',
  },
  localCatalog: [],
  localCatalogFetchedAt: 0,
};

function jsonResponseLike<T>(body: T, ok = true): unknown {
  return {
    ok,
    json: async () => body,
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body)),
  };
}

describe('useClay', () => {
  let unmountRef: (() => void) | null = null;

  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
    useAppStore.getState().resetAll();
    useAppStore.setState({ settings: { ...baseSettings } as never });
  });

  afterEach(() => {
    if (unmountRef) {
      unmountRef();
      unmountRef = null;
    }
    globalThis.fetch = originalFetch;
  });

  function render(): HookResult {
    const { result, unmount } = renderHook(() => useClay());
    unmountRef = unmount;
    return result;
  }

  it('initializes in demo mode when no API key is set', async () => {
    const r = render();
    await flush(6);
    expect(r.current.services?.ready).toBe(true);
    expect(r.current.needsConfiguration).toBe(true);
    expect(r.current.loading).toBe(false);
    expect(r.current.error).toBeNull();
  });

  it('exposes pickedModels in demo mode (empty picks)', async () => {
    const r = render();
    await flush(6);
    expect(r.current.pickedModels).toBeDefined();
    expect(r.current.pickedModels.answer).toBeUndefined();
  });

  it('fetches NIM catalog when apiKey is set', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponseLike({
      data: [
        { id: 'mistral-small-24b-instruct', created: 1, owned_by: 'mistralai' },
        { id: 'meta/llama-3.1-405b-instruct', created: 2, owned_by: 'meta' },
        { id: 'intfloat/e5-large-v2', created: 3, owned_by: 'intfloat' },
      ],
    }));
    globalThis.fetch = fetchMock as never;

    useAppStore.setState({
      settings: { ...baseSettings, apiKey: 'nvapi-test' } as never,
    });

    const r = render();
    await flush(10);

    expect(r.current.services?.ready).toBe(true);
    expect(useAppStore.getState().availableModels.length).toBe(3);
    expect(useAppStore.getState().modelsLoading).toBe(false);
    expect(r.current.pickedModels.answer).toBeTruthy();
  });

  it('handles NIM fetch failure gracefully (initializes, sets modelsError)', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false, status: 401, statusText: 'Unauthorized' }) as never;

    useAppStore.setState({
      settings: { ...baseSettings, apiKey: 'bad' } as never,
    });

    const r = render();
    await flush(8);

    expect(r.current.services?.ready).toBe(true);
    expect(useAppStore.getState().modelsError).toBeTruthy();
  });

  it('loadSampleData populates sandboxDatasets with sample tables', async () => {
    globalThis.fetch = ((url: string | URL | Request): Promise<unknown> => {
      const u = String(url);
      if (u.endsWith('/data/datasets/index.json')) {
        return Promise.resolve(jsonResponseLike({ files: ['employees.csv', 'projects.csv'] }));
      }
      if (u.endsWith('employees.csv')) {
        return Promise.resolve(jsonResponseLike('name,age\nAlice,30\nBob,25'));
      }
      if (u.endsWith('projects.csv')) {
        return Promise.resolve(jsonResponseLike('id,budget\n1,100\n2,200'));
      }
      return Promise.reject(new Error('unexpected fetch: ' + u));
    }) as never;

    const r = render();
    await flush(6);

    await act(async () => {
      await r.current.loadSampleData();
    });
    await flush(2);

    const ds = useAppStore.getState().sandboxDatasets;
    expect(ds.length).toBe(2);
    expect(ds.every(d => d.isSample === true)).toBe(true);
    expect(ds.every(d => d.csv !== undefined)).toBe(true);
  });

  it('clearSandboxData removes all sandbox state', async () => {
    globalThis.fetch = ((url: string | URL | Request): Promise<unknown> => {
      const u = String(url);
      if (u.endsWith('/data/datasets/index.json')) {
        return Promise.resolve(jsonResponseLike({ files: ['employees.csv'] }));
      }
      if (u.endsWith('employees.csv')) {
        return Promise.resolve(jsonResponseLike('name,age\nAlice,30'));
      }
      return Promise.reject(new Error('unexpected fetch: ' + u));
    }) as never;

    const r = render();
    await flush(6);

    await act(async () => {
      await r.current.loadSampleData();
    });
    await flush(2);
    expect(useAppStore.getState().sandboxDatasets.length).toBe(1);

    act(() => {
      r.current.clearSandboxData();
    });
    expect(useAppStore.getState().sandboxDatasets).toEqual([]);
    expect(useAppStore.getState().sandboxDocuments).toEqual([]);
    expect(useAppStore.getState().sandboxProcessing).toEqual([]);
  });

  it('addFiles processes a CSV File into a sandbox dataset', async () => {
    const r = render();
    await flush(6);
    expect(r.current.services?.ready).toBe(true);

    const csv = 'name,age\nAlice,30\nBob,25';
    const file = new File([csv], 'people.csv', { type: 'text/csv' });

    await act(async () => {
      await r.current.addFiles([file]);
    });
    await flush(3);

    const ds = useAppStore.getState().sandboxDatasets;
    expect(ds.some(d => d.name === 'people')).toBe(true);
    const people = ds.find(d => d.name === 'people');
    expect(people?.columns).toEqual(['name', 'age']);
    expect(people?.rowCount).toBe(2);
  });

  it('addFiles rejects an unsupported file type with an error status', async () => {
    const r = render();
    await flush(6);

    const file = new File(['x'], 'unknown.xyz', { type: 'application/octet-stream' });

    await act(async () => {
      await r.current.addFiles([file]);
    });
    await flush(3);

    const item = useAppStore.getState().sandboxProcessing.find(p => p.fileName === 'unknown.xyz');
    expect(item?.status).toBe('error');
    expect(item?.error).toMatch(/unsupported/i);
  });

  it('addFiles rejects files larger than 25MB', async () => {
    const r = render();
    await flush(6);

    const big = new File([new Uint8Array(0)], 'big.csv', { type: 'text/csv' });
    Object.defineProperty(big, 'size', { value: 26 * 1024 * 1024 });

    await act(async () => {
      await r.current.addFiles([big]);
    });
    await flush(3);

    const item = useAppStore.getState().sandboxProcessing.find(p => p.fileName === 'big.csv');
    expect(item?.status).toBe('error');
    expect(item?.error).toMatch(/too large/i);
  });

  it('addFiles processes a text File into a vectorstore document', async () => {
    const r = render();
    await flush(6);
    const sv = r.current.services as ClayServices | undefined;
    expect(sv?.ready).toBe(true);

    (sv!.embeddings.embed as unknown) = vi.fn(async (input: string | string[]) => {
      const arr = Array.isArray(input) ? input : [input];
      return arr.map(() => new Array(8).fill(0.1));
    });

    const text =
      'This is the first chunk of a text file meant to exceed the chunk size threshold. '.repeat(20) +
      'Second chunk here with a bit more content, again repeated to ensure multiple chunks form. '.repeat(20);
    const file = new File([text], 'notes.txt', { type: 'text/plain' });

    await act(async () => {
      await r.current.addFiles([file]);
    });
    await flush(3);

    const docs = useAppStore.getState().sandboxDocuments;
    const doc = docs.find(d => d.fileName === 'notes.txt');
    expect(doc).toBeDefined();
    expect(doc?.chunkCount).toBeGreaterThan(0);
    expect(sv!.vectorstore.stats.entries).toBeGreaterThan(0);
  });

  it('refreshModels with NIM API key triggers a models fetch', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponseLike({
      data: [{ id: 'mistral-small', created: 1, owned_by: 'mistralai' }],
    }));
    globalThis.fetch = fetchMock as never;

    useAppStore.setState({
      settings: { ...baseSettings, apiKey: 'nvapi-test' } as never,
    });
    const r = render();
    await flush(8);

    fetchMock.mockClear();
    await act(async () => {
      await r.current.refreshModels();
    });
    await flush(3);

    expect(fetchMock).toHaveBeenCalled();
    expect(useAppStore.getState().availableModels.length).toBe(1);
  });

  it('refreshModels with local provider triggers a local catalog fetch', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponseLike({
      data: [{ id: 'llama3:8b', created: 1, owned_by: 'ollama' }],
    }));
    globalThis.fetch = fetchMock as never;

    useAppStore.setState({
      settings: {
        ...baseSettings,
        provider: 'local',
        localServerUrl: 'http://localhost:11434/v1',
        localModels: {
          routing: 'llama3:8b',
          codeGen: '',
          answer: 'llama3:8b',
          eval: '',
          embedding: '',
        },
      } as never,
    });

    const r = render();
    await flush(6);

    fetchMock.mockClear();
    await act(async () => {
      await r.current.refreshModels();
    });
    await flush(3);

    expect(fetchMock).toHaveBeenCalled();
    expect(useAppStore.getState().settings.localCatalog.length).toBe(1);
  });

  it('pickedModels in local mode returns localModels picks (empty trimmed to undefined)', async () => {
    useAppStore.setState({
      settings: {
        ...baseSettings,
        provider: 'local',
        localServerUrl: 'http://localhost:11434/v1',
        localModels: {
          routing: 'm1',
          codeGen: '',
          answer: 'm2',
          eval: '   ',
          embedding: 'm3',
        },
      } as never,
    });
    const r = render();
    await flush(6);

    expect(r.current.pickedModels).toEqual({
      routing: 'm1',
      codeGen: undefined,
      answer: 'm2',
      eval: undefined,
      embedding: 'm3',
    });
  });
});
