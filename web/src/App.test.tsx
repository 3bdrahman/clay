import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import App from './App';
import type { UseClayResult } from './hooks/useClay';
import type { PickedModels } from './lib/models';

const clayState = vi.hoisted(() => ({
  current: null as UseClayResult | null,
  chatPanelClay: null as UseClayResult | null,
  dataSandboxAddFiles: null as UseClayResult['addFiles'] | null,
}));

vi.mock('./hooks/useClay', () => ({
  useClay: () => {
    if (!clayState.current) {
      throw new Error('useClay test fixture was not initialized');
    }
    return clayState.current;
  },
}));

vi.mock('./components/ChatPanel', () => ({
  ChatPanel: ({ clay }: { clay: UseClayResult }) => {
    clayState.chatPanelClay = clay;
    return <div data-testid="chat-panel" />;
  },
}));

vi.mock('./components/LazyPanels', () => ({
  SettingsPanelSuspense: () => null,
  DataSandboxSuspense: ({ addFiles }: { addFiles: UseClayResult['addFiles'] }) => {
    clayState.dataSandboxAddFiles = addFiles;
    return null;
  },
}));

function makeClayState(): UseClayResult {
  const vectorstore = {
    entries: 0,
    addFromUpload() {
      this.entries += 1;
    },
  };
  const addFiles = vi.fn(async () => {
    vectorstore.addFromUpload();
  });

  return {
    services: {
      llm: { invoke: vi.fn(), stream: vi.fn() },
      embeddings: { embed: vi.fn() },
      vectorstore: vectorstore as never,
      webSearch: { search: vi.fn() },
      analyzer: { analyze: vi.fn(), listDatasets: vi.fn(), getDatasetSummary: vi.fn(), dispose: vi.fn() },
      ready: true,
      dispose: vi.fn(),
    },
    loading: false,
    error: null,
    needsConfiguration: false,
    persistenceAvailable: true,
    pickedModels: { chat: 'demo-model' } as PickedModels,
    refreshModels: vi.fn(async () => {}),
    addFiles,
    loadSampleData: vi.fn(async () => {}),
    clearSandboxData: vi.fn(),
    removeSandboxDocument: vi.fn(),
    removeSandboxDataset: vi.fn(),
  };
}

describe('App service ownership', () => {
  let unmount: (() => void) | null = null;

  beforeEach(() => {
    clayState.current = makeClayState();
    clayState.chatPanelClay = null;
    clayState.dataSandboxAddFiles = null;
  });

  afterEach(() => {
    unmount?.();
    unmount = null;
  });

  it('passes one useClay result to chat and data actions so uploads are immediately visible to chat', async () => {
    await act(async () => {
      const container = document.createElement('div');
      document.body.appendChild(container);
      const root = createRoot(container);
      root.render(<App />);
      unmount = () => {
        act(() => root.unmount());
        container.remove();
      };
    });

    expect(clayState.chatPanelClay).toBe(clayState.current);
    expect(clayState.dataSandboxAddFiles).toBe(clayState.current?.addFiles);

    await act(async () => {
      const file = new File(['notes'], 'notes.txt', { type: 'text/plain' });
      await clayState.dataSandboxAddFiles?.([file]);
    });

    expect(clayState.chatPanelClay?.services?.vectorstore).toBe(clayState.current?.services?.vectorstore);
    expect((clayState.chatPanelClay?.services?.vectorstore as { entries: number } | undefined)?.entries).toBe(1);
  });

  it('shows a reload data-loss warning when persistence is unavailable', async () => {
    clayState.current = {
      ...makeClayState(),
      persistenceAvailable: false,
    };

    await act(async () => {
      const container = document.createElement('div');
      document.body.appendChild(container);
      const root = createRoot(container);
      root.render(<App />);
      unmount = () => {
        act(() => root.unmount());
        container.remove();
      };
    });

    expect(document.querySelector('[role="status"]')?.textContent).toContain('Data may be lost on reload.');
  });
});
