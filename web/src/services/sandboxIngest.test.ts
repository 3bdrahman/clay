import { beforeEach, describe, expect, it, vi } from 'vitest';
import { addFiles, loadSampleData, type SandboxIngestDeps } from './sandboxIngest';
import { useAppStore, type SandboxDataset, type SandboxProcessing } from '../store';
import type { ClayServices } from '../hooks/useClay';

vi.mock('./sandboxTables', () => ({
  registerSandboxTable: vi.fn(),
  persistSandboxCsv: vi.fn(async () => {}),
}));

const { mockLoadSampleDatasets } = vi.hoisted(() => ({
  mockLoadSampleDatasets: vi.fn(),
}));

vi.mock('../services/datasets', async importOriginal => ({
  ...(await importOriginal<typeof import('../services/datasets')>()),
  loadSampleDatasets: mockLoadSampleDatasets,
}));

function makeDeps(services: ClayServices): SandboxIngestDeps {
  return {
    services,
    setSandboxProcessing: items => useAppStore.setState({ sandboxProcessing: items }),
    updateSandboxProcessingItem: (fileName, patch) => {
      useAppStore.setState(state => ({
        sandboxProcessing: state.sandboxProcessing.map(item =>
          item.fileName === fileName ? { ...item, ...patch } : item,
        ),
      }));
    },
    addSandboxDataset: (dataset: SandboxDataset) => {
      useAppStore.getState().addSandboxDataset(dataset);
      services.dispose();
    },
    addSandboxDocument: document => useAppStore.getState().addSandboxDocument(document),
    getSandboxProcessing: () => useAppStore.getState().sandboxProcessing,
    getSandboxDatasets: () => useAppStore.getState().sandboxDatasets,
    setSandboxDatasets: datasets => useAppStore.setState({ sandboxDatasets: datasets }),
  };
}

function makeServices(): ClayServices & { disposed: boolean } {
  const service = {
    disposed: false,
    llm: { invoke: vi.fn(), stream: vi.fn() },
    embeddings: {
      embed: vi.fn(async (input: string | string[]) => {
        if (service.disposed) {
          throw new Error('embedding client was disposed before document ingest finished');
        }
        const texts = Array.isArray(input) ? input : [input];
        return texts.map(() => new Array(8).fill(0.1));
      }),
    },
    vectorstore: {
      load: vi.fn(async () => {}),
      getSourceHashes: vi.fn(() => new Set<string>()),
      removeBySource: vi.fn(),
      addEntries: vi.fn(),
      similaritySearch: vi.fn(async () => []),
      clear: vi.fn(),
      listSources: vi.fn(() => []),
      stats: { entries: 0 },
      persistenceAvailable: true,
    },
    webSearch: { search: vi.fn(async () => []) },
    analyzer: { analyze: vi.fn(), listDatasets: vi.fn(() => []), getDatasetSummary: vi.fn(), dispose: vi.fn() },
    ready: true,
    dispose: vi.fn(() => {
      service.disposed = true;
    }),
  };
  return service as unknown as ClayServices & { disposed: boolean };
}

describe('sandboxIngest', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    localStorage.clear();
    useAppStore.getState().resetAll();
  });

  it('does not dispose the active service between CSV and document files in one batch', async () => {
    const services = makeServices();
    const csv = new File(['name,score\nAda,10'], 'scores.csv', { type: 'text/csv' });
    const text = 'Document content that should embed after a CSV in the same batch. '.repeat(20);
    const doc = new File([text], 'notes.txt', { type: 'text/plain' });

    await addFiles([csv, doc], makeDeps(services));

    expect(useAppStore.getState().sandboxDatasets.map(d => d.fileName)).toEqual(['scores.csv']);
    expect(useAppStore.getState().sandboxDocuments.map(d => d.fileName)).toEqual(['notes.txt']);
    expect(services.vectorstore.addEntries).toHaveBeenCalled();
    expect(services.dispose).toHaveBeenCalledTimes(1);
  });

  it('keeps error processing rows after completed rows are cleaned up', async () => {
    vi.useFakeTimers();
    const services = makeServices();
    const good = new File(['name,score\nAda,10'], 'scores.csv', { type: 'text/csv' });
    const bad = new File(['x'], 'bad.exe', { type: 'application/octet-stream' });

    await addFiles([good, bad], makeDeps(services));
    expect(useAppStore.getState().sandboxProcessing.map((p: SandboxProcessing) => p.status)).toContain('error');

    vi.advanceTimersByTime(3000);

    const processing = useAppStore.getState().sandboxProcessing;
    expect(processing).toHaveLength(1);
    expect(processing[0]).toMatchObject({
      fileName: 'bad.exe',
      status: 'error',
    });
  });

  it('commits successful sample datasets before surfacing a partial-load warning', async () => {
    const { SampleDatasetLoadError } = await import('../services/datasets');
    const aq = await import('arquero');
    const okTable = aq.fromCSV('name,score\nAda,10');
    const partialResult = {
      tables: new Map([['ok', okTable]]),
      metadata: { ok: { columns: ['name', 'score'], rowCount: 1 } },
      rawCsv: { ok: 'name,score\nAda,10' },
      arquero: aq,
    };
    mockLoadSampleDatasets.mockRejectedValue(
      new SampleDatasetLoadError(
        [{ name: 'missing', cause: new Error('HTTP 404 Not Found') }],
        ['ok'],
        partialResult,
      ),
    );

    await expect(loadSampleData(makeDeps(makeServices()))).rejects.toThrow(/missing/);

    const datasets = useAppStore.getState().sandboxDatasets;
    expect(datasets).toHaveLength(1);
    expect(datasets[0]).toMatchObject({
      name: 'ok',
      fileName: 'sample/ok.csv',
      isSample: true,
      rowCount: 1,
    });
  });

  it('preserves existing samples when every sample file fails', async () => {
    const { SampleDatasetLoadError } = await import('../services/datasets');
    useAppStore.getState().addSandboxDataset({
      name: 'existing',
      fileName: 'sample/existing.csv',
      columns: ['name'],
      rowCount: 1,
      loadedAt: Date.now(),
      csv: 'name\nGrace',
      isSample: true,
    });
    const aq = await import('arquero');
    mockLoadSampleDatasets.mockRejectedValue(
      new SampleDatasetLoadError(
        [{ name: 'missing', cause: new Error('HTTP 404 Not Found') }],
        [],
        { tables: new Map(), metadata: {}, rawCsv: {}, arquero: aq },
      ),
    );

    await expect(loadSampleData(makeDeps(makeServices()))).rejects.toThrow(/could not be loaded/i);

    expect(useAppStore.getState().sandboxDatasets.map(dataset => dataset.name)).toEqual(['existing']);
  });
});
