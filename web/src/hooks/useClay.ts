import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as aq from 'arquero';
import { createLLMClient, LLMConfigError, type LLMClient } from '../lib/llm';
import { createDemoLLMClient } from '../lib/demo-llm';
import { createEmbeddingsClient, type EmbeddingsClient } from '../lib/embeddings';
import { createWebSearchClient, type WebSearchClient } from '../lib/websearch';
import { createVectorStore, type VectorStore } from '../lib/vectorstore';
import { createDataAnalyzer, type DataAnalyzer, type DatasetMeta } from '../services/analyzer';
import { loadSampleDatasets } from '../services/datasets';
import { processFile, embedDocumentChunks, existingSourceHashes, type ProcessedFile } from '../services/files';
import {
  registerSandboxTable,
  unregisterSandboxTable,
  clearSandboxTables,
} from '../services/sandboxTables';
import { useAppStore, type SandboxDataset } from '../store';
import {
  listNimModels,
  listLocalCatalog,
  resolveModels,
  pickLocalModels,
  type ModelInfo,
  type PickedModels,
} from '../lib/models';
import { resolveProviderEndpoint } from '../lib/providers';

export interface ClayServices {
  llm: LLMClient;
  embeddings: EmbeddingsClient;
  vectorstore: VectorStore;
  webSearch: WebSearchClient;
  analyzer: DataAnalyzer;
  ready: boolean;
}

const MODEL_TTL_MS = 60 * 60 * 1000;
const LOCAL_CATALOG_TTL_MS = 60 * 60 * 1000;

export function useClay(): {
  services: ClayServices | null;
  loading: boolean;
  error: string | null;
  needsConfiguration: boolean;
  pickedModels: PickedModels;
  refreshModels: () => Promise<void>;
  addFiles: (files: FileList | File[]) => Promise<void>;
  loadSampleData: () => Promise<void>;
  clearSandboxData: () => void;
} {
  const settings = useAppStore(s => s.settings);
  const availableModels = useAppStore(s => s.availableModels);
  const modelsFetchedAt = useAppStore(s => s.modelsFetchedAt);
  const sandboxDatasets = useAppStore(s => s.sandboxDatasets);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const sandboxProcessing = useAppStore(s => s.sandboxProcessing);
  const setModels = useAppStore(s => s.setModels);
  const setModelsLoading = useAppStore(s => s.setModelsLoading);
  const setModelsError = useAppStore(s => s.setModelsError);
  const setLocalCatalog = useAppStore(s => s.setLocalCatalog);
  const addSandboxDataset = useAppStore(s => s.addSandboxDataset);
  const addSandboxDocument = useAppStore(s => s.addSandboxDocument);
  const setSandboxProcessing = useAppStore(s => s.setSandboxProcessing);
  const updateSandboxProcessingItem = useAppStore(s => s.updateSandboxProcessingItem);
  const clearSandbox = useAppStore(s => s.clearSandbox);

  const [services, setServices] = useState<ClayServices | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [needsConfiguration, setNeedsConfiguration] = useState(false);
  const servicesRef = useRef<ClayServices | null>(null);

  const fetchedKeyRef = useRef<string | null>(null);

  const fetchNimModels = useCallback(
    async (key: string, force = false): Promise<ModelInfo[]> => {
      if (
        !force &&
        fetchedKeyRef.current === key &&
        Date.now() - modelsFetchedAt < MODEL_TTL_MS &&
        availableModels.length > 0
      ) {
        return availableModels;
      }
      setModelsLoading(true);
      setModelsError(null);
      try {
        const models = await listNimModels(key);
        setModels(models);
        fetchedKeyRef.current = key;
        return models;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setModelsError(msg);
        return [];
      } finally {
        setModelsLoading(false);
      }
    },
    [availableModels, modelsFetchedAt, setModels, setModelsLoading, setModelsError],
  );

  const fetchLocalModels = useCallback(
    async (baseUrl: string, force = false): Promise<ModelInfo[]> => {
      const stamp = useAppStore.getState().settings.localCatalogFetchedAt;
      const existing = useAppStore.getState().settings.localCatalog;
      if (!force && existing.length > 0 && Date.now() - stamp < LOCAL_CATALOG_TTL_MS) {
        return existing;
      }
      setModelsLoading(true);
      setModelsError(null);
      try {
        const models = await listLocalCatalog(baseUrl, '');
        setLocalCatalog(models);
        return models;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setModelsError(msg);
        return [];
      } finally {
        setModelsLoading(false);
      }
    },
    [setLocalCatalog, setModelsLoading, setModelsError],
  );

  const refreshModels = useCallback(async () => {
    if (settings.provider === 'local') {
      const url = settings.localServerUrl.trim();
      if (url) await fetchLocalModels(url, true);
    } else if (settings.apiKey) {
      await fetchNimModels(settings.apiKey, true);
    }
  }, [settings.provider, settings.apiKey, settings.localServerUrl, fetchNimModels, fetchLocalModels]);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        setLoading(true);
        setError(null);
        setNeedsConfiguration(false);

        const endpoint = resolveProviderEndpoint(settings);
        
        // Determine if we should use demo mode (no API key and not local)
        const isDemoMode = settings.provider !== 'local' && !settings.apiKey;
        
        const llm = isDemoMode
          ? createDemoLLMClient()
          : createLLMClient({
              baseUrl: endpoint.baseUrl,
              apiKey: endpoint.apiKey,
              temperature: settings.temperature,
              providerLabel: endpoint.providerLabel,
            });

        let catalog = availableModels;
        if (settings.provider === 'local') {
          const url = settings.localServerUrl.trim();
          if (url) {
            const local = await fetchLocalModels(url);
            if (local.length > 0) catalog = local;
          }
        } else if (settings.apiKey) {
          const fresh = await fetchNimModels(settings.apiKey);
          if (fresh.length > 0) catalog = fresh;
        }
        const { picked } = resolveModels(
          { ...settings, localCatalog: catalog },
          catalog,
        );

        const embeddingKey =
          settings.provider === 'local'
            ? ''
            : (settings.embeddingApiKey || settings.apiKey);
        const embeddings = createEmbeddingsClient({
          baseUrl: endpoint.baseUrl,
          apiKey: embeddingKey,
          embeddingModel: picked.embedding ?? '',
          providerLabel: endpoint.providerLabel,
        });
        const vectorstore = createVectorStore(embeddings);
        const webSearch = createWebSearchClient(settings);

        const tables = new Map<string, unknown>([['aq', aq]]);
        const metadata: DatasetMeta = {};

        for (const d of sandboxDatasets) {
          let table: unknown;
          if (d.csv !== undefined) {
            table = aq.fromCSV(d.csv);
          } else {
            table = aq.from(
              d.columns.map(() => ({})),
            );
          }
          tables.set(d.name, table);
          metadata[d.name] = { columns: d.columns, rowCount: d.rowCount };
        }

        const analyzer = createDataAnalyzer({
          llm,
          embeddings,
          datasets: tables,
          metadata,
          codeGenModel: picked.codeGen,
        });

        vectorstore.load().catch(() => {});

        if (cancelled) return;

        const newServices: ClayServices = {
          llm,
          embeddings,
          vectorstore,
          webSearch,
          analyzer,
          ready: true,
        };
        servicesRef.current = newServices;
        setServices(newServices);
        
        if (isDemoMode) {
          setNeedsConfiguration(true);
        }
      } catch (e) {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          setError(msg);
          if (e instanceof LLMConfigError) setNeedsConfiguration(true);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, [settings, availableModels, sandboxDatasets, fetchNimModels, fetchLocalModels]);

  const addFiles = useCallback(
    async (files: FileList | File[]) => {
      const arr = Array.from(files);
      if (arr.length === 0) return;
      const sv = servicesRef.current;
      if (!sv) {
        throw new Error('Services not ready — add an API key first');
      }

      const initial: typeof sandboxProcessing = arr.map(f => ({
        fileName: f.name,
        status: 'processing',
      }));
      setSandboxProcessing([...sandboxProcessing, ...initial]);

      for (const file of arr) {
        try {
          updateSandboxProcessingItem(file.name, { status: 'processing' });
          const processed: ProcessedFile = await processFile(file);
          if (processed.error) {
            updateSandboxProcessingItem(file.name, { status: 'error', error: processed.error });
            continue;
          }
          if (processed.dataset) {
            const csv = await file.text();
            registerSandboxTable(processed.dataset.name, processed.dataset.table);
            addSandboxDataset({
              name: processed.dataset.name,
              fileName: file.name,
              columns: processed.dataset.columns,
              rowCount: processed.dataset.rowCount,
              loadedAt: Date.now(),
              csv,
              isSample: false,
            });
            updateSandboxProcessingItem(file.name, { status: 'done' });
            continue;
          }
          if (processed.document) {
            updateSandboxProcessingItem(file.name, { status: 'embedding' });
            const hashes = await existingSourceHashes(sv.vectorstore, processed.document.source);
            if (hashes.has(processed.document.sourceHash)) {
              updateSandboxProcessingItem(file.name, { status: 'done' });
              addSandboxDocument({
                id: processed.document.source,
                fileName: file.name,
                source: processed.document.source,
                chunkCount: processed.document.chunks.length,
                loadedAt: Date.now(),
                chunks: processed.document.chunks,
              });
              continue;
            }
            const embedded = await embedDocumentChunks(processed.document, sv.embeddings);
            sv.vectorstore.addEntries(embedded.map(e => ({
              id: e.id,
              text: e.text,
              source: e.source,
              page: e.page,
              embedding: e.embedding,
            })));
            addSandboxDocument({
              id: processed.document.source,
              fileName: file.name,
              source: processed.document.source,
              chunkCount: processed.document.chunks.length,
              loadedAt: Date.now(),
              chunks: processed.document.chunks,
            });
            updateSandboxProcessingItem(file.name, { status: 'done' });
            continue;
          }
          updateSandboxProcessingItem(file.name, { status: 'error', error: 'No content extracted' });
        } catch (e) {
          updateSandboxProcessingItem(file.name, {
            status: 'error',
            error: e instanceof Error ? e.message : String(e),
          });
        }
      }

      setTimeout(() => {
        useAppStore.setState(state => ({
          sandboxProcessing: state.sandboxProcessing.filter(
            p => p.status === 'processing' || p.status === 'embedding',
          ),
        }));
      }, 3000);
    },
    [
      sandboxProcessing,
      setSandboxProcessing,
      updateSandboxProcessingItem,
      addSandboxDataset,
      addSandboxDocument,
    ],
  );

  const loadSampleData = useCallback(async () => {
    const sample = await loadSampleDatasets();
    sample.tables.forEach((table, name) => {
      if (name === 'aq') return;
      registerSandboxTable(name, table as Parameters<typeof registerSandboxTable>[1]);
    });
    const newDatasets: SandboxDataset[] = [];
    sample.tables.forEach((table, name) => {
      if (name === 'aq') return;
      const t = table as { columnNames?: () => string[]; numRows?: () => number };
      const columns = typeof t.columnNames === 'function' ? t.columnNames() : [];
      const rowCount = typeof t.numRows === 'function' ? t.numRows() : 0;
      const originalCsv = sample.rawCsv[name];
      newDatasets.push({
        name,
        fileName: `sample/${name}.csv`,
        columns,
        rowCount,
        loadedAt: Date.now(),
        csv: originalCsv,
        isSample: true,
      });
    });
    useAppStore.setState(() => ({
      sandboxDatasets: newDatasets,
    }));
  }, []);

  const clearSandboxData = useCallback(() => {
    const sv = servicesRef.current;
    if (sv) {
      sv.vectorstore.clear();
    }
    sandboxDatasets.forEach(d => unregisterSandboxTable(d.name));
    clearSandboxTables();
    clearSandbox();
  }, [clearSandbox, sandboxDatasets]);

  const pickedModels: PickedModels = useMemo(() => {
    if (settings.provider === 'local') {
      return pickLocalModels(settings.localModels);
    }
    return resolveModels(settings, availableModels).picked;
  }, [settings, availableModels]);

  return {
    services,
    loading,
    error,
    needsConfiguration,
    pickedModels,
    refreshModels,
    addFiles,
    loadSampleData,
    clearSandboxData,
  };
}
