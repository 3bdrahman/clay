import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { type LLMClient, ProviderUnreachableError } from '../lib/llm';
import type { EmbeddingsClient } from '../lib/embeddings';
import type { WebSearchClient } from '../lib/websearch';
import type { VectorStore } from '../lib/vectorstore';
import type { DataAnalyzer } from '../services/analyzer';
import {
  unregisterSandboxTable,
  clearSandboxTables,
  rehydrateSandboxTables,
  loadPersistedSandboxCsvs,
  deletePersistedSandboxCsv,
  clearPersistedSandboxCsvs,
} from '../services/sandboxTables';
import { createClayServiceBundle } from '../services/clayServices';
import { useAppStore, type SandboxDataset } from '../store';
import { pickLocalModels, resolveModels, type PickedModels } from '../lib/models';
import { resolveProviderEndpoint } from '../lib/providers';
import {
  useFetchNimModels,
  useFetchLocalModels,
  useRefreshModels,
  type ModelCatalogDeps,
} from './useModelCatalog';
import { addFiles, loadSampleData, type SandboxIngestDeps } from '../services/sandboxIngest';

export interface ClayServices {
  llm: LLMClient;
  embeddings: EmbeddingsClient;
  vectorstore: VectorStore;
  webSearch: WebSearchClient;
  analyzer: DataAnalyzer;
  ready: boolean;
  dispose: () => void;
}

export function useClay(): {
  services: ClayServices | null;
  loading: boolean;
  error: string | null;
  needsConfiguration: boolean;
  persistenceAvailable: boolean;
  pickedModels: PickedModels;
  refreshModels: () => Promise<void>;
  addFiles: (files: FileList | File[]) => Promise<void>;
  loadSampleData: () => Promise<void>;
  clearSandboxData: () => void;
  removeSandboxDocument: (fileName: string) => void;
  removeSandboxDataset: (name: string) => void;
} {
  const settings = useAppStore(s => s.settings);
  const availableModels = useAppStore(s => s.availableModels);
  const modelsFetchedAt = useAppStore(s => s.modelsFetchedAt);
  const sandboxDatasets = useAppStore(s => s.sandboxDatasets);
  const sandboxDocuments = useAppStore(s => s.sandboxDocuments);
  const setModels = useAppStore(s => s.setModels);
  const setModelsLoading = useAppStore(s => s.setModelsLoading);
  const setModelsError = useAppStore(s => s.setModelsError);
  const setLocalCatalog = useAppStore(s => s.setLocalCatalog);
  const addSandboxDataset = useAppStore(s => s.addSandboxDataset);
  const addSandboxDocument = useAppStore(s => s.addSandboxDocument);
  const setSandboxProcessing = useAppStore(s => s.setSandboxProcessing);
  const updateSandboxProcessingItem = useAppStore(s => s.updateSandboxProcessingItem);
  const clearSandbox = useAppStore(s => s.clearSandbox);
  const removeSandboxDatasetFromStore = useAppStore(s => s.removeSandboxDataset);
  const removeSandboxDocumentFromStore = useAppStore(s => s.removeSandboxDocument);

  const [services, setServices] = useState<ClayServices | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [needsConfiguration, setNeedsConfiguration] = useState(false);
  const [persistenceAvailable, setPersistenceAvailable] = useState(true);
  const servicesRef = useRef<ClayServices | null>(null);

  const modelCatalogDeps: ModelCatalogDeps = {
    settings,
    availableModels,
    modelsFetchedAt,
    setModels,
    setModelsLoading,
    setModelsError,
    setLocalCatalog,
  };

  const fetchNimModels = useFetchNimModels(modelCatalogDeps);
  const fetchLocalModels = useFetchLocalModels(modelCatalogDeps);
  const refreshModels = useRefreshModels(modelCatalogDeps, fetchNimModels, fetchLocalModels);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        setLoading(true);
        setError(null);
        setNeedsConfiguration(false);

        const endpoint = resolveProviderEndpoint(settings);
        const isLocal = settings.provider === 'local';
        const hasValidConfig = isLocal
          ? endpoint.baseUrl.length > 0
          : endpoint.apiKey.trim().length > 0;

        if (!hasValidConfig) {
          setNeedsConfiguration(true);
          setServices(null);
          setLoading(false);
          return;
        }

        let catalog = availableModels;
        if (settings.provider === 'local') {
          const url = settings.localServerUrl.trim();
          if (url) {
            const local = await fetchLocalModels(url);
            if (local.length > 0) catalog = local;
          }
        } else if (endpoint.apiKey) {
          const fresh = await fetchNimModels(endpoint.apiKey);
          if (fresh.length > 0) catalog = fresh;
        }

        const persistedCsvs = await loadPersistedSandboxCsvs();
        const hydratedDatasets = sandboxDatasets.map(d =>
          d.csv === undefined && persistedCsvs.has(d.name)
            ? { ...d, csv: persistedCsvs.get(d.name) }
            : d,
        );
        const { tables, metadata } = rehydrateSandboxTables(hydratedDatasets);

        const bundle = createClayServiceBundle({
          settings,
          catalog,
          analyzerTables: tables,
          analyzerMetadata: metadata,
        });

        bundle.vectorstore
          .load()
          .then(() => {
            if (cancelled) return;
            if (!bundle.vectorstore.persistenceAvailable) setPersistenceAvailable(false);
          })
          .catch((e: unknown) => {
            if (cancelled) return;
            setPersistenceAvailable(false);
            const msg = e instanceof Error ? e.message : String(e);
            if (import.meta.env.DEV) {
              console.warn('[useClay] vectorstore persistence unavailable:', msg);
            }
          });

        if (cancelled) return;

        const newServices: ClayServices = {
          llm: bundle.llm,
          embeddings: bundle.embeddings,
          vectorstore: bundle.vectorstore,
          webSearch: bundle.webSearch,
          analyzer: bundle.analyzer,
          ready: true,
          dispose: bundle.dispose,
        };
        servicesRef.current = newServices;
        setServices(newServices);
      } catch (e) {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          setError(msg);
          if (e instanceof ProviderUnreachableError) setNeedsConfiguration(true);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    init();
    return () => {
      cancelled = true;
      // The bundle is replaced on every settings/catalog/dataset change and
      // on unmount. Without this teardown, every recreation leaks the
      // embeddings worker and (after the first analysis) the sandbox worker.
      servicesRef.current?.dispose();
    };
  }, [settings, availableModels, sandboxDatasets, fetchNimModels, fetchLocalModels]);

  const sandboxIngestDeps = useMemo<SandboxIngestDeps>(() => ({
    services,
    setSandboxProcessing,
    updateSandboxProcessingItem,
    addSandboxDataset,
    addSandboxDocument,
    getSandboxProcessing: () => useAppStore.getState().sandboxProcessing,
    getSandboxDatasets: () => useAppStore.getState().sandboxDatasets,
    setSandboxDatasets: (datasets: SandboxDataset[]) => useAppStore.setState({ sandboxDatasets: datasets }),
  }), [services, setSandboxProcessing, updateSandboxProcessingItem, addSandboxDataset, addSandboxDocument]);

  const addFilesCallback = useCallback(
    async (files: FileList | File[]) => {
      await addFiles(files, sandboxIngestDeps);
    },
    [sandboxIngestDeps],
  );

  const loadSampleDataCallback = useCallback(
    async () => {
      await loadSampleData(sandboxIngestDeps);
    },
    [sandboxIngestDeps],
  );

  const clearSandboxData = useCallback(() => {
    const sv = servicesRef.current;
    if (sv) {
      sv.vectorstore.clear();
    }
    const currentDatasets = useAppStore.getState().sandboxDatasets;
    currentDatasets.forEach(d => unregisterSandboxTable(d.name));
    clearSandboxTables();
    clearSandbox();
    void clearPersistedSandboxCsvs();
  }, [clearSandbox]);

  const removeSandboxDocument = useCallback(
    (fileName: string) => {
      const sv = servicesRef.current;
      const doc = sandboxDocuments.find(d => d.fileName === fileName);
      if (doc) {
        if (sv) {
          sv.vectorstore.removeBySource(doc.source);
        }
        removeSandboxDocumentFromStore(fileName);
      }
    },
    [sandboxDocuments, removeSandboxDocumentFromStore],
  );

  const removeSandboxDataset = useCallback(
    (name: string) => {
      unregisterSandboxTable(name);
      removeSandboxDatasetFromStore(name);
      void deletePersistedSandboxCsv(name);
    },
    [removeSandboxDatasetFromStore],
  );

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
    persistenceAvailable,
    pickedModels,
    refreshModels,
    addFiles: addFilesCallback,
    loadSampleData: loadSampleDataCallback,
    clearSandboxData,
    removeSandboxDocument,
    removeSandboxDataset,
  };
}