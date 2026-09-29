import { useCallback, useRef } from 'react';
import { listModels, listLocalCatalog, type ModelInfo } from '../lib/models';
import { resolveProviderEndpoint } from '../lib/providers';
import type { Settings } from '../lib/types';

const MODEL_TTL_MS = 60 * 60 * 1000;
const LOCAL_CATALOG_TTL_MS = 60 * 60 * 1000;

export interface ModelCatalogDeps {
  settings: Settings;
  availableModels: ModelInfo[];
  modelsFetchedAt: number;
  setModels: (models: ModelInfo[]) => void;
  setModelsLoading: (loading: boolean) => void;
  setModelsError: (error: string | null) => void;
  setLocalCatalog: (models: ModelInfo[]) => void;
}

export function useFetchNimModels(deps: ModelCatalogDeps) {
  const fetchedKeyRef = useRef<string | null>(null);
  const { settings, availableModels, modelsFetchedAt, setModels, setModelsLoading, setModelsError } = deps;

  return useCallback(
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
        const models = await listModels(settings.provider, key);
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
    [availableModels, modelsFetchedAt, setModels, setModelsLoading, setModelsError, settings.provider],
  );
}

export function useFetchLocalModels(deps: ModelCatalogDeps) {
  const { settings, setLocalCatalog, setModelsLoading, setModelsError } = deps;

  return useCallback(
    async (baseUrl: string, force = false): Promise<ModelInfo[]> => {
      const stamp = settings.localCatalogFetchedAt;
      const existing = settings.localCatalog;
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
    [setLocalCatalog, setModelsLoading, setModelsError, settings.localCatalog, settings.localCatalogFetchedAt],
  );
}

export function useRefreshModels(
  deps: ModelCatalogDeps,
  fetchNimModels: ReturnType<typeof useFetchNimModels>,
  fetchLocalModels: ReturnType<typeof useFetchLocalModels>,
) {
  const { settings } = deps;

  return useCallback(async () => {
    if (settings.provider === 'local') {
      const url = settings.localServerUrl.trim();
      if (url) await fetchLocalModels(url, true);
    } else {
      const endpoint = resolveProviderEndpoint(settings);
      if (endpoint.apiKey) await fetchNimModels(endpoint.apiKey, true);
    }
  }, [settings, fetchNimModels, fetchLocalModels]);
}