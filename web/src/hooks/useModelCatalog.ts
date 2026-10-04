import { useCallback, useRef } from 'react';
import { listModels, listLocalCatalog, type ModelInfo } from '../lib/models';
import { resolveProviderEndpoint } from '../lib/providers';
import { inspectLocalServerUrl } from '../lib/localEndpoint';
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
  setLocalCatalog: (models: ModelInfo[], baseUrl?: string) => void;
  getSettings: () => Settings;
}

export function useFetchCloudModels(deps: ModelCatalogDeps) {
  const fetchedEndpointRef = useRef<{ provider: Settings['provider']; baseUrl: string; key: string } | null>(null);
  const requestIdRef = useRef(0);
  const { settings, availableModels, modelsFetchedAt, setModels, setModelsLoading, setModelsError, getSettings } = deps;

  return useCallback(
    async (key: string, force = false): Promise<ModelInfo[]> => {
      const provider = settings.provider;
      const endpoint = resolveProviderEndpoint(settings);
      const matchesCurrentSettings = () => {
        const current = getSettings();
        if (current.provider !== provider) return false;
        const currentEndpoint = resolveProviderEndpoint(current);
        return currentEndpoint.baseUrl === endpoint.baseUrl && currentEndpoint.apiKey === key;
      };
      if (!matchesCurrentSettings()) return [];
      const cached = fetchedEndpointRef.current;
      if (
        !force &&
        cached?.key === key && cached.provider === provider && cached.baseUrl === endpoint.baseUrl &&
        Date.now() - modelsFetchedAt < MODEL_TTL_MS &&
        availableModels.length > 0
      ) {
        return availableModels;
      }
      const requestId = ++requestIdRef.current;
      const isCurrent = () => requestId === requestIdRef.current && matchesCurrentSettings();
      setModelsLoading(true);
      setModelsError(null);
      try {
        const models = await listModels(provider, key, endpoint.baseUrl);
        if (!isCurrent()) return [];
        setModels(models);
        fetchedEndpointRef.current = { provider, baseUrl: endpoint.baseUrl, key };
        return models;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (isCurrent()) setModelsError(msg);
        return [];
      } finally {
        if (isCurrent()) setModelsLoading(false);
      }
    },
    [availableModels, modelsFetchedAt, setModels, setModelsLoading, setModelsError, settings, getSettings],
  );
}

export function useFetchLocalModels(deps: ModelCatalogDeps) {
  const requestIdRef = useRef(0);
  const { settings, setLocalCatalog, setModelsLoading, setModelsError, getSettings } = deps;

  return useCallback(
    async (baseUrl: string, force = false): Promise<ModelInfo[]> => {
      const currentAtStart = getSettings();
      if (currentAtStart.provider !== 'local') return [];
      const inspected = inspectLocalServerUrl(baseUrl);
      if (inspected.error) {
        if (currentAtStart.localServerUrl.trim() === baseUrl.trim()) {
          ++requestIdRef.current;
          setModelsError(inspected.error);
          setModelsLoading(false);
        }
        return [];
      }
      const normalizedBaseUrl = inspected.baseUrl;

      const matchesCurrentSettings = () => {
        const current = getSettings();
        if (current.provider !== 'local') return false;
        const endpoint = inspectLocalServerUrl(current.localServerUrl);
        return !endpoint.error && endpoint.baseUrl === normalizedBaseUrl;
      };
      if (!matchesCurrentSettings()) return [];

      const stamp = settings.localCatalogFetchedAt;
      const existing = settings.localCatalog;
      if (
        !force &&
        existing.length > 0 &&
        settings.localCatalogBaseUrl === normalizedBaseUrl &&
        Date.now() - stamp < LOCAL_CATALOG_TTL_MS
      ) {
        return existing;
      }
      const requestId = ++requestIdRef.current;
      const isCurrent = () => requestId === requestIdRef.current && matchesCurrentSettings();
      setModelsLoading(true);
      setModelsError(null);
      try {
        const models = await listLocalCatalog(normalizedBaseUrl, '');
        if (!isCurrent()) return [];
        setLocalCatalog(models, normalizedBaseUrl);
        return models;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (isCurrent()) {
          const current = getSettings();
          if (current.localCatalog.length > 0 || current.localCatalogBaseUrl) setLocalCatalog([], '');
          setModelsError(msg);
        }
        return [];
      } finally {
        if (isCurrent()) setModelsLoading(false);
      }
    },
    [
      getSettings,
      setLocalCatalog,
      setModelsLoading,
      setModelsError,
      settings.localCatalog,
      settings.localCatalogBaseUrl,
      settings.localCatalogFetchedAt,
    ],
  );
}

export function useRefreshModels(
  deps: ModelCatalogDeps,
  fetchCloudModels: ReturnType<typeof useFetchCloudModels>,
  fetchLocalModels: ReturnType<typeof useFetchLocalModels>,
) {
  const { settings } = deps;

  return useCallback(async () => {
    if (settings.provider === 'local') {
      const url = settings.localServerUrl.trim();
      if (url) await fetchLocalModels(url, true);
    } else {
      const endpoint = resolveProviderEndpoint(settings);
      if (endpoint.apiKey) await fetchCloudModels(endpoint.apiKey, true);
    }
  }, [settings, fetchCloudModels, fetchLocalModels]);
}
