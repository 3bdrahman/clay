import { useState } from 'react';
import { getProviderConfig, getProviderApiKeyField, resolveProviderEndpoint, type ProviderApiKeyField } from '../../lib/providers';
import { getCloudModelLabel, getCloudModelOptions } from '../../lib/modelPolicy';
import type { Settings, ModelInfo } from '../../lib/types';
import type { PickedModels } from '../../lib/models';

interface Props {
  settings: Settings;
  availableModels: ModelInfo[];
  modelsLoading: boolean;
  modelsError: string | null;
  pickedModels: PickedModels;
  updateSettings: (patch: Partial<Settings>) => void;
  refreshModels: () => Promise<void>;
}

export function CloudProviderSettings({
  settings,
  availableModels,
  modelsLoading,
  modelsError,
  pickedModels,
  updateSettings,
  refreshModels,
}: Props) {
  const [showKey, setShowKey] = useState(false);
  const provider = settings.provider;
  if (provider === 'local') return null;

  const config = getProviderConfig(provider);
  const apiKeyField = getProviderApiKeyField(settings.provider);
  const currentApiKey = apiKeyField !== undefined ? settings[apiKeyField] : '';
  const endpoint = resolveProviderEndpoint(settings);
  const modelOptions = getCloudModelOptions(provider, availableModels);
  const selectedModel = settings.pickedModelsOverride.chatModel ?? '';
  const selectionUnavailable = !!selectedModel && !modelOptions.some(model => model.id === selectedModel);

  return (
    <>
      <div className="rounded-lg border border-brand-200 dark:border-brand-800 bg-brand-50/60 dark:bg-brand-900/20 p-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm">{config.displayName}</span>
          <span className="text-[10px] uppercase font-bold text-emerald-600 dark:text-emerald-400">
            {settings.provider === 'nim' ? 'Developer credits' : 'Free models'}
          </span>
          <span className="text-[10px] text-ink-400 ml-auto">
            {availableModels.length > 0 ? `${modelOptions.length} approved models` : 'not loaded'}
          </span>
        </div>
        <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-1">
          {endpoint.configurationError ? 'Connect your relay to enable NIM requests.' : `Model requests go to ${endpoint.baseUrl}.`}
          {' '}One chat model handles routing, analysis, and answers. Embeddings run locally in your browser.
        </p>
      </div>

      {settings.provider === 'nim' && (
        <div className="space-y-2">
          <label className="block text-xs font-semibold text-ink-600 dark:text-ink-300" htmlFor="nim-relay-base-url">NIM relay base URL</label>
          <input
            id="nim-relay-base-url"
            aria-label="NIM relay base URL"
            type="url"
            value={settings.nimBaseUrl ?? ''}
            onChange={e => updateSettings({ nimBaseUrl: e.target.value })}
            placeholder={import.meta.env.VITE_NIM_BASE_URL || (import.meta.env.DEV ? '/nim-api/v1 (development proxy)' : 'https://your-relay.workers.dev/v1')}
            aria-describedby="nim-relay-help"
            aria-invalid={!!endpoint.configurationError}
            className="w-full px-3 py-2 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none"
          />
          <p id="nim-relay-help" className="text-xs text-ink-500 dark:text-ink-400">
            NVIDIA blocks direct browser requests. Use a relay you control; your key and model requests pass through it.
            {' '}<a href="https://github.com/3bdrahman/clay/blob/master/docs/nim-relay.md" target="_blank" rel="noopener noreferrer" className="text-brand-600 dark:text-brand-400 underline">Relay setup</a>
          </p>
          {endpoint.configurationError && <p role="alert" className="text-xs text-rose-600 dark:text-rose-400">{endpoint.configurationError}</p>}
          <p className="text-xs text-ink-500 dark:text-ink-400">NVIDIA developer credits and account limits apply. Clay cannot verify your remaining credits.</p>
        </div>
      )}

      <div>
        <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
          {config.displayName} API Key <span className="text-ink-400 normal-case">({config.apiKeyHint})</span>
        </label>
        <div className="relative">
          <input
            aria-label={`${config.displayName} API key`}
            type={showKey ? 'text' : 'password'}
            value={currentApiKey}
            onChange={e => {
              const patch: Partial<Pick<Settings, ProviderApiKeyField>> = {};
              if (apiKeyField !== undefined) patch[apiKeyField] = e.target.value;
              updateSettings(patch);
            }}
            placeholder={config.apiKeyHint}
            className="w-full px-3 py-2 pr-10 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none font-mono"
          />
          <button
            onClick={() => setShowKey(s => !s)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-ink-400 hover:text-ink-700 dark:hover:text-ink-200"
            type="button"
            aria-label={showKey ? 'Hide key' : 'Show key'}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d={
                  showKey
                    ? 'M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21'
                    : 'M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z'
                }
              />
            </svg>
          </button>
        </div>
        <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-1.5">
          {settings.provider === 'nim'
            ? 'Stored in this browser. Sent through your configured relay to NVIDIA.'
            : `Stored in this browser. Sent only to ${config.displayName}.`}
        </p>
        {config.apiKeyUrl && (
          <a
            href={config.apiKeyUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 mt-2 text-[11px] font-semibold text-brand-600 dark:text-brand-400 hover:underline"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13.828 10.172a4 4 0 015.656 0l1.415 1.415a4 4 0 010 5.656l-3 3a4 4 0 01-5.656 0M10.172 13.828a4 4 0 01-5.656 0l-1.415-1.415a4 4 0 010-5.656l3-3a4 4 0 015.656 0"
              />
            </svg>
            Get API key for {config.displayName}
            <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
              />
            </svg>
          </a>
        )}
      </div>

      <div className="rounded-lg border border-ink-200 dark:border-ink-700 p-3 space-y-2 bg-ink-50/50 dark:bg-ink-800/30">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
              Model selection
            </div>
            <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-0.5">
              {provider === 'openrouter'
                ? 'Two approved free choices. Live prices are checked and paid routing is blocked.'
                : 'Two approved developer-access choices. NVIDIA credits and account limits apply.'}
            </p>
          </div>
          <button
            onClick={refreshModels}
            disabled={!currentApiKey || modelsLoading || !!endpoint.configurationError}
            className="px-2 py-1 text-[11px] font-semibold text-brand-600 dark:text-brand-400 hover:bg-brand-50 dark:hover:bg-brand-900/30 rounded disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
            type="button"
            title={`Fetch latest model catalog from ${config.displayName}`}
          >
            <svg
              className={`w-3 h-3 ${modelsLoading ? 'animate-spin' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
            {modelsLoading ? 'Loading…' : 'Refresh'}
          </button>
        </div>

        {modelsError && modelsError !== endpoint.configurationError && (
          <div className="text-[11px] text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-900/30 rounded px-2 py-1.5">
            {modelsError}
          </div>
        )}

        {availableModels.length === 0 && !modelsLoading && !modelsError && !endpoint.configurationError && (
          <div className="text-[11px] text-ink-500 dark:text-ink-400 italic px-1">
            Add an API key to load the catalog.
          </div>
        )}

        {availableModels.length > 0 && modelOptions.length === 0 && !modelsLoading && (
          <p role="status" className="text-xs text-amber-700 dark:text-amber-300">No approved models are currently available. Refresh the catalog later or choose another provider.</p>
        )}

        <div className="space-y-3">
          <div>
            <label htmlFor="cloud-chat-model" className="block text-xs font-semibold text-ink-500 dark:text-ink-400 mb-1">Chat model</label>
            <select
              id="cloud-chat-model"
              aria-label="Chat model"
              value={selectedModel || pickedModels.chat || ''}
              onChange={e => updateSettings({ pickedModelsOverride: { chatModel: e.target.value } })}
              disabled={modelOptions.length === 0 || modelsLoading || !!endpoint.configurationError}
              className="w-full px-3 py-2 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none disabled:opacity-50"
            >
              <option value="" disabled>Choose an approved model</option>
              {selectionUnavailable && <option value={selectedModel} disabled>Previous selection unavailable</option>}
              {modelOptions.map(model => <option key={model.id} value={model.id}>{getCloudModelLabel(provider, model.id)}</option>)}
            </select>
            {selectionUnavailable && <p role="status" className="mt-2 text-xs text-amber-700 dark:text-amber-300">Your previous selection is outside the current approved list. Choose one of the available models.</p>}
            <p className="mt-1 text-xs text-ink-500 dark:text-ink-400">Used for routing, analysis, answers, and evaluation.</p>
          </div>
          <p className="text-[10px] text-ink-500 dark:text-ink-400">
            Embeddings run locally in your browser (transformers.js) — nothing to pick.
          </p>
        </div>

      </div>
    </>
  );
}
