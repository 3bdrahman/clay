import { useState } from 'react';
import { getProviderConfig, getProviderApiKeyField, type ProviderApiKeyField } from '../../lib/providers';
import { modelClass } from '../../lib/models';
import type { Settings, ModelInfo } from '../../lib/types';
import type { PickedModels } from '../../lib/models';

const apiProviderTaskDisplay: Array<{ key: 'chatModel'; label: string; hint: string }> = [
  { key: 'chatModel', label: 'Chat model', hint: 'Used for routing, code generation, answer, and evaluation' },
];

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

  const config = getProviderConfig(settings.provider);
  const apiKeyField = getProviderApiKeyField(settings.provider);
  const currentApiKey = apiKeyField !== undefined ? settings[apiKeyField] : '';

  return (
    <>
      <div className="rounded-lg border border-brand-200 dark:border-brand-800 bg-brand-50/60 dark:bg-brand-900/20 p-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm">{config.displayName}</span>
          <span className="text-[10px] uppercase font-bold text-emerald-600 dark:text-emerald-400">
            {config.freeTier ? 'Free tier' : 'Paid'}
          </span>
          <span className="text-[10px] text-ink-400 ml-auto">
            {availableModels.length > 0 ? `${availableModels.length} models` : 'not loaded'}
          </span>
        </div>
        <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-1">
          All LLM calls go to {config.baseUrl}. One API key — one chat model for all tasks (routing, code gen, answer, eval). Embeddings run locally in your browser.
        </p>
      </div>

      <div>
        <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
          {config.displayName} API Key <span className="text-ink-400 normal-case">({config.apiKeyHint})</span>
        </label>
        <div className="relative">
          <input
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
          Stored locally in your browser only. Never sent anywhere except {config.displayName}.
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
              Choose one chat model from {config.displayName}, or enter its model ID.
            </p>
          </div>
          <button
            onClick={refreshModels}
            disabled={!currentApiKey || modelsLoading}
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

        {modelsError && (
          <div className="text-[11px] text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-900/30 rounded px-2 py-1.5">
            {modelsError}
          </div>
        )}

        {availableModels.length === 0 && !modelsLoading && !modelsError && (
          <div className="text-[11px] text-ink-500 dark:text-ink-400 italic px-1">
            Add an API key to load the catalog.
          </div>
        )}

        <div className="space-y-3">
          {apiProviderTaskDisplay.map(t => (
            <div key={t.key}>
              <label className="block text-[10px] font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-1">
                {t.label}
                <span className="ml-1 normal-case text-ink-400">— {t.hint}</span>
              </label>
              <div className="flex items-center gap-2">
                <input
                  aria-label={t.label}
                  list={`provider-models-${t.key}`}
                  placeholder="Choose or enter a chat model"
                  value={settings.pickedModelsOverride[t.key] || ''}
                  onChange={e => updateSettings({
                    pickedModelsOverride: { chatModel: e.target.value }
                  })}
                  className="flex-1 px-3 py-2 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none font-mono"
                />
                <datalist id={`provider-models-${t.key}`}>
                  {availableModels.map((m: ModelInfo) => (
                    <option key={m.id} value={m.id} />
                  ))}
                </datalist>
                {settings.pickedModelsOverride[t.key] && (
                  <button
                    onClick={() => updateSettings({
                      pickedModelsOverride: { chatModel: '' }
                    })}
                    className="px-2 py-1 text-[10px] text-ink-500 hover:text-ink-700 dark:hover:text-ink-300"
                    type="button"
                    title="Clear selection"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>
              <p className="text-[10px] text-ink-500 dark:text-ink-400 mt-0.5">
                Current: <span className="font-mono text-ink-700 dark:text-ink-200">
                  {pickedModels.chat || 'Choose a chat model'}
                </span>
              </p>
            </div>
          ))}
          <p className="text-[10px] text-ink-500 dark:text-ink-400">
            Embeddings run locally in your browser (transformers.js) — nothing to pick.
          </p>
        </div>

        {pickedModels.chat && (
          <div className="pt-2 mt-1 border-t border-ink-200 dark:border-ink-700 text-[10px] text-ink-500 dark:text-ink-400">
            <span>Class: {modelClass(pickedModels.chat)}</span>
          </div>
        )}
      </div>
    </>
  );
}