import { useState, useEffect } from 'react';
import { LOCAL_PROVIDER_HINT, OLLAMA_CORS_HINT, isOllamaUrl, getProviderConfig } from '../../lib/providers';
import type { LocalModelPicks, Settings } from '../../lib/types';

type LocalModelKey = keyof LocalModelPicks;

const tasks: Array<{ key: LocalModelKey; label: string; hint: string }> = [
  { key: 'chat', label: 'Chat model', hint: 'Used for routing, code generation, answer, and evaluation' },
];

interface Props {
  settings: Settings;
  localCatalog: Settings['localCatalog'];
  modelsLoading: boolean;
  modelsError: string | null;
  urlValidationError: string | null;
  updateSettings: (patch: Partial<Settings>) => void;
  refreshModels: () => Promise<void>;
  setLocalModel: (key: LocalModelKey, value: string) => void;
}

export function LocalServerSettings({
  settings,
  localCatalog,
  modelsLoading,
  modelsError,
  urlValidationError,
  updateSettings,
  refreshModels,
  setLocalModel,
}: Props) {
  const [showCorsHint, setShowCorsHint] = useState(false);

  const showOllamaHint = isOllamaUrl(settings.localServerUrl);
  const config = getProviderConfig('local');

  useEffect(() => {
    if (!modelsError) {
      setShowCorsHint(false);
    }
  }, [modelsError]);

  return (
    <>
      <div className="rounded-lg border border-emerald-200 dark:border-emerald-800 bg-emerald-50/60 dark:bg-emerald-900/20 p-3 space-y-2">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-sm">{config.displayName}</span>
          <span className="text-[10px] uppercase font-bold text-emerald-600 dark:text-emerald-400">
            Private
          </span>
          <span className="text-[10px] text-ink-400 ml-auto">
            {localCatalog.length} models loaded
          </span>
        </div>
        <p className="text-[11px] text-ink-500 dark:text-ink-400">
          All LLM calls go to {settings.localServerUrl || '(unset)'}. No API key required.
        </p>
        <p className="text-[10px] text-ink-400 italic">{LOCAL_PROVIDER_HINT}</p>
      </div>

      <div>
        <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
          Local server URL
        </label>
        <input
          type="text"
          value={settings.localServerUrl}
          onChange={e => updateSettings({ localServerUrl: e.target.value })}
          placeholder="http://localhost:11434/v1"
          className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-ink-800 text-sm focus:ring-2 dark:focus:ring-brand-900 outline-none font-mono ${
            urlValidationError
              ? 'border-rose-400 dark:border-rose-600 focus:border-rose-500 focus:ring-rose-200'
              : 'border-ink-200 dark:border-ink-700 focus:border-brand-500 focus:ring-brand-200'
          }`}
        />
        {urlValidationError ? (
          <p className="text-[11px] text-rose-600 dark:text-rose-400 mt-1.5">
            {urlValidationError}
          </p>
        ) : (
          <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-1.5">
            Ollama default is <span className="font-mono">http://localhost:11434/v1</span>.
            LM Studio: <span className="font-mono">http://localhost:1234/v1</span>.
            vLLM: <span className="font-mono">http://localhost:8000/v1</span>.
          </p>
        )}
        {showOllamaHint && !modelsError && (
          <button
            onClick={() => setShowCorsHint(s => !s)}
            className="mt-1.5 text-[11px] font-semibold text-amber-700 dark:text-amber-400 hover:underline"
            type="button"
          >
            Ollama detected — need CORS help?
          </button>
        )}
        {showOllamaHint && showCorsHint && (
          <p className="text-[11px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded px-2 py-1.5 mt-1.5">
            {OLLAMA_CORS_HINT}
          </p>
        )}
      </div>

      <div className="rounded-lg border border-ink-200 dark:border-ink-700 p-3 space-y-3 bg-ink-50/50 dark:bg-ink-800/30">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
              Local catalog
            </div>
            <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-0.5">
              Fetched from <span className="font-mono">{settings.localServerUrl || '(unset)'}/models</span>.
              Pick a model per task below.
            </p>
          </div>
          <button
            onClick={refreshModels}
            disabled={!!urlValidationError || modelsLoading}
            className="px-2 py-1 text-[11px] font-semibold text-brand-600 dark:text-brand-400 hover:bg-brand-50 dark:hover:bg-brand-900/30 rounded disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
            type="button"
            title="Fetch /models from the local server"
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
            {modelsLoading ? 'Loading…' : 'Discover'}
          </button>
        </div>

        {modelsError && (
          <div className="space-y-1.5">
            {modelsError.includes('CORS') ? (
              <details className="group">
                <summary className="cursor-pointer text-[11px] font-medium text-rose-600 dark:text-rose-400 select-none flex items-center gap-1.5">
                  <svg className="w-3.5 h-3.5 flex-shrink-0 text-rose-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  CORS blocked — the provider does not allow requests from this origin
                </summary>
                <div className="mt-2 text-[11px] text-ink-600 dark:text-ink-300 bg-rose-50 dark:bg-rose-900/30 border border-rose-200 dark:border-rose-800 rounded px-3 py-2 whitespace-pre-line animate-fade-in">
                  {modelsError}
                </div>
              </details>
            ) : (
              <div className="text-[11px] text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-900/30 rounded px-2 py-1.5">
                {modelsError}
              </div>
            )}
            {showOllamaHint && (
              <details className="text-[11px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded px-2 py-1.5">
                <summary className="cursor-pointer font-semibold">
                  Ollama detected — likely CORS issue
                </summary>
                <p className="mt-1.5 whitespace-pre-line">{OLLAMA_CORS_HINT}</p>
              </details>
            )}
          </div>
        )}

        {localCatalog.length === 0 && !modelsLoading && !modelsError && (
          <div className="text-[11px] text-ink-500 dark:text-ink-400 italic px-1">
            {urlValidationError
              ? 'Fix the URL above, then click Discover.'
              : 'Click Discover to load available models.'}
          </div>
        )}
      </div>

      <div className="space-y-3">
        {tasks.map(t => (
          <div key={t.key}>
            <label className="block text-[10px] font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-1">
              {t.label}
              <span className="ml-1 normal-case text-ink-400">— {t.hint}</span>
            </label>
            <input
              type="text"
              list={`local-models-${t.key}`}
              value={settings.localModels[t.key]}
              onChange={e => setLocalModel(t.key, e.target.value)}
              placeholder={localCatalog.length > 0 ? 'pick from catalog or type a model id' : 'model id (e.g. llama3.1:8b-instruct)'}
              className="w-full px-3 py-2 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none font-mono"
            />
            <datalist id={`local-models-${t.key}`}>
              {localCatalog.map(m => (
                <option key={m.id} value={m.id} />
              ))}
            </datalist>
          </div>
        ))}
        <p className="text-[10px] text-ink-500 dark:text-ink-400">
          Embeddings run locally in your browser (transformers.js) — nothing to pick.
        </p>
      </div>
    </>
  );
}