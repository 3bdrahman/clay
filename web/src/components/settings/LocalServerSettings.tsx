import { useState, useEffect } from 'react';
import { LOCAL_PROVIDER_HINT, isOllamaUrl, getProviderConfig } from '../../lib/providers';
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
  normalizedBaseUrl: string;
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
  normalizedBaseUrl,
  updateSettings,
  refreshModels,
  setLocalModel,
}: Props) {
  const [showCorsHint, setShowCorsHint] = useState(false);

  const showOllamaHint = isOllamaUrl(settings.localServerUrl);
  const config = getProviderConfig('local');
  const pageOrigin = typeof window === 'undefined' ? 'this page origin' : window.location.origin;
  const catalogUrl = normalizedBaseUrl ? `${normalizedBaseUrl}/models` : '';
  const localSetupSummary = [
    `Ollama: stop any running Ollama process, then start it with OLLAMA_ORIGINS='${pageOrigin}' ollama serve.`,
    'LM Studio: enable CORS in Server Settings, or start the CLI server with lms server start --cors.',
    'Chrome/Edge may ask for local-network or loopback access. Allow it for this site.',
    'Use 127.0.0.1 if localhost resolves to IPv6 on your machine. Do not enter 0.0.0.0 as the browser URL.',
  ].join('\n');

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
          <span className="text-[10px] text-ink-400 ml-auto">
            {localCatalog.length} {localCatalog.length === 1 ? 'model' : 'models'} loaded
          </span>
        </div>
        <p className="text-[11px] text-ink-500 dark:text-ink-400">
          Model requests go to {normalizedBaseUrl || settings.localServerUrl || '(unset)'}.
          Cloud provider keys are never sent to this endpoint.
        </p>
        <p className="text-[10px] text-ink-400 italic">{LOCAL_PROVIDER_HINT}</p>
      </div>

      <div>
        <label htmlFor="local-server-url" className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
          Local server URL
        </label>
        <input
          id="local-server-url"
          aria-label="Local server URL"
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
          <div className="text-[11px] text-ink-500 dark:text-ink-400 mt-1.5 space-y-1">
            <p>
              Ollama: <span className="font-mono">http://127.0.0.1:11434/v1</span>.
              LM Studio: <span className="font-mono">http://127.0.0.1:1234/v1</span>.
              vLLM: <span className="font-mono">http://127.0.0.1:8000/v1</span>.
            </p>
            {catalogUrl && (
              <p>
                Model catalog: <span className="font-mono">{catalogUrl}</span>
              </p>
            )}
          </div>
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
            {localSetupSummary}
          </p>
        )}
        <details className="mt-2 text-[11px] text-ink-600 dark:text-ink-300 bg-ink-50 dark:bg-ink-800/40 border border-ink-200 dark:border-ink-700 rounded px-2 py-1.5">
          <summary className="cursor-pointer font-semibold">
            Local setup for this page
          </summary>
          <div className="mt-1.5 space-y-1.5">
            <p>
              Allow server CORS for <span className="font-mono">{pageOrigin}</span>.
            </p>
            <p className="whitespace-pre-line">{localSetupSummary}</p>
            <p>
              Do not use browser security flags, <span className="font-mono">no-cors</span> mode, or a public proxy for local models.
            </p>
            <a
              href="https://github.com/3bdrahman/clay/blob/master/docs/local-models.md"
              target="_blank"
              rel="noreferrer"
              className="inline-flex font-semibold text-brand-600 dark:text-brand-400 hover:underline"
            >
              Local model setup guide
            </a>
          </div>
        </details>
      </div>

      <div className="rounded-lg border border-ink-200 dark:border-ink-700 p-3 space-y-3 bg-ink-50/50 dark:bg-ink-800/30">
        <div className="flex items-center justify-between gap-2">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
              Local catalog
            </div>
            <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-0.5">
              Fetched from <span className="font-mono">{catalogUrl || '(unset)'}</span>.
              Pick a chat model below.
            </p>
          </div>
          <button
            onClick={refreshModels}
            disabled={!!urlValidationError || modelsLoading}
            aria-label="Discover local models"
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

        {modelsLoading && (
          <div role="status" className="text-[11px] text-ink-600 dark:text-ink-300 bg-ink-100 dark:bg-ink-800 rounded px-2 py-1.5">
            Discovering models from {catalogUrl || 'the configured local server'}...
          </div>
        )}

        {modelsError && (
          <div className="space-y-1.5">
            <div role="alert" className="text-[11px] text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-900/30 rounded px-2 py-1.5 whitespace-pre-line">
              {modelsError}
            </div>
            {showOllamaHint && (
              <details className="text-[11px] text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded px-2 py-1.5">
                <summary className="cursor-pointer font-semibold">
                  Ollama setup checklist
                </summary>
                <p className="mt-1.5 whitespace-pre-line">{localSetupSummary}</p>
              </details>
            )}
          </div>
        )}

        {localCatalog.length > 0 && !modelsLoading && !modelsError && (
          <div role="status" className="text-[11px] text-emerald-700 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-900/20 rounded px-2 py-1.5">
            Loaded {localCatalog.length} local {localCatalog.length === 1 ? 'model' : 'models'} from {catalogUrl}.
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
            <label htmlFor={`local-model-${t.key}`} className="block text-[10px] font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-1">
              {t.label}
              <span className="ml-1 normal-case text-ink-400">— {t.hint}</span>
            </label>
            <input
              id={`local-model-${t.key}`}
              aria-label={t.label}
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
