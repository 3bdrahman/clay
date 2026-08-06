import { useEffect, useState } from 'react';
import { useAppStore } from '../store';
import {
  NIM_FREE_KEY_URL,
  LOCAL_PROVIDER_HINT,
  OLLAMA_CORS_HINT,
  isOllamaUrl,
} from '../lib/providers';
import { modelClass } from '../lib/models';
import type { LocalModelPicks } from '../lib/types';
import { useConfirm } from '../hooks/useConfirm';

function validateLocalServerUrl(url: string): string | null {
  const trimmed = url.trim();
  if (!trimmed) return 'Server URL is required.';
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return 'Not a valid URL. Include the scheme, e.g. http://localhost:11434/v1';
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    return 'URL must use http:// or https://';
  }
  if (!parsed.hostname) {
    return 'URL is missing a hostname.';
  }
  return null;
}

type LocalModelKey = keyof LocalModelPicks;

interface Props {
  open: boolean;
  onClose: () => void;
  refreshModels: () => Promise<void>;
  pickedModels: {
    routing: string | undefined;
    codeGen: string | undefined;
    answer: string | undefined;
    eval: string | undefined;
    embedding: string | undefined;
  };
  resetAll: () => void;
  clearSandboxData: () => void;
}

export function SettingsPanel({ open, onClose, refreshModels, pickedModels, resetAll, clearSandboxData }: Props) {
  const settings = useAppStore(s => s.settings);
  const updateSettings = useAppStore(s => s.updateSettings);
  const availableModels = useAppStore(s => s.availableModels);
  const localCatalog = useAppStore(s => s.settings.localCatalog);
  const modelsLoading = useAppStore(s => s.modelsLoading);
  const modelsError = useAppStore(s => s.modelsError);
  const [showKey, setShowKey] = useState(false);
  const [showEmbKey, setShowEmbKey] = useState(false);
  const [showCorsHint, setShowCorsHint] = useState(false);
  const [confirm, renderConfirmDialog] = useConfirm();

  const handleResetAll = async () => {
    const ok = await confirm({
      title: 'Reset everything?',
      message: 'Reset all settings, chat history, sandbox data, and vector store. This cannot be undone.',
      confirmLabel: 'Reset everything',
      destructive: true,
    });
    if (ok) {
      clearSandboxData();
      resetAll();
      onClose();
    }
  };

  const urlValidationError = settings.provider === 'local'
    ? validateLocalServerUrl(settings.localServerUrl)
    : null;
  const showOllamaHint = settings.provider === 'local' && isOllamaUrl(settings.localServerUrl);

  useEffect(() => {
    if (!modelsError) {
      setShowCorsHint(false);
    }
  }, [modelsError]);

  if (!open) return null;

  const tasks: Array<{ key: LocalModelKey; label: string; hint: string }> = [
    { key: 'routing', label: 'Routing', hint: 'Decides source (docs/data/web)' },
    { key: 'codeGen', label: 'Code generation', hint: 'Writes Arquero code' },
    { key: 'answer', label: 'Answer', hint: 'Final RAG answer' },
    { key: 'eval', label: 'Evaluation', hint: 'Doc relevance + answer grading' },
    { key: 'embedding', label: 'Embedding', hint: 'Document + query vectors' },
  ];

  const isLocal = settings.provider === 'local';
  const providerBadge = isLocal
    ? `${localCatalog.length} models loaded`
    : availableModels.length > 0
    ? `${availableModels.length} models`
    : 'not loaded';
  const providerHintLine = isLocal
    ? `All LLM calls go to ${settings.localServerUrl || '(unset)'}. No API key required.`
    : 'All LLM calls go to integrate.api.nvidia.com/v1. One API key — Clay picks the best model per task from the live catalog.';

  function setLocalModel(key: LocalModelKey, value: string) {
    updateSettings({ localModels: { ...settings.localModels, [key]: value } });
  }

  return (
    <div className="fixed inset-0 z-50 flex" onClick={onClose}>
      <div className="absolute inset-0 bg-black/30 animate-fade-in" />
      <div
        className="relative ml-auto w-full max-w-md bg-white dark:bg-ink-900 shadow-2xl overflow-y-auto animate-slide-up"
        onClick={e => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-white dark:bg-ink-900 border-b border-ink-200 dark:border-ink-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Settings</h2>
          <button
            onClick={onClose}
            className="text-ink-500 hover:text-ink-800 dark:hover:text-ink-200"
            type="button"
            aria-label="Close settings"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
           </svg>
         </button>
       </div>

        <div className="px-6 py-4 space-y-6">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
              Provider
           </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => updateSettings({ provider: 'nim' })}
                className={`px-3 py-2 rounded-lg border text-sm font-medium ${
                  !isLocal
                    ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/30'
                    : 'border-ink-200 dark:border-ink-700'
                }`}
                type="button"
              >
                NVIDIA NIM
                <div className="text-[10px] text-ink-500 dark:text-ink-400 font-normal">cloud · needs key</div>
             </button>
              <button
                onClick={() => {
                  const switching = !isLocal;
                  updateSettings({ provider: 'local' });
                  if (switching && settings.localServerUrl.trim() && !urlValidationError && localCatalog.length === 0) {
                    void refreshModels();
                  }
                }}
                className={`px-3 py-2 rounded-lg border text-sm font-medium ${
                  isLocal
                    ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/30'
                    : 'border-ink-200 dark:border-ink-700'
                }`}
                type="button"
              >
                Local server
                <div className="text-[10px] text-ink-500 dark:text-ink-400 font-normal">Ollama / LM Studio / vLLM</div>
            </button>
           </div>
         </div>

          {isLocal ? (
            <div className="rounded-lg border border-emerald-200 dark:border-emerald-800 bg-emerald-50/60 dark:bg-emerald-900/20 p-3 space-y-2">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm">Local server</span>
                <span className="text-[10px] uppercase font-bold text-emerald-600 dark:text-emerald-400">
                  Private
               </span>
                <span className="text-[10px] text-ink-400 ml-auto">{providerBadge}</span>
             </div>
              <p className="text-[11px] text-ink-500 dark:text-ink-400">{providerHintLine}</p>
              <p className="text-[10px] text-ink-400 italic">{LOCAL_PROVIDER_HINT}</p>
           </div>
          ) : (
            <div className="rounded-lg border border-brand-200 dark:border-brand-800 bg-brand-50/60 dark:bg-brand-900/20 p-3">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm">NVIDIA NIM</span>
                <span className="text-[10px] uppercase font-bold text-emerald-600 dark:text-emerald-400">
                  Free tier
               </span>
                <span className="text-[10px] text-ink-400 ml-auto">{providerBadge}</span>
             </div>
              <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-1">{providerHintLine}</p>
           </div>
          )}

          {isLocal ? (
            <>
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
                    <div className="text-[11px] text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-900/30 rounded px-2 py-1.5">
                      {modelsError}
                   </div>
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
             </div>
            </>
          ) : (
            <>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
                  NVIDIA NIM API Key <span className="text-ink-400 normal-case">(nvapi-</span>
               </label>
                <div className="relative">
                  <input
                    type={showKey ? 'text' : 'password'}
                    value={settings.apiKey}
                    onChange={e => updateSettings({ apiKey: e.target.value })}
                    placeholder="nvapi-..."
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
                  Stored locally in your browser only. Never sent anywhere except NVIDIA NIM.
               </p>
                <a
                  href={NIM_FREE_KEY_URL}
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
                  Get a free NVIDIA NIM API key
                  <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                    />
                 </svg>
               </a>
             </div>

              <div className="rounded-lg border border-ink-200 dark:border-ink-700 p-3 space-y-2 bg-ink-50/50 dark:bg-ink-800/30">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
                      Auto-picked task models
                   </div>
                    <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-0.5">
                      Picked from NIM's live catalog by size + family. Refresh after NIM updates its lineup.
                   </p>
                 </div>
                  <button
                    onClick={refreshModels}
                    disabled={!settings.apiKey || modelsLoading}
                    className="px-2 py-1 text-[11px] font-semibold text-brand-600 dark:text-brand-400 hover:bg-brand-50 dark:hover:bg-brand-900/30 rounded disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
                    type="button"
                    title="Fetch latest model catalog from NIM"
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

                <ul className="text-[11px] space-y-1">
                  {tasks.map(t => (
                    <li key={t.key} className="flex items-center justify-between gap-2">
                      <div className="flex flex-col min-w-0">
                        <span className="text-ink-600 dark:text-ink-300 font-medium">{t.label}</span>
                        <span className="text-[10px] text-ink-400">{t.hint}</span>
                     </div>
                      <span
                        className="font-mono text-ink-700 dark:text-ink-200 truncate text-right max-w-[60%]"
                        title={pickedModels[t.key] ?? '—'}
                      >
                        {pickedModels[t.key] ?? <span className="text-ink-400 italic">—</span>}
                     </span>
                   </li>
                  ))}
               </ul>

                {pickedModels.answer && (
                  <div className="pt-2 mt-1 border-t border-ink-200 dark:border-ink-700 text-[10px] text-ink-500 dark:text-ink-400 flex items-center justify-between">
                    <span>Class: {modelClass(pickedModels.answer)}</span>
                    {pickedModels.embedding && (
                      <span>Embed class: {modelClass(pickedModels.embedding)}</span>
                    )}
                 </div>
                )}
             </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
                  Embeddings key <span className="text-ink-400 normal-case">(optional</span>
               </label>
                <div className="relative">
                  <input
                    type={showEmbKey ? 'text' : 'password'}
                    value={settings.embeddingApiKey}
                    onChange={e => updateSettings({ embeddingApiKey: e.target.value })}
                    placeholder="Defaults to your LLM key"
                    className="w-full px-3 py-2 pr-10 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none font-mono"
                  />
                  <button
                    onClick={() => setShowEmbKey(s => !s)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-ink-400 hover:text-ink-700 dark:hover:text-ink-200"
                    type="button"
                    aria-label={showEmbKey ? 'Hide key' : 'Show key'}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d={
                          showEmbKey
                            ? 'M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21'
                            : 'M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z'
                        }
                      />
                   </svg>
                 </button>
               </div>
                <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-1.5">
                  Leave empty to reuse your LLM key.
               </p>
             </div>
            </>
          )}

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
              Web Search
           </label>
            <select
              value={settings.webSearchProvider}
              onChange={e =>
                updateSettings({ webSearchProvider: e.target.value as typeof settings.webSearchProvider })
              }
              className="w-full px-3 py-2 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none"
            >
              <option value="duckduckgo">DuckDuckGo (no key</option>
              <option value="serper">Serper (Google, requires key</option>
              <option value="none">Disabled</option>
           </select>
            {settings.webSearchProvider === 'serper' && (
              <input
                type="password"
                value={settings.serperApiKey}
                onChange={e => updateSettings({ serperApiKey: e.target.value })}
                placeholder="Serper API key"
                className="w-full mt-2 px-3 py-2 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none font-mono"
              />
            )}
         </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
              Theme
           </label>
            <div className="flex gap-2">
              {(['light', 'dark', 'system'] as const).map(t => (
                <button
                  key={t}
                  onClick={() => updateSettings({ theme: t })}
                  className={`flex-1 px-3 py-2 rounded-lg border text-sm capitalize ${
                    settings.theme === t
                      ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/30'
                      : 'border-ink-200 dark:border-ink-700'
                  }`}
                  type="button"
                >
                  {t}
               </button>
              ))}
           </div>
         </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
              Temperature: <span className="font-mono">{settings.temperature.toFixed(2)}</span>
           </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={settings.temperature}
              onChange={e => updateSettings({ temperature: parseFloat(e.target.value) })}
              className="w-full accent-brand-500"
            />
         </div>

          <div className="pt-4 border-t border-ink-200 dark:border-ink-700">
            <button
              onClick={handleResetAll}
              className="text-sm text-rose-600 dark:text-rose-400 hover:underline"
              type="button"
            >
              Reset everything
           </button>
         </div>

<div className="pt-4 border-t border-ink-200 dark:border-ink-700 text-xs text-ink-500 dark:text-ink-400 space-y-1.5">
            <p className="font-semibold">Clay — AI Internal Assistant</p>
            <p>
              Runs entirely in your browser. No backend. Your API key never leaves your browser except to your configured
              provider.
            </p>
            <p className="text-[10px] opacity-70">
              Built for static deployment on GitHub Pages or any static host.
            </p>
          </div>
       </div>
     </div>
     {renderConfirmDialog()}
   </div>
  );
}
