import { useAppStore } from '../store';
import type { PickedModels } from '../lib/models';
import type { LocalModelPicks } from '../lib/types';
import { useConfirm } from '../hooks/useConfirm';
import { useModalFocus } from '../hooks/useModalFocus';
import { ProviderSelector } from './settings/ProviderSelector';
import { LocalServerSettings } from './settings/LocalServerSettings';
import { CloudProviderSettings } from './settings/CloudProviderSettings';
import { WebSearchSetting } from './settings/WebSearchSetting';
import { ThemeSetting } from './settings/ThemeSetting';
import { GenerationTuning } from './settings/GenerationTuning';

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
  pickedModels: PickedModels;
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
  const [confirm, renderConfirmDialog] = useConfirm();
  const { dialogRef, stopBackdrop } = useModalFocus(open);

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
  const isLocal = settings.provider === 'local';

  function setLocalModel(key: LocalModelKey, value: string) {
    updateSettings({ localModels: { ...settings.localModels, [key]: value } });
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex" onClick={onClose}>
      <div className="absolute inset-0 bg-black/30 animate-fade-in" />
      <div
        ref={dialogRef}
        className="relative ml-auto w-full max-w-md bg-white dark:bg-ink-900 shadow-2xl overflow-y-auto animate-slide-up"
        onClick={stopBackdrop}
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
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
          <ProviderSelector
            provider={settings.provider}
            onChange={kind => updateSettings({ provider: kind })}
          />

          {isLocal ? (
            <LocalServerSettings
              settings={settings}
              localCatalog={localCatalog}
              modelsLoading={modelsLoading}
              modelsError={modelsError}
              urlValidationError={urlValidationError}
              updateSettings={updateSettings}
              refreshModels={refreshModels}
              setLocalModel={setLocalModel}
            />
          ) : (
            <CloudProviderSettings
              key={settings.provider}
              settings={settings}
              availableModels={availableModels}
              modelsLoading={modelsLoading}
              modelsError={modelsError}
              pickedModels={pickedModels}
              updateSettings={updateSettings}
              refreshModels={refreshModels}
            />
          )}

          <WebSearchSetting settings={settings} updateSettings={updateSettings} />
          <ThemeSetting settings={settings} updateSettings={updateSettings} />
          <GenerationTuning settings={settings} updateSettings={updateSettings} />

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
            <p className="font-semibold">Clay — RAG Assistant</p>
            <p>
              The workspace runs in your browser. Provider keys are stored here; NIM keys also pass through the relay you configure.
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
