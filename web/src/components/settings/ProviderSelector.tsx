import { PROVIDER_REGISTRY, type ProviderKind } from '../../lib/providers';
import type { Settings } from '../../lib/types';

interface Props {
  provider: Settings['provider'];
  onChange: (kind: ProviderKind) => void;
  localServerUrl: string;
  urlValidationError: string | null;
  localCatalogLength: number;
  refreshModels: () => Promise<void>;
}

export function ProviderSelector({
  provider,
  onChange,
  localServerUrl,
  urlValidationError,
  localCatalogLength,
  refreshModels,
}: Props) {
  return (
    <div>
      <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
        Provider
      </label>
      <>
        {Object.entries(PROVIDER_REGISTRY).map(([kind, config]) => (
          <button
            key={kind}
            onClick={() => {
              const switching = provider !== kind;
              onChange(kind as ProviderKind);
              // Auto-fetch models when switching to local with URL set
              if (switching && kind === 'local' && localServerUrl.trim() && !urlValidationError && localCatalogLength === 0) {
                void refreshModels();
              }
            }}
            className={`px-3 py-2 rounded-lg border text-sm font-medium ${
              provider === kind
                ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/30'
                : 'border-ink-200 dark:border-ink-700'
            }`}
            type="button"
          >
            {config.displayName}
            <div className="text-[10px] text-ink-500 dark:text-ink-400 font-normal">
              {config.freeTier ? 'free tier' : 'paid'} · {config.requiresApiKey ? 'needs key' : 'no key'}
            </div>
          </button>
        ))}
      </>
    </div>
  );
}