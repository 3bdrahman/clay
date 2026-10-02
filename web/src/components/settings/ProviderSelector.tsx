import { PROVIDER_REGISTRY, type ProviderKind } from '../../lib/providers';
import type { Settings } from '../../lib/types';

interface Props {
  provider: Settings['provider'];
  onChange: (kind: ProviderKind) => void;
}

export function ProviderSelector({
  provider,
  onChange,
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
            onClick={() => onChange(kind as ProviderKind)}
            className={`px-3 py-2 rounded-lg border text-sm font-medium ${
              provider === kind
                ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/30'
                : 'border-ink-200 dark:border-ink-700'
            }`}
            type="button"
            aria-pressed={provider === kind}
          >
            {config.displayName}
            <div className="text-[10px] text-ink-500 dark:text-ink-400 font-normal">
              {config.requiresApiKey ? 'API key required' : 'Your installed models'}
            </div>
          </button>
        ))}
      </>
    </div>
  );
}
