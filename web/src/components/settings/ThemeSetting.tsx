import type { Settings } from '../../lib/types';

interface Props {
  settings: Settings;
  updateSettings: (patch: Partial<Settings>) => void;
}

export function ThemeSetting({ settings, updateSettings }: Props) {
  return (
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
  );
}