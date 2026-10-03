import type { Settings } from '../../lib/types';
import { getWebSearchAvailability } from '../../lib/websearch';

interface Props {
  settings: Settings;
  updateSettings: (patch: Partial<Settings>) => void;
}

export function WebSearchSetting({ settings, updateSettings }: Props) {
  const availability = getWebSearchAvailability(settings);
  const guidance =
    settings.webSearchProvider === 'mwmbl'
      ? 'Queries go directly to Mwmbl’s public independent index. No API key is needed; coverage and freshness vary by query.'
      : settings.webSearchProvider === 'serper' && settings.serperApiKey.trim().length === 0
      ? 'Serper is selected, but an API key is required. Add a key to enable live Google results.'
      : availability.message;

  return (
    <div>
      <label
        htmlFor="web-search-provider"
        className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2"
      >
        Web Search
      </label>
      <select
        id="web-search-provider"
        aria-label="Web search provider"
        value={settings.webSearchProvider}
        onChange={e =>
          updateSettings({ webSearchProvider: e.target.value as typeof settings.webSearchProvider })
        }
        className="w-full px-3 py-2 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none"
      >
        <option value="mwmbl">Mwmbl (no key)</option>
        <option value="serper">Serper (Google, requires key)</option>
        <option value="none">Disabled</option>
      </select>
      {settings.webSearchProvider === 'serper' && (
        <input
          type="password"
          aria-label="Serper API key"
          value={settings.serperApiKey}
          onChange={e => updateSettings({ serperApiKey: e.target.value })}
          placeholder="Serper API key"
          className="w-full mt-2 px-3 py-2 border border-ink-200 dark:border-ink-700 rounded-lg bg-white dark:bg-ink-800 text-sm focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none font-mono"
        />
      )}
      <p className="mt-2 text-xs text-ink-500 dark:text-ink-400">
        {guidance}
      </p>
    </div>
  );
}
