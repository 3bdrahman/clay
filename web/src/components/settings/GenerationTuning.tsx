import type { Settings } from '../../lib/types';

const MIN_TOOL_LOOP_TOKENS = 25_000;
const MAX_TOOL_LOOP_TOKENS = 400_000;
const TOOL_LOOP_TOKENS_STEP = 25_000;
const DEFAULT_TOOL_LOOP_TOKENS = 100_000;

interface Props {
  settings: Settings;
  updateSettings: (patch: Partial<Settings>) => void;
}

export function GenerationTuning({ settings, updateSettings }: Props) {
  return (
    <>
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

      <div>
        <label className="block text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">
          Analysis token budget: <span className="font-mono">{(settings.maxToolLoopTokens ?? DEFAULT_TOOL_LOOP_TOKENS).toLocaleString()}</span>
        </label>
        <input
          type="range"
          min={MIN_TOOL_LOOP_TOKENS}
          max={MAX_TOOL_LOOP_TOKENS}
          step={TOOL_LOOP_TOKENS_STEP}
          value={settings.maxToolLoopTokens ?? DEFAULT_TOOL_LOOP_TOKENS}
          onChange={e => updateSettings({ maxToolLoopTokens: parseInt(e.target.value, 10) })}
          className="w-full accent-brand-500"
        />
        <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-1.5">
          Cumulative token budget for the data-analysis tool loop (every LLM call in one analysis). Lower it to cap spend on data questions.
        </p>
      </div>
    </>
  );
}