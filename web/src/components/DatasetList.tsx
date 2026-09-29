import { SandboxDataset } from '../store';

interface DatasetListProps {
  datasets: SandboxDataset[];
  onRemove: (name: string) => void;
}

export function DatasetList({ datasets, onRemove }: DatasetListProps) {
  if (datasets.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
          Datasets
        </div>
        <span className="text-[10px] text-ink-400">{datasets.length}</span>
      </div>
      <ul className="space-y-1.5">
        {datasets.map(d => (
          <li
            key={d.name}
            className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg border border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800"
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <svg className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 002 2v8a2 2 0 002 2z" />
                </svg>
                <span className="font-mono text-xs font-semibold text-ink-800 dark:text-ink-100 truncate">
                  {d.name}
                </span>
              </div>
              <div className="text-[10px] text-ink-500 dark:text-ink-400 mt-0.5 ml-5.5">
                {d.rowCount} row{d.rowCount === 1 ? '' : 's'} · {d.columns.length} col{d.columns.length === 1 ? '' : 's'}
              </div>
            </div>
            <button
              onClick={() => onRemove(d.name)}
              className="text-ink-400 hover:text-rose-500 transition flex-shrink-0"
              type="button"
              aria-label={`Remove ${d.name}`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}