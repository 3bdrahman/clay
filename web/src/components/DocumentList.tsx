import { SandboxDocument } from '../store';

interface DocumentListProps {
  documents: SandboxDocument[];
  onRemove: (fileName: string) => void;
}

export function DocumentList({ documents, onRemove }: DocumentListProps) {
  if (documents.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
          Documents
        </div>
        <span className="text-[10px] text-ink-400">{documents.length}</span>
      </div>
      <ul className="space-y-1.5">
        {documents.map(d => (
          <li
            key={d.id}
            className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg border border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800"
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <svg className="w-3.5 h-3.5 text-brand-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span className="font-mono text-xs font-semibold text-ink-800 dark:text-ink-100 truncate">
                  {d.fileName}
                </span>
              </div>
              <div className="text-[10px] text-ink-500 dark:text-ink-400 mt-0.5 ml-5.5">
                {d.chunkCount} chunk{d.chunkCount === 1 ? '' : 's'} embedded
              </div>
            </div>
            <button
              onClick={() => onRemove(d.fileName)}
              className="text-ink-400 hover:text-rose-500 transition flex-shrink-0"
              type="button"
              aria-label={`Remove ${d.fileName}`}
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