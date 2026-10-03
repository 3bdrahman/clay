import { useState } from 'react';
import { useAppStore } from '../store';
import { deriveDataQueries, deriveDocumentQueries } from '../lib/exampleQueries';
import { getWebSearchAvailability } from '../lib/websearch';

const CAPABILITIES = [
  {
    title: 'Vector Search',
    description: 'Find relevant passages in uploaded documents.',
  },
  {
    title: 'Data Analysis',
    description: 'Query CSV tables with generated Arquero code.',
  },
  {
    title: 'Optional Web Search',
    description: 'Use live results only when a provider is available.',
  },
];

const WEB_QUERIES = [
  'What does independent web indexing mean for search quality?',
  'Find practical RAG evaluation methods',
  'Compare open-source vector databases',
];

export function LandingHero({
  onLoadSample,
  onAddData,
  onExampleSelect,
  onOpenSettings,
}: {
  onLoadSample: () => void | Promise<void>;
  onAddData: () => void;
  onExampleSelect?: (q: string) => void;
  onOpenSettings?: () => void;
}) {
  const sandboxDatasets = useAppStore(s => s.sandboxDatasets);
  const sandboxDocuments = useAppStore(s => s.sandboxDocuments);
  const settings = useAppStore(s => s.settings);
  const [sampleState, setSampleState] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [sampleError, setSampleError] = useState<string | null>(null);
  const hasData = sandboxDatasets.length > 0 || sandboxDocuments.length > 0;
  const dataQueries = deriveDataQueries(sandboxDatasets);
  const docQueries = deriveDocumentQueries(sandboxDocuments);
  const webAvailability = getWebSearchAvailability(settings);

  const handleLoadSample = async () => {
    setSampleState('loading');
    setSampleError(null);
    try {
      await onLoadSample();
      setSampleState('success');
    } catch (e) {
      setSampleState('error');
      setSampleError(e instanceof Error ? e.message : String(e));
    }
  };

  const exampleGroups: Array<{ category: string; queries: string[] }> = [];
  if (dataQueries.length > 0) exampleGroups.push({ category: 'Data', queries: dataQueries });
  if (docQueries.length > 0) exampleGroups.push({ category: 'Documents', queries: docQueries });
  if (webAvailability.available) exampleGroups.push({ category: 'Web', queries: WEB_QUERIES });

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <section className="px-4 py-8 sm:py-10 max-w-5xl mx-auto w-full">
        <div className="text-center mb-6">
          <h1 className="text-3xl sm:text-4xl font-bold text-ink-900 dark:text-ink-50 tracking-tight mb-3">
            Ask Questions About Your Data
          </h1>
          <p className="text-base sm:text-lg text-ink-600 dark:text-ink-300 max-w-2xl mx-auto mb-5 leading-relaxed">
            Clay combines vector search, in-browser data analysis, and optional web search in one chat workspace.
            Load sample data or your own files now. Configure an API key or local model before asking questions.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            {sandboxDatasets.length === 0 && (
              <button
                onClick={handleLoadSample}
                disabled={sampleState === 'loading'}
                className="inline-flex items-center justify-center gap-2 px-5 py-3 bg-brand-600 hover:bg-brand-700 text-white font-medium rounded-lg transition-colors text-base disabled:opacity-60"
              >
                {sampleState === 'loading' ? (
                  <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.25" />
                    <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2} aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                  </svg>
                )}
                {sampleState === 'loading' ? 'Loading sample data' : 'Load Sample Data'}
              </button>
            )}
            <button
              onClick={onAddData}
              className="inline-flex items-center justify-center gap-2 px-5 py-3 border-2 border-ink-200 dark:border-ink-700 hover:bg-ink-50 dark:hover:bg-ink-800 text-ink-700 dark:text-ink-200 font-medium rounded-lg transition-colors text-base"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Add Your Own Data
            </button>
            {onOpenSettings && (
              <button
                type="button"
                onClick={onOpenSettings}
                className="inline-flex items-center justify-center gap-2 px-5 py-3 border-2 border-brand-200 dark:border-brand-800 hover:bg-brand-50 dark:hover:bg-brand-900/30 text-brand-700 dark:text-brand-300 font-medium rounded-lg transition-colors text-base"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2} aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h3m-6 4h9m-12 4h15m-13 4h11" />
                </svg>
                Configure Model
              </button>
            )}
          </div>
          <div className="mt-4 space-y-2" aria-live="polite">
            {sampleState === 'success' && (
              <p className="text-sm text-emerald-700 dark:text-emerald-300">Sample data loaded. Configure a model, then ask about the datasets.</p>
            )}
            {sampleState === 'error' && sampleError && (
              <p className="text-sm text-rose-600 dark:text-rose-400" role="alert">{sampleError}</p>
            )}
            <p className="text-xs text-ink-500 dark:text-ink-400 max-w-2xl mx-auto">
              The LLM receives your question plus relevant document passages and dataset context selected for the answer.
            </p>
          </div>
        </div>

        {hasData && (
          <div className="mb-7">
            <h2 className="text-xl font-bold text-ink-900 dark:text-ink-50 text-center mb-4">
              Try These Questions
            </h2>
            <div className={`grid grid-cols-1 ${exampleGroups.length === 3 ? 'sm:grid-cols-3' : exampleGroups.length === 2 ? 'sm:grid-cols-2' : ''} gap-3 max-w-4xl mx-auto`}>
              {exampleGroups.map(group => (
                <div key={group.category} className="p-3 rounded-lg border border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-2">{group.category}</h3>
                  <div className="space-y-2">
                    {group.queries.map(q => (
                      <button
                        key={q}
                        onClick={() => onExampleSelect?.(q)}
                        className="w-full text-left px-3 py-2 text-sm text-ink-700 dark:text-ink-200 bg-ink-50 dark:bg-ink-700 rounded-lg hover:bg-brand-50 dark:hover:bg-brand-900/30 hover:border-brand-300 border transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-5">
          {CAPABILITIES.map(capability => (
            <div key={capability.title} className="rounded-lg border border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">{capability.title}</h3>
              <p className="mt-1 text-xs text-ink-600 dark:text-ink-300 leading-relaxed">{capability.description}</p>
            </div>
          ))}
        </div>

        {!webAvailability.available && (
          <div className="max-w-3xl mx-auto rounded-lg border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-900/20 px-4 py-3 text-sm text-amber-900 dark:text-amber-100 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <span>{webAvailability.message}</span>
            {onOpenSettings && (
              <button
                type="button"
                data-testid="landing-configure-web"
                onClick={onOpenSettings}
                className="text-xs font-semibold text-amber-900 dark:text-amber-100 underline underline-offset-2 self-start sm:self-auto"
              >
                Configure web search
              </button>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
