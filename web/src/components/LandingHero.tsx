import { useAppStore } from '../store';

const FEATURES = [
  {
    icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z',
    title: 'Vector Search',
    description: 'Upload PDFs, Markdown, or text files. Clay chunks, embeds, and retrieves relevant passages for grounded answers.',
    color: 'text-brand-600 bg-brand-50 dark:bg-brand-900/30 dark:text-brand-400',
  },
  {
    icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
    title: 'Data Analysis',
    description: 'Drop CSV files and query them with natural language. Clay generates Arquero code to filter, aggregate, join, and chart your data.',
    color: 'text-emerald-600 bg-emerald-50 dark:bg-emerald-900/30 dark:text-emerald-400',
  },
  {
    icon: 'M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9',
    title: 'Web Search',
    description: 'Access current information from the web. DuckDuckGo (no key) or Serper (Google) for up-to-date facts and general knowledge.',
    color: 'text-amber-600 bg-amber-50 dark:bg-amber-900/30 dark:text-amber-400',
  },
  {
    icon: 'M13 10V3L4 14h7v7l9-11h-7z',
    title: 'Self-Correcting Loop',
    description: 'LLM-as-judge evaluates every answer for hallucinations and relevance. Automatically retries with different sources if quality is low.',
    color: 'text-purple-600 bg-purple-50 dark:bg-purple-900/30 dark:text-purple-400',
  },
{
    icon: 'M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z',
    title: 'Runs in Your Browser',
    description: 'No backend server. Your data stays local; only queries go to your LLM provider. Deploy anywhere as static files.',
    color: 'text-rose-600 bg-rose-50 dark:bg-rose-900/30 dark:text-rose-400',
  },
  {
    icon: 'M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z',
    title: 'Dynamic Model Picker',
    description: 'Fetches NIM\'s live catalog (~100 models) and picks the best per task: routing, code generation, answer, evaluation, embeddings.',
    color: 'text-indigo-600 bg-indigo-50 dark:bg-indigo-900/30 dark:text-indigo-400',
  },
];

const EXAMPLE_QUERIES = [
  { category: 'Data', queries: ['Average salary by department', 'Project count by status', 'Total budget across projects'] },
  { category: 'Documents', queries: ['Summarize my documents', 'Find action items', 'Key themes across files'] },
  { category: 'Web', queries: ['Latest AI trends for business', 'Best practices for RAG', 'Compare cloud ML platforms'] },
];

export function LandingHero({
  onLoadSample,
  onAddData,
  onExampleSelect,
}: {
  onLoadSample: () => void;
  onAddData: () => void;
  onExampleSelect?: (q: string) => void;
}) {
  const sandboxDatasets = useAppStore(s => s.sandboxDatasets);
  const sandboxDocuments = useAppStore(s => s.sandboxDocuments);
  const hasData = sandboxDatasets.length > 0 || sandboxDocuments.length > 0;

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Hero Section */}
      <section className="px-4 py-10 sm:py-16 max-w-5xl mx-auto w-full">
        <div className="text-center mb-10">
          <h1 className="text-4xl sm:text-5xl font-bold text-ink-900 dark:text-ink-50 tracking-tight mb-4">
            Ask Questions About{' '}
            <span className="bg-gradient-to-r from-brand-600 to-purple-600 bg-clip-text text-transparent">
              Your Data
            </span>
          </h1>
          <p className="text-lg sm:text-xl text-ink-600 dark:text-ink-300 max-w-2xl mx-auto mb-8 leading-relaxed">
            Clay combines vector search, in-browser data analysis, and web search into a single chat interface.
            Drop your files, ask naturally, and watch the reasoning pipeline unfold in real time.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <button
              onClick={onLoadSample}
              className="inline-flex items-center gap-2 px-6 py-3 bg-brand-600 hover:bg-brand-700 text-white font-medium rounded-lg transition-colors text-base"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
              </svg>
              Load Sample Data & Try It
            </button>
            <button
              onClick={onAddData}
              className="inline-flex items-center gap-2 px-6 py-3 border-2 border-ink-200 dark:border-ink-700 hover:bg-ink-50 dark:hover:bg-ink-800 text-ink-700 dark:text-ink-200 font-medium rounded-lg transition-colors text-base"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Add Your Own Data
            </button>
          </div>
        </div>

        {/* Features Grid */}
        <div className="mb-10">
          <h2 className="text-2xl font-bold text-ink-900 dark:text-ink-50 text-center mb-8">
            How Clay Works
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {FEATURES.map((feature, i) => (
              <div
                key={feature.title}
                className="group p-5 rounded-xl border border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800 hover:border-brand-300 dark:hover:border-brand-700 hover:shadow-lg transition-all duration-300"
                style={{ animationDelay: `${i * 100}ms` }}
              >
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${feature.color}`}>
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={feature.icon} />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-ink-900 dark:text-ink-50 mb-2">{feature.title}</h3>
                <p className="text-sm text-ink-600 dark:text-ink-300 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Example Queries */}
        {hasData && (
          <div className="mb-10">
            <h2 className="text-2xl font-bold text-ink-900 dark:text-ink-50 text-center mb-8">
              Try These Questions
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-4xl mx-auto">
              {EXAMPLE_QUERIES.map((group) => (
                <div key={group.category} className="p-4 rounded-xl border border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800">
                  <h3 className="text-sm font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400 mb-3">{group.category}</h3>
                  <div className="space-y-2">
                    {group.queries.map((q) => (
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
      </section>
    </div>
  );
}