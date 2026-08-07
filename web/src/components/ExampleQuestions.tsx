// ExampleQuestions — quick-start prompts showcasing each path

import { useAppStore } from '../store';

interface Props {
  onSelect: (q: string) => void;
}

const DOC_QUESTIONS = [
  'Summarize the key points from my documents',
  'What are the main themes across my files?',
  'Find any action items mentioned',
];

const DATA_QUESTIONS = [
  'How many employees do we have by department?',
  'Show me the distribution of project statuses',
  'What is the total budget across all projects?',
  'Average rating from client feedback',
];

const WEB_QUESTIONS = [
  'What are the latest trends in AI for business?',
  'Compare AWS vs Azure for ML workloads',
  'What are the best practices for vector search?',
];

export function ExampleQuestions({ onSelect }: Props) {
  const sandboxDatasets = useAppStore(s => s.sandboxDatasets);
  const sandboxDocuments = useAppStore(s => s.sandboxDocuments);
  const hasData = sandboxDatasets.length > 0;
  const hasDocs = sandboxDocuments.length > 0;

  return (
    <div className="space-y-5 max-w-3xl">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-ink-900 dark:text-ink-50">Try Clay</h2>
        <p className="text-sm text-ink-500 dark:text-ink-400 mt-1">
          Click any example below, or ask your own question above.
        </p>
      </div>

      {hasData && (
        <Group
          category="Data Analysis (Arquero)"
          color="text-emerald-600 bg-emerald-50 dark:bg-emerald-900/30 dark:text-emerald-400"
          icon="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
          questions={DATA_QUESTIONS}
          onSelect={onSelect}
        />
      )}

      {hasDocs ? (
        <Group
          category="Documents (Vector Store)"
          color="text-brand-600 bg-brand-50 dark:bg-brand-900/30 dark:text-brand-400"
          icon="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          questions={DOC_QUESTIONS}
          onSelect={onSelect}
        />
      ) : (
        <Group
          category="Documents (upload a PDF/MD/TXT to enable)"
          color="text-ink-500 bg-ink-100 dark:bg-ink-800 dark:text-ink-400"
          icon="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          questions={DOC_QUESTIONS}
          onSelect={onSelect}
          disabled
        />
      )}

      <Group
        category="Web Search"
        color="text-amber-600 bg-amber-50 dark:bg-amber-900/30 dark:text-amber-400"
        icon="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9"
        questions={WEB_QUESTIONS}
        onSelect={onSelect}
      />
    </div>
  );
}

function Group({
  category,
  icon,
  color,
  questions,
  onSelect,
  disabled,
}: {
  category: string;
  icon: string;
  color: string;
  questions: string[];
  onSelect: (q: string) => void;
  disabled?: boolean;
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-2.5">
        <span className={`w-7 h-7 rounded-lg flex items-center justify-center ${color}`} aria-hidden="true">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2} aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d={icon} />
          </svg>
        </span>
        <h3 className="text-sm font-semibold text-ink-700 dark:text-ink-200">{category}</h3>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2" role="list" aria-label={category}>
        {questions.map((q, i) => (
          <button
            key={i}
            onClick={() => onSelect(q)}
            disabled={disabled}
            role="listitem"
            aria-disabled={disabled}
            className="text-left px-3 py-2.5 bg-white dark:bg-ink-800 border border-ink-200 dark:border-ink-700 rounded-lg hover:border-brand-400 hover:shadow-sm transition text-xs text-ink-700 dark:text-ink-200 leading-relaxed disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2 dark:focus:ring-offset-ink-900"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
