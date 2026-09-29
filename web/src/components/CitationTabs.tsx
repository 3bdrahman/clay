// CitationTabs — tab button, badge, and tab panel components for CitationPanel

import type { Citation, DataAnalysisResult, Document, WebResult } from '../lib/types';

function TabBtn({
  role,
  id,
  ariaControls,
  ariaSelected,
  active,
  onClick,
  disabled,
  children,
}: {
  role: 'tab';
  id: string;
  ariaControls: string;
  ariaSelected: boolean;
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      role={role}
      id={id}
      aria-controls={ariaControls}
      aria-selected={ariaSelected}
      aria-disabled={disabled}
      onClick={onClick}
      disabled={disabled}
      tabIndex={active ? 0 : -1}
      className={`px-3 py-2 text-xs font-medium whitespace-nowrap border-b-2 transition flex items-center gap-1.5 ${
        active
          ? 'border-brand-500 text-brand-600 dark:text-brand-400'
          : 'border-transparent text-ink-500 dark:text-ink-400 hover:text-ink-700 dark:hover:text-ink-200'
      } ${disabled ? 'opacity-30 cursor-not-allowed' : ''}`}
    >
      {children}
    </button>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[10px] bg-ink-100 dark:bg-ink-800 text-ink-600 dark:text-ink-300 px-1.5 py-0.5 rounded-full" aria-hidden="true">
      {children}
    </span>
  );
}

function Empty({ msg }: { msg: string }) {
  return <div className="text-center text-sm text-ink-400 dark:text-ink-500 py-6" role="status">{msg}</div>;
}

function DocsTab({
  id,
  role,
  ariaLabelledby,
  documents,
}: {
  id: string;
  role: 'tabpanel';
  ariaLabelledby: string;
  documents: Document[];
}) {
  if (documents.length === 0) return <Empty msg="No documents retrieved" />;
  return (
    <div id={id} role={role} aria-labelledby={ariaLabelledby} className="space-y-2">
      {documents.map((doc, i) => (
        <div key={doc.id || i} className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800">
          <div className="flex items-center justify-between gap-2 mb-1.5">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-brand-600 dark:text-brand-400">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span className="truncate">{doc.source}</span>
              {doc.page && <span className="text-ink-400 font-normal">p. {doc.page}</span>}
            </div>
            {doc.score !== undefined && (
              <span className="text-[10px] text-ink-400 font-mono">{doc.score.toFixed(3)}</span>
            )}
          </div>
          <p className="text-xs text-ink-700 dark:text-ink-300 leading-relaxed line-clamp-4">
            {doc.content.slice(0, 400)}
            {doc.content.length > 400 ? '…' : ''}
          </p>
        </div>
      ))}
    </div>
  );
}

function WebTab({
  id,
  role,
  ariaLabelledby,
  results,
}: {
  id: string;
  role: 'tabpanel';
  ariaLabelledby: string;
  results: WebResult[];
}) {
  if (results.length === 0) return <Empty msg="No web results" />;
  return (
    <div id={id} role={role} aria-labelledby={ariaLabelledby} className="space-y-2">
      {results.map((r, i) => (
        <div key={i} className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-brand-600 dark:text-brand-400 mb-1.5">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
            <span className="truncate">{r.title}</span>
          </div>
          <p className="text-xs text-ink-700 dark:text-ink-300 leading-relaxed line-clamp-4">{r.content}</p>
          {r.url && (
            <a href={r.url} target="_blank" rel="noreferrer" className="text-[10px] text-brand-500 hover:underline mt-1 block truncate">
              {r.url}
            </a>
          )}
        </div>
      ))}
    </div>
  );
}

function AnalysisTab({
  id,
  role,
  ariaLabelledby,
  analysis,
}: {
  id: string;
  role: 'tabpanel';
  ariaLabelledby: string;
  analysis: DataAnalysisResult;
}) {
  return (
    <div id={id} role={role} aria-labelledby={ariaLabelledby} className="space-y-3">
      <div className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400 mb-2">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
          <span>Generated Code (Arquero)</span>
        </div>
        <pre className="text-[11px] font-mono bg-ink-50 dark:bg-ink-900 p-2 rounded overflow-x-auto leading-relaxed">
          <code>{analysis.code}</code>
        </pre>
      </div>

      <div className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800">
        <div className="text-xs font-semibold text-ink-500 uppercase tracking-wide mb-2">Explanation</div>
        <p className="text-xs text-ink-700 dark:text-ink-300 leading-relaxed">{analysis.explanation}</p>
      </div>

      {analysis.insights?.length && (
        <div className="space-y-2">
          <div className="text-xs font-semibold text-ink-500 uppercase tracking-wide mb-2">Insights</div>
          {analysis.insights.map((insight, i) => (
            <div
              key={i}
              className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-semibold text-ink-900 dark:text-ink-100">{insight.finding}</p>
                  <p className="text-xs text-ink-500 dark:text-ink-400 mt-1">{insight.evidence}</p>
                  {insight.implication && (
                    <p className="text-xs text-ink-500 dark:text-ink-400 italic mt-1">{insight.implication}</p>
                  )}
                </div>
                <span
                  className={`text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded whitespace-nowrap shrink-0 ${
                    insight.confidence === 'high'
                      ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                      : insight.confidence === 'medium'
                      ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                      : 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400'
                  }`}
                >
                  {insight.confidence.toUpperCase()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {analysis.resultType !== 'error' && (
        <div className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800">
          <div className="text-xs font-semibold text-ink-500 uppercase tracking-wide mb-2">Result</div>
          <ResultRenderer result={analysis.result} chartConfig={analysis.chartConfig} />
        </div>
      )}

      {analysis.resultType === 'error' && (
        <div className="border border-rose-300 dark:border-rose-700 rounded-lg p-3 bg-rose-50 dark:bg-rose-900/20" role="alert">
          <div className="text-xs font-semibold text-rose-700 dark:text-rose-300 mb-1">Error</div>
          <pre className="text-xs font-mono text-rose-700 dark:text-rose-300 whitespace-pre-wrap">
            {String(analysis.result)}
          </pre>
        </div>
      )}

      <div className="text-[10px] text-ink-400 flex justify-between">
        <span>
          {analysis.attempts} attempt{analysis.attempts > 1 ? 's' : ''}
        </span>
        <span>{Math.round(analysis.durationMs)}ms</span>
      </div>
    </div>
  );
}

function CitationsTab({
  id,
  role,
  ariaLabelledby,
  citations,
}: {
  id: string;
  role: 'tabpanel';
  ariaLabelledby: string;
  citations: Citation[];
}) {
  if (citations.length === 0) return <Empty msg="No inline citations" />;
  return (
    <div id={id} role={role} aria-labelledby={ariaLabelledby} className="space-y-1.5">
      {citations.map((c, i) => (
        <div key={i} className="flex gap-2 text-xs border-l-2 border-brand-400 pl-2 py-1">
          <span className="font-mono text-ink-400">[{i + 1}]</span>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-ink-700 dark:text-ink-300 truncate">
              <span>{c.source}</span>
              {c.page && <span className="text-ink-400 font-normal"> — p.{c.page}</span>}
              <span className="ml-1.5 text-[10px] uppercase text-ink-400 font-semibold">{c.type}</span>
            </div>
            <p className="text-ink-500 dark:text-ink-400 line-clamp-2 mt-0.5">{c.excerpt}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

// ResultRenderer is imported from ResultRenderer.tsx to avoid circular dependency
import { ResultRenderer } from './ResultRenderer';

export { TabBtn, Badge, DocsTab, WebTab, AnalysisTab, CitationsTab, Empty, ResultRenderer };