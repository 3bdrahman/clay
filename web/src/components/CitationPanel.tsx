// CitationPanel — shows retrieved documents, web results, and analysis results

import { lazy, Suspense, useState } from 'react';
import type { Citation, DataAnalysisResult, Document, WebResult } from '../lib/types';

const ChartRenderer = lazy(() => import('./ChartRenderer'));

interface Props {
  documents: Document[];
  webResults: WebResult[];
  analysis?: DataAnalysisResult;
  citations: Citation[];
}

type Tab = 'docs' | 'web' | 'analysis' | 'citations';

export function CitationPanel({ documents, webResults, analysis, citations }: Props) {
  const [tab, setTab] = useState<Tab>('docs');

  const docCount = documents.length;
  const webCount = webResults.length;
  const hasAnalysis = !!analysis && analysis.resultType !== 'error';
  const hasErrorAnalysis = !!analysis && analysis.resultType === 'error';

  if (docCount + webCount + (analysis ? 1 : 0) === 0) {
    return (
      <div className="text-center text-sm text-ink-400 dark:text-ink-500 py-8">
        No sources for this query.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-1 border-b border-ink-200 dark:border-ink-700 overflow-x-auto">
        <TabBtn active={tab === 'docs'} onClick={() => setTab('docs')} disabled={docCount === 0}>
          <span>Documents</span>
          <Badge>{docCount}</Badge>
        </TabBtn>
        <TabBtn active={tab === 'web'} onClick={() => setTab('web')} disabled={webCount === 0}>
          <span>Web</span>
          <Badge>{webCount}</Badge>
        </TabBtn>
        {(hasAnalysis || hasErrorAnalysis) && (
          <TabBtn active={tab === 'analysis'} onClick={() => setTab('analysis')}>
            <span>Analysis</span>
          </TabBtn>
        )}
        <TabBtn active={tab === 'citations'} onClick={() => setTab('citations')} disabled={citations.length === 0}>
          <span>Citations</span>
          <Badge>{citations.length}</Badge>
        </TabBtn>
      </div>

      {tab === 'docs' && <DocsTab documents={documents} />}
      {tab === 'web' && <WebTab results={webResults} />}
      {tab === 'analysis' && analysis && <AnalysisTab analysis={analysis} />}
      {tab === 'citations' && <CitationsTab citations={citations} />}
    </div>
  );
}

function TabBtn({
  active,
  onClick,
  disabled,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
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
    <span className="text-[10px] bg-ink-100 dark:bg-ink-800 text-ink-600 dark:text-ink-300 px-1.5 py-0.5 rounded-full">
      {children}
    </span>
  );
}

function DocsTab({ documents }: { documents: Document[] }) {
  if (documents.length === 0) return <Empty msg="No documents retrieved" />;
  return (
    <div className="space-y-2">
      {documents.map((doc, i) => (
        <div key={doc.id || i} className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800">
          <div className="flex items-center justify-between gap-2 mb-1.5">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-brand-600 dark:text-brand-400">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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

function WebTab({ results }: { results: WebResult[] }) {
  if (results.length === 0) return <Empty msg="No web results" />;
  return (
    <div className="space-y-2">
      {results.map((r, i) => (
        <div key={i} className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-brand-600 dark:text-brand-400 mb-1.5">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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

function AnalysisTab({ analysis }: { analysis: DataAnalysisResult }) {
  return (
    <div className="space-y-3">
      <div className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400 mb-2">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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

      {analysis.resultType !== 'error' && (
        <div className="border border-ink-200 dark:border-ink-700 rounded-lg p-3 bg-white dark:bg-ink-800">
          <div className="text-xs font-semibold text-ink-500 uppercase tracking-wide mb-2">Result</div>
          <ResultRenderer result={analysis.result} chartConfig={analysis.chartConfig} />
        </div>
      )}

      {analysis.resultType === 'error' && (
        <div className="border border-rose-300 dark:border-rose-700 rounded-lg p-3 bg-rose-50 dark:bg-rose-900/20">
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

function CitationsTab({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return <Empty msg="No inline citations" />;
  return (
    <div className="space-y-1.5">
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

function Empty({ msg }: { msg: string }) {
  return <div className="text-center text-sm text-ink-400 dark:text-ink-500 py-6">{msg}</div>;
}

function ResultRenderer({ result, chartConfig }: { result: unknown; chartConfig?: DataAnalysisResult['chartConfig'] }) {
  if (chartConfig) {
    return (
      <Suspense fallback={<div className="text-xs text-ink-400">Loading chart…</div>}>
        <ChartRenderer config={chartConfig} />
      </Suspense>
    );
  }
  if (Array.isArray(result)) {
    return (
      <div className="overflow-x-auto">
        <table className="text-[11px] w-full">
          <tbody>
            {result.slice(0, 20).map((row, i) => (
              <tr key={i} className="border-b border-ink-100 dark:border-ink-700">
                {typeof row === 'object' && row !== null ? (
                  Object.entries(row as Record<string, unknown>).map(([k, v]) => (
                    <td key={k} className="px-2 py-1 font-mono">
                      <span className="text-ink-500">{k}:</span> {String(v)}
                    </td>
                  ))
                ) : (
                  <td className="px-2 py-1 font-mono">{String(row)}</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
        {result.length > 20 && <div className="text-[10px] text-ink-400 mt-1">+ {result.length - 20} more rows</div>}
      </div>
    );
  }
  if (result && typeof result === 'object') {
    return (
      <pre className="text-[11px] font-mono bg-ink-50 dark:bg-ink-900 p-2 rounded overflow-x-auto">
        {JSON.stringify(result, null, 2)}
      </pre>
    );
  }
  return <div className="text-sm font-mono">{String(result)}</div>;
}
