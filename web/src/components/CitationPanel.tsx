// CitationPanel — shows retrieved documents, web results, and analysis results

import { useState, useId } from 'react';
import type { Citation, DataAnalysisResult, Document, WebResult } from '../lib/types';

import { TabBtn, Badge, DocsTab, WebTab, AnalysisTab, CitationsTab, ResultRenderer } from './CitationTabs';

interface Props {
  documents: Document[];
  webResults: WebResult[];
  analysis?: DataAnalysisResult;
  citations: Citation[];
}

type Tab = 'docs' | 'web' | 'analysis' | 'citations';

export function CitationPanel({ documents, webResults, analysis, citations }: Props) {
  const docCount = documents.length;
  const webCount = webResults.length;
  const hasAnalysis = !!analysis && analysis.resultType !== 'error';
  const hasErrorAnalysis = !!analysis && analysis.resultType === 'error';
  const analysisVisible = hasAnalysis || hasErrorAnalysis;
  const defaultTab: Tab = docCount > 0 ? 'docs' : webCount > 0 ? 'web' : analysisVisible ? 'analysis' : 'citations';
  const [tab, setTab] = useState<Tab>(defaultTab);
  const tablistId = useId();

  if (docCount + webCount + (analysis ? 1 : 0) === 0) {
    return (
      <div className="text-center text-sm text-ink-400 dark:text-ink-500 py-8" role="status">
        No sources for this query.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div role="tablist" aria-label="Source types" className="flex gap-1 border-b border-ink-200 dark:border-ink-700 overflow-x-auto" id={tablistId}>
        <TabBtn
          role="tab"
          id={`${tablistId}-tab-docs`}
          ariaControls={`${tablistId}-panel-docs`}
          ariaSelected={tab === 'docs'}
          active={tab === 'docs'}
          onClick={() => setTab('docs')}
          disabled={docCount === 0}
        >
          <span>Documents</span>
          <Badge>{docCount}</Badge>
        </TabBtn>
        <TabBtn
          role="tab"
          id={`${tablistId}-tab-web`}
          ariaControls={`${tablistId}-panel-web`}
          ariaSelected={tab === 'web'}
          active={tab === 'web'}
          onClick={() => setTab('web')}
          disabled={webCount === 0}
        >
          <span>Web</span>
          <Badge>{webCount}</Badge>
        </TabBtn>
        {(hasAnalysis || hasErrorAnalysis) && (
          <TabBtn
            role="tab"
            id={`${tablistId}-tab-analysis`}
            ariaControls={`${tablistId}-panel-analysis`}
            ariaSelected={tab === 'analysis'}
            active={tab === 'analysis'}
            onClick={() => setTab('analysis')}
          >
            <span>Analysis</span>
          </TabBtn>
        )}
        <TabBtn
          role="tab"
          id={`${tablistId}-tab-citations`}
          ariaControls={`${tablistId}-panel-citations`}
          ariaSelected={tab === 'citations'}
          active={tab === 'citations'}
          onClick={() => setTab('citations')}
          disabled={citations.length === 0}
        >
          <span>Citations</span>
          <Badge>{citations.length}</Badge>
        </TabBtn>
      </div>

      {tab === 'docs' && (
        <DocsTab
          id={`${tablistId}-panel-docs`}
          role="tabpanel"
          ariaLabelledby={`${tablistId}-tab-docs`}
          documents={documents}
        />
      )}
      {tab === 'web' && (
        <WebTab
          id={`${tablistId}-panel-web`}
          role="tabpanel"
          ariaLabelledby={`${tablistId}-tab-web`}
          results={webResults}
        />
      )}
      {tab === 'analysis' && analysis && (
        <AnalysisTab
          id={`${tablistId}-panel-analysis`}
          role="tabpanel"
          ariaLabelledby={`${tablistId}-tab-analysis`}
          analysis={analysis}
        />
      )}
      {tab === 'citations' && (
        <CitationsTab
          id={`${tablistId}-panel-citations`}
          role="tabpanel"
          ariaLabelledby={`${tablistId}-tab-citations`}
          citations={citations}
        />
      )}
    </div>
  );
}

// Re-export internal components for backward compatibility
export { TabBtn, Badge, DocsTab, WebTab, AnalysisTab, CitationsTab, ResultRenderer };