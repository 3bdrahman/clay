import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { CitationPanel } from './CitationPanel';
import type { DataAnalysisResult, Insight } from '../lib/types';

const container = document.createElement('div');
let root: ReturnType<typeof createRoot>;

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function renderPanel(analysis?: DataAnalysisResult) {
  document.body.appendChild(container);
  root = createRoot(container);
  act(() =>
    root.render(
      <CitationPanel
        documents={[]}
        webResults={[]}
        analysis={analysis}
        citations={[]}
      />
    )
  );
  return container;
}

function clickAnalysisTab(container: HTMLElement) {
  const tabBtn = container.querySelector<HTMLButtonElement>('[role="tab"][aria-controls$="-panel-analysis"]');
  if (tabBtn) {
    act(() => tabBtn.click());
  }
}

describe('CitationPanel AnalysisTab insights', () => {
  const baseAnalysis: DataAnalysisResult = {
    type: 'data_analysis',
    question: 'Test question',
    code: 'const result = table.groupby("col").count()',
    explanation: 'Test explanation',
    resultType: 'table',
    result: [{ col: 'A', count: 1 }],
    attempts: 1,
    durationMs: 100,
    timestamp: Date.now(),
  };

  it('renders insights with confidence badges', () => {
    const insights: Insight[] = [
      {
        finding: 'High confidence finding',
        evidence: 'Strong evidence supports this',
        confidence: 'high',
        implication: 'This implies something important',
      },
      {
        finding: 'Low confidence finding',
        evidence: 'Weak evidence',
        confidence: 'low',
      },
    ];

    renderPanel({ ...baseAnalysis, insights });
    clickAnalysisTab(container);

    // Both findings should appear
    expect(container.textContent).toContain('High confidence finding');
    expect(container.textContent).toContain('Low confidence finding');

    // Confidence badges should render (emerald for high, rose for low)
    expect(container.textContent).toContain('HIGH');
    expect(container.textContent).toContain('LOW');
  });

  it('empty insights render no section', () => {
    renderPanel({ ...baseAnalysis, insights: [] });
    clickAnalysisTab(container);

    // Explanation and Result should still render
    expect(container.textContent).toContain('Explanation');
    expect(container.textContent).toContain('Test explanation');
    expect(container.textContent).toContain('Result');

    // But no Insights section should appear
    expect(container.textContent).not.toContain('Insights');
  });
});