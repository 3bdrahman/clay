import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { WorkflowGraph } from './WorkflowGraph';

const container = document.createElement('div');
let root: ReturnType<typeof createRoot>;

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function renderWorkflowGraph(steps: Array<{ id: string; node: string; label: string; status: 'pending' | 'running' | 'done' | 'error' | 'skipped'; durationMs?: number; detail?: string }>) {
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root.render(<WorkflowGraph steps={steps as any} />));
  return container;
}

function getLabels(container: HTMLElement): string[] {
  // The label is in a div with class "text-sm font-medium truncate"
  return Array.from(container.querySelectorAll('.text-sm.font-medium.truncate')).map(el => el.textContent || '');
}

describe('WorkflowGraph tool-call sub-step labels', () => {
  it('renders tool-call sub-step labels', () => {
    const steps = [
      { id: 'analyze:profile_column-1', node: 'analyze:profile_column', label: 'profile_column', status: 'done' as const },
    ];
    const container = renderWorkflowGraph(steps);
    const labels = getLabels(container);
    // The label 'profile_column' should appear, NOT the raw nodeId 'analyze:profile_column'
    expect(labels).toContain('profile_column');
    expect(labels).not.toContain('analyze:profile_column');
  });

  it('renders multiple distinct tool nodes in order', () => {
    const steps = [
      { id: 'analyze:profile_column-1', node: 'analyze:profile_column', label: 'profile_column', status: 'done' as const },
      { id: 'analyze:correlate-1', node: 'analyze:correlate', label: 'correlate', status: 'done' as const },
    ];
    const container = renderWorkflowGraph(steps);
    const labels = getLabels(container);
    // Both labels should appear
    expect(labels).toContain('profile_column');
    expect(labels).toContain('correlate');
    // And in order (profile_column before correlate)
    expect(labels.indexOf('profile_column')).toBeLessThan(labels.indexOf('correlate'));
  });

  it('unknown node without a step label falls back to nodeId', () => {
    const steps = [
      { id: 'mystery_node', node: 'mystery_node', label: 'mystery_node', status: 'done' as const },
    ];
    const container = renderWorkflowGraph(steps);
    const labels = getLabels(container);
    // The nodeId 'mystery_node' should render (label equals nodeId here)
    expect(labels).toContain('mystery_node');
  });
});