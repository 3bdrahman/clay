// WorkflowGraph — animated visualization of the workflow state machine

import type { StepTrace } from '../lib/types';

interface NodeDef {
  id: string;
  label: string;
  description: string;
  icon: string;
}

const NODE_DEFS: Record<string, NodeDef> = {
  start: { id: 'start', label: 'Question', description: 'Your input', icon: 'M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z' },
  route: { id: 'route', label: 'Router', description: 'Classify intent', icon: 'M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4' },
  retrieve: { id: 'retrieve', label: 'Vector DB', description: 'Retrieve documents', icon: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4' },
  grade_docs: { id: 'grade_docs', label: 'Doc Grader', description: 'Filter relevant docs', icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z' },
  analyze: { id: 'analyze', label: 'Data Analysis', description: 'Run Arquero query', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  web_search: { id: 'web_search', label: 'Web Search', description: 'External knowledge', icon: 'M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3 m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9' },
  generate: { id: 'generate', label: 'Generate', description: 'Compose answer', icon: 'M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z' },
  evaluate: { id: 'evaluate', label: 'Quality Check', description: 'Verify groundedness', icon: 'M5 13l4 4L19 7' },
  decide: { id: 'decide', label: 'Re-routing', description: 'Retry with different source', icon: 'M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4' },
  end: { id: 'end', label: 'Answer', description: 'Final response', icon: 'M5 13l4 4L19 7' },
};

interface Props {
  steps: StepTrace[];
  routing?: string;
}

export function WorkflowGraph({ steps, routing }: Props) {
  // Build the ordered list of node ids to display:
  //   - "start" (Question) is always shown — it represents the user's input
  //   - Every node that appears in `steps` (the orchestrator only adds a step
  //     when it actually begins executing that node)
  //   - "end" (Answer) is NOT force-shown until the orchestrator emits it
  const nodeIds: string[] = ['start'];
  for (const s of steps) {
    if (s.node !== 'start' && !nodeIds.includes(s.node)) nodeIds.push(s.node);
  }

  const activeNodes = new Set<string>();
  const statusMap = new Map<string, StepTrace>();
  for (const s of steps) {
    if (s.status === 'running') activeNodes.add(s.node);
    statusMap.set(s.node, s);
  }

  return (
    <div className="space-y-1">
      {nodeIds.map((nodeId, i) => {
        const node = NODE_DEFS[nodeId] ?? {
          id: nodeId,
          label: nodeId,
          description: '',
          icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
        };
        const step = statusMap.get(nodeId);
        const isActive = activeNodes.has(nodeId);
        const isComplete = step?.status === 'done';
        const isError = step?.status === 'error';

        const stateClass = isActive
          ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/30 text-brand-700 dark:text-brand-300 shadow-md'
          : isComplete
          ? 'border-emerald-300 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300'
          : isError
          ? 'border-rose-300 bg-rose-50 dark:bg-rose-900/20 text-rose-700 dark:text-rose-300'
          : 'border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800 text-ink-600 dark:text-ink-300';

        return (
          <div key={`${nodeId}-${i}`} className="relative">
            {i < nodeIds.length - 1 && (
              <div
                className={`absolute left-[19px] top-9 w-0.5 h-3 ${
                  isComplete ? 'bg-emerald-400' : isActive ? 'bg-brand-400' : 'bg-ink-200 dark:bg-ink-700'
                }`}
              />
            )}
            <div className={`relative flex items-start gap-3 rounded-lg border p-2.5 transition-all duration-200 ${stateClass}`}>
              <div
                className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  isActive
                    ? 'bg-brand-500 text-white'
                    : isComplete
                    ? 'bg-emerald-500 text-white'
                    : isError
                    ? 'bg-rose-500 text-white'
                    : 'bg-ink-200 dark:bg-ink-700 text-ink-500 dark:text-ink-400'
                }`}
              >
                {isActive ? (
                  <Spinner />
                ) : isComplete ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                ) : isError ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={node.icon} />
                  </svg>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline justify-between gap-2">
                  <div className="text-sm font-medium truncate">{node.label}</div>
                  {step?.durationMs !== undefined && (
                    <div className="text-[10px] text-ink-400 dark:text-ink-500 font-mono tabular-nums">
                      {step.durationMs}ms
                    </div>
                  )}
                </div>
                <div className="text-[11px] text-ink-500 dark:text-ink-400 mt-0.5">
                  {step?.status === 'running'
                    ? routing && nodeId === 'route'
                      ? `→ ${routing}`
                      : node.description
                    : step?.detail ?? (routing && nodeId === 'route' ? `→ ${routing}` : node.description)}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Spinner() {
  return (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}
