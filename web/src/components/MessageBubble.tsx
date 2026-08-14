// MessageBubble — renders user/assistant message with workflow details

import { useState } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import type { ChatMessage } from '../lib/types';
import { CitationPanel } from './CitationPanel';
import { WorkflowGraph } from './WorkflowGraph';

interface Props {
  message: ChatMessage;
}

export function MessageBubble({ message }: Props) {
  const [showWorkflow, setShowWorkflow] = useState(false);
  const [showSources, setShowSources] = useState(true);

  const isUser = message.role === 'user';
  const isError = !!message.error;
  const workflowError = message.workflow?.error;
  const isWorkflowError = !!workflowError;
  const isCorsError = isWorkflowError && 
    (workflowError.code === 'CORS_BLOCKED' || 
     String(workflowError.message).includes('CORS'));
  const wf = message.workflow;
  const isStreaming = !!message.streaming && !isError && !isWorkflowError;

  if (isUser) {
    return (
      <div className="flex justify-end animate-slide-up">
        <div className="max-w-[80%]">
          <div className="bg-brand-500 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 shadow-sm">
            <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">{message.content}</div>
          </div>
          <div className="text-[10px] text-ink-400 mt-1 text-right">{formatTime(message.timestamp)}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start animate-slide-up">
      <div className="max-w-[85%] w-full">
        <div className="flex items-start gap-2">
          <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center text-white shadow-sm">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
<div className="flex-1 min-w-0">
            <div
              className={`rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm ${
                isError || isWorkflowError
                  ? 'bg-rose-50 dark:bg-rose-900/20 border border-rose-200 dark:border-rose-800'
                  : 'bg-white dark:bg-ink-800 border border-ink-200 dark:border-ink-700'
              }`}
            >
              {isError ? (
                <div className="text-sm text-rose-700 dark:text-rose-300">
                  <div className="font-semibold mb-1">Error</div>
                  <div className="whitespace-pre-wrap">{message.error}</div>
                </div>
              ) : isWorkflowError ? (
                isCorsError ? (
                  <details className="group">
                    <summary className="cursor-pointer text-sm font-medium text-rose-700 dark:text-rose-300 select-none flex items-center gap-1.5">
                      <svg className="w-4 h-4 flex-shrink-0 text-rose-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      CORS blocked — NVIDIA NIM does not allow requests from this origin
                    </summary>
                    <div className="mt-2 text-sm text-rose-700 dark:text-rose-300 bg-rose-50 dark:bg-rose-900/30 border border-rose-200 dark:border-rose-800 rounded px-3 py-2 whitespace-pre-line animate-fade-in">
                      {workflowError.message}
                    </div>
                  </details>
                ) : (
                  <div className="text-sm text-rose-700 dark:text-rose-300">
                    <div className="font-semibold mb-1">Error</div>
                    <div className="whitespace-pre-wrap">{workflowError.message}</div>
                  </div>
                )
              ) : isStreaming ? (
                <div className="markdown-content text-ink-800 dark:text-ink-100 text-sm leading-relaxed whitespace-pre-wrap break-words">
                  {message.content}
                  <span className="inline-block w-1.5 h-4 ml-0.5 bg-brand-500 animate-pulse align-middle" />
               </div>
              ) : (
                <div
                  className="markdown-content text-ink-800 dark:text-ink-100"
                  dangerouslySetInnerHTML={{
                    __html: DOMPurify.sanitize(marked.parse(message.content || '') as string),
                  }}
                />
              )}

              {wf && (wf.steps?.length ?? 0) > 0 && (
                <div className="mt-3 pt-3 border-t border-ink-100 dark:border-ink-700 space-y-2">
                  <div className="flex gap-2 flex-wrap items-center">
                    {wf.routing && (
                      <span className="inline-flex items-center gap-1 text-[10px] uppercase font-semibold text-brand-600 dark:text-brand-400 bg-brand-50 dark:bg-brand-900/30 px-2 py-0.5 rounded">
                        <span>Routed to</span>
                        <span className="font-bold">{wf.routing}</span>
                      </span>
                    )}
                    {wf.finishedAt && wf.startedAt && (
                      <span className="text-[10px] text-ink-500 dark:text-ink-400 font-mono">
                        {Math.round(wf.finishedAt - wf.startedAt)}ms total
                      </span>
                    )}
                    {wf.retryCount > 0 && (
                      <span className="text-[10px] text-amber-700 dark:text-amber-400 font-semibold">
                        {wf.retryCount} retry
                      </span>
                    )}
                  </div>

                  <div className="flex gap-1.5">
                    <button
                      onClick={() => setShowWorkflow(s => !s)}
                      className="text-[11px] px-2 py-1 rounded bg-ink-100 dark:bg-ink-700 hover:bg-ink-200 dark:hover:bg-ink-600 text-ink-700 dark:text-ink-200 font-medium"
                    >
                      {showWorkflow ? 'Hide' : 'Show'} workflow
                    </button>
                    {((wf.documents?.length ?? 0) > 0 || (wf.webResults?.length ?? 0) > 0 || wf.dataAnalysis) && (
                      <button
                        onClick={() => setShowSources(s => !s)}
                        className="text-[11px] px-2 py-1 rounded bg-ink-100 dark:bg-ink-700 hover:bg-ink-200 dark:hover:bg-ink-600 text-ink-700 dark:text-ink-200 font-medium"
                      >
                        {showSources ? 'Hide' : 'Show'} sources ({wf.citations?.length ?? 0})
                      </button>
                    )}
                  </div>

                  {showSources && ((wf.documents?.length ?? 0) > 0 || (wf.webResults?.length ?? 0) > 0 || wf.dataAnalysis) && (
                    <div className="mt-2">
                      <CitationPanel
                        documents={wf.documents ?? []}
                        webResults={wf.webResults ?? []}
                        analysis={wf.dataAnalysis}
                        citations={wf.citations ?? []}
                      />
                    </div>
                  )}

                  {showWorkflow && (
                    <div className="mt-2 p-2 bg-ink-50 dark:bg-ink-900 rounded">
                      <WorkflowGraph steps={wf.steps ?? []} routing={wf.routing} />
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="text-[10px] text-ink-400 mt-1">{formatTime(message.timestamp)}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
