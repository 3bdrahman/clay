import { useEffect, useRef, useState, useCallback } from 'react';
import type { ChatMessage, WorkflowState } from '../lib/types';
import { useAppStore } from '../store';
import { useShallow } from 'zustand/shallow';
import { useClay } from '../hooks/useClay';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { ExampleQuestions } from './ExampleQuestions';
import { LandingHero } from './LandingHero';
import { createWorkflowOrchestrator } from '../services/orchestrator';
import { resolveModels, pickLocalModels } from '../lib/models';

const EMPTY_MESSAGES: ChatMessage[] = [];

function useActiveMessages(): ChatMessage[] {
  const messages = useAppStore(
    useShallow(s => {
      const conv = s.conversations.find(c => c.id === s.activeConversationId);
      return conv?.messages ?? EMPTY_MESSAGES;
    }),
  ) as ChatMessage[];
  return messages;
}

export function ChatPanel({ onOpenSettings }: { onOpenSettings: () => void }) {
  const messages = useActiveMessages();
  const addMessage = useAppStore(s => s.addMessage);
  const updateMessage = useAppStore(s => s.updateMessage);
  const isRunning = useAppStore(s => s.isRunning);
  const setRunning = useAppStore(s => s.setRunning);
  const { services, loading, error, needsConfiguration } = useClay();

  const scrollRef = useRef<HTMLDivElement>(null);
  const [streamingContent, setStreamingContent] = useState('');
  const streamingMessageIdRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, isRunning, streamingContent]);

  const handleSubmit = async (text: string) => {
    if (!services) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };
    addMessage(userMsg);

    setRunning(true);

    const assistantId = crypto.randomUUID();
    streamingMessageIdRef.current = assistantId;
    setStreamingContent('');

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const placeholder: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      streaming: true,
    };
    addMessage(placeholder);

    const updateAssistant = (workflow: WorkflowState) => {
      const state = useAppStore.getState();
      const conv = state.conversations.find(c => c.id === state.activeConversationId);
      const currentMsg = conv?.messages.find(m => m.id === assistantId);
      const currentContent = currentMsg?.content || '';
      const finalContent = workflow.answer || currentContent;
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: finalContent,
        timestamp: Date.now(),
        workflow: { ...workflow },
        streaming: !workflow.answer,
      };
      updateMessage(assistantId, () => assistantMsg);
    };

    let pendingToken = '';
    let rafScheduled = false;
    const flushTokens = () => {
      rafScheduled = false;
      const toFlush = pendingToken;
      pendingToken = '';
      if (!toFlush) return;
      const id = streamingMessageIdRef.current;
      if (!id) return;
      setStreamingContent(prev => prev + toFlush);
      useAppStore.setState(state => ({
        conversations: state.conversations.map(c =>
          c.id === state.activeConversationId
            ? { ...c, messages: c.messages.map(m =>
                m.id === id ? { ...m, content: (m.content || '') + toFlush } : m,
              ), updatedAt: Date.now() }
            : c,
        ),
      }));
    };

    const onToken = (token: string) => {
      pendingToken += token;
      if (!rafScheduled) {
        rafScheduled = true;
        requestAnimationFrame(flushTokens);
      }
    };

    try {
      const settings = useAppStore.getState().settings;
      const availableModels = useAppStore.getState().availableModels;
      const pickedModels =
        settings.provider === 'local'
          ? pickLocalModels(settings.localModels)
          : resolveModels(settings, availableModels).picked;
      const orchestrator = createWorkflowOrchestrator(
        text,
        {
          llm: services.llm,
          vectorstore: services.vectorstore,
          webSearch: services.webSearch,
          analyzer: services.analyzer,
          settings,
          pickedModels,
        },
        {
          onPartialUpdate: (state: WorkflowState) => updateAssistant(state),
          onToken,
        }
      );

      const finalState = await orchestrator.run(controller.signal);
      setStreamingContent('');
      streamingMessageIdRef.current = null;
      updateAssistant(finalState);
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      const isAbort = err.name === 'AbortError' || controller.signal.aborted;
      const state = useAppStore.getState();
      const conv = state.conversations.find(c => c.id === state.activeConversationId);
      const existingContent = conv?.messages.find(m => m.id === assistantId)?.content || '';
      const errorMsg: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: isAbort ? existingContent : '',
        error: isAbort ? undefined : err.message,
        timestamp: Date.now(),
        streaming: false,
      };
      updateMessage(assistantId, () => errorMsg);
    } finally {
      setRunning(false);
      setStreamingContent('');
      streamingMessageIdRef.current = null;
      if (abortControllerRef.current === controller) abortControllerRef.current = null;
    }
  };

const showExamples = messages.length === 0 && !isRunning;
  const isDemoMode = needsConfiguration && !loading && !error;

  return (
    <div className="flex-1 flex flex-col h-full min-h-0">
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-6"
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
      >
        <div className="max-w-4xl mx-auto space-y-5">
          {isDemoMode ? (
            <LandingHero
              onGetStarted={onOpenSettings}
              onLoadSample={() => {
                const { loadSampleData } = useClay();
                loadSampleData();
              }}
            />
          ) : showExamples ? (
            <div className="pt-8">
              {loading ? (
                <div className="text-center py-12">
                  <div className="inline-block w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mb-3" />
                  <p className="text-sm text-ink-500">Loading Clay</p>
               </div>
              ) : services ? (
                <ExampleQuestions onSelect={handleSubmit} />
              ) : error ? (
                <div className="text-center py-12">
                  <p className="text-sm text-rose-600 dark:text-rose-400 mb-2">{String(error)}</p>
                  <button
                    onClick={() => location.reload()}
                    className="text-sm text-brand-600 hover:underline"
                  >
                    Retry
                 </button>
                </div>
              ) : null}
            </div>
          ) : (
            messages.map(m => <MessageBubble key={m.id} message={m} />)
          )}
          {isRunning && !showExamples && !isDemoMode && (
            <div className="flex justify-start">
              <div className="bg-white dark:bg-ink-800 border border-ink-200 dark:border-ink-700 rounded-2xl rounded-tl-sm px-4 py-2.5 shadow-sm">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 bg-brand-500 rounded-full animate-pulse" />
                  <span className="w-2 h-2 bg-brand-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
                  <span className="w-2 h-2 bg-brand-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }} />
               </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <ChatInput
        onSubmit={handleSubmit}
        onCancel={cancel}
        disabled={isRunning || !services}
      />
   </div>
  );
}
