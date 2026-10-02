import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import type { ChatMessage, WorkflowState } from '../lib/types';
import { useAppStore } from '../store';
import { useShallow } from 'zustand/shallow';
import type { UseClayResult } from '../hooks/useClay';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { ExampleQuestions } from './ExampleQuestions';
import { LandingHero } from './LandingHero';
import { createWorkflowOrchestrator } from '../services/orchestrator';
import { resolveModels, pickLocalModels } from '../lib/models';

const EMPTY_MESSAGES: ChatMessage[] = [];

const STREAMING_STORE_THROTTLE_MS = 500;

export function ChatPanel({
  clay,
  onOpenData,
  onOpenSettings,
}: {
  clay: UseClayResult;
  onOpenData: () => void;
  onOpenSettings: () => void;
}) {
  const messages = useAppStore(
    useShallow(s => {
      const conv = s.conversations.find(c => c.id === s.activeConversationId);
      return conv?.messages ?? EMPTY_MESSAGES;
    }),
  ) as ChatMessage[];
  const addMessage = useAppStore(s => s.addMessage);
  const updateMessage = useAppStore(s => s.updateMessage);
  const { services, loading, error, needsConfiguration, loadSampleData, pickedModels } = clay;
  const canSubmit = !!services && !!pickedModels.chat && !loading && !needsConfiguration;

  const scrollRef = useRef<HTMLDivElement>(null);
  const [streamingContent, setStreamingContent] = useState('');
  const [sampleLoadError, setSampleLoadError] = useState<string | null>(null);
  const streamingMessageIdRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Derived from the streaming placeholder, not a global flag: switching
  // conversations mid-generation must not clear it (the pipeline is still
  // running) and must not let a second orchestrator start.
  const isRunning = useMemo(() => messages.some(m => m.streaming === true), [messages]);

  const cancel = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  }, []);

  const handleLoadSample = useCallback(async () => {
    setSampleLoadError(null);
    try {
      await loadSampleData();
    } catch (e) {
      setSampleLoadError(e instanceof Error ? e.message : String(e));
    }
  }, [loadSampleData]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [messages, isRunning, streamingContent]);

  const handleSubmit = async (text: string) => {
    if (!canSubmit) {
      onOpenSettings();
      return;
    }
    if (isRunning) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };
    addMessage(userMsg);

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

    let pipelineSettled = false;

    const updateAssistant = (workflow: WorkflowState, final = false) => {
      // emitSteps schedules a rAF that re-invokes this even after the
      // pipeline settled; a late call would re-flag the placeholder as
      // streaming (its answer is empty on the abort path) and brick the
      // conversation. Only pre-settle updates apply.
      if (pipelineSettled) return;
      const currentContent = useAppStore.getState().conversations.flatMap(c => c.messages).find(m => m.id === assistantId)?.content || '';
      const finalContent = workflow.answer || currentContent;
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: finalContent,
        timestamp: Date.now(),
        workflow: { ...workflow },
        streaming: final ? false : !workflow.answer,
      };
      updateMessage(assistantId, () => assistantMsg);
    };

    let pendingToken = '';
    let rafScheduled = false;
    let lastStoreFlush = 0;
    const flushTokens = () => {
      rafScheduled = false;
      const toFlush = pendingToken;
      pendingToken = '';
      if (!toFlush) return;
      const id = streamingMessageIdRef.current;
      if (!id) return;
      setStreamingContent(prev => prev + toFlush);
      const now = Date.now();
      if (now - lastStoreFlush >= STREAMING_STORE_THROTTLE_MS) {
        lastStoreFlush = now;
        useAppStore.setState(state => ({
          conversations: state.conversations.map(c =>
            c.messages.some(m => m.id === id)
              ? { ...c, messages: c.messages.map(m =>
                  m.id === id ? { ...m, content: (m.content || '') + toFlush } : m,
                ), updatedAt: Date.now() }
              : c,
          ),
        }));
      }
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
      
      // Get the last completed analysis from conversation history for cross-turn memory
      const state = useAppStore.getState();
      const conv = state.conversations.find(c => c.id === state.activeConversationId);
      const lastAnalysis = conv?.messages
        .filter(m => m.role === 'assistant' && m.workflow?.dataAnalysis)
        .pop()?.workflow?.dataAnalysis;
      
      const orchestrator = createWorkflowOrchestrator(
        text,
        {
          llm: services.llm,
          vectorstore: services.vectorstore,
          webSearch: services.webSearch,
          analyzer: services.analyzer,
          settings,
          pickedModels,
          previousAnalysis: lastAnalysis,
        },
        {
          onPartialUpdate: (state: WorkflowState) => updateAssistant(state),
          onToken,
        }
      );

      const finalState = await orchestrator.run(controller.signal);
      if (controller.signal.aborted) {
        // The orchestrator returned the partial state on abort — keep the
        // streamed content and clear the streaming cursor; applying
        // updateAssistant would flag the message as still-streaming.
        const partial = useAppStore.getState().conversations.flatMap(c => c.messages).find(m => m.id === assistantId);
        if (partial) {
          updateMessage(assistantId, (m) => ({ ...m, streaming: false }));
        }
      } else {
        updateAssistant(finalState, true);
      }
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      const isAbort = err.name === 'AbortError' || controller.signal.aborted;
      const existingContent = useAppStore.getState().conversations.flatMap(c => c.messages).find(m => m.id === assistantId)?.content || '';
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
      // The settle guard is set here so the explicit updates above land
      // first; the late rAF (fires on the next frame) is then blocked.
      pipelineSettled = true;
      setStreamingContent('');
      streamingMessageIdRef.current = null;
      if (abortControllerRef.current === controller) abortControllerRef.current = null;
    }
  };

  const showExamples = messages.length === 0 && !isRunning;
  const showLanding = messages.length === 0 && needsConfiguration && !error;

  let content: React.ReactNode;
  if (showLanding) {
    content = (
      <LandingHero
        onLoadSample={loadSampleData}
        onAddData={onOpenData}
        onOpenSettings={onOpenSettings}
        onExampleSelect={handleSubmit}
      />
    );
  } else if (showExamples) {
    if (loading) {
      content = (
        <div className="pt-8">
          <div className="text-center py-12">
            <div className="inline-block w-8 h-8 border-2 border-brand-500 border-t-transparent rounded-full animate-spin mb-3" />
            <p className="text-sm text-ink-500">Loading Clay</p>
          </div>
        </div>
      );
    } else if (services) {
      content = (
        <div className="pt-8">
          <ExampleQuestions
            onSelect={handleSubmit}
            onLoadSample={handleLoadSample}
            onOpenSettings={onOpenSettings}
          />
        </div>
      );
    } else if (error) {
      content = (
        <div className="pt-8">
          <div className="text-center py-12">
            <p className="text-sm text-rose-600 dark:text-rose-400 mb-2">{String(error)}</p>
            <button
              onClick={() => location.reload()}
              className="text-sm text-brand-600 hover:underline"
            >
              Retry
            </button>
          </div>
        </div>
      );
    } else {
      content = null;
    }
  } else {
    content = messages.map(m => <MessageBubble key={m.id} message={m} />);
  }

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
          {content}
          {sampleLoadError && (
            <div role="alert" className="text-sm text-rose-600 dark:text-rose-400 text-center">
              {sampleLoadError}
            </div>
          )}
          {isRunning && !showExamples && !showLanding && (
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
        disabled={!canSubmit}
        isRunning={isRunning}
        onConfigure={canSubmit ? undefined : onOpenSettings}
      />
    </div>
  );
}
