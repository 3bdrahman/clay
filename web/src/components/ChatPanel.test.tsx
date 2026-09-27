// Component test: per-conversation concurrent pipelines (wave 4, item 4).
//
// The wave-1 chat-switch fix made updateMessage route to the conversation
// OWNING the message id and isRunning derive from the active conversation's
// streaming placeholder. These tests verify the resulting behavior at the
// component level — a mid-flight switch must not lose the in-flight
// pipeline, and the other conversation must be able to run its own pipeline
// concurrently. No re-implementation is involved.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import { ChatPanel } from './ChatPanel';
import { useAppStore } from '../store';
import type { ChatMessage } from '../lib/types';

const { mockLLM, mockVectorstore } = vi.hoisted(() => {
  const fakeDoc = {
    id: 'doc-1',
    content: 'Relevant context about the question.',
    source: 'notes.txt',
    score: 0.9,
    metadata: {},
  };
  return {
    mockLLM: { invoke: vi.fn(), stream: vi.fn() },
    mockVectorstore: { similaritySearch: vi.fn(async () => [fakeDoc]) },
  };
});

vi.mock('../hooks/useClay', () => ({
  useClay: () => ({
    services: {
      llm: mockLLM,
      embeddings: { embed: vi.fn(async () => [[0.1, 0.2, 0.3]]) },
      vectorstore: mockVectorstore,
      webSearch: { search: vi.fn(async () => []) },
      analyzer: { analyze: vi.fn(), listDatasets: vi.fn(() => []), getDatasetSummary: vi.fn() },
      ready: true,
    },
    loading: false,
    error: null,
    needsConfiguration: false,
    loadSampleData: vi.fn(async () => {}),
    pickedModels: { chat: 'mock-model' },
  }),
}));

// Registration order must match call order: the first invoke of each
// pipeline is the route call (parked on a deferred), the second invoke of
// each pipeline is the other pipeline's route, and every later invoke
// (HyDE, grade, hallucination, answer eval) consumes the default.
const ROUTE_ANSWER = '{"datasource": "vectorstore", "confidence": 1}';
const EVAL_ANSWER = '{"binary_score": "yes"}';

function deferredRoute(): { promise: Promise<{ content: string }>; resolve: (v: { content: string }) => void } {
  let resolve!: (v: { content: string }) => void;
  const promise = new Promise<{ content: string }>(r => { resolve = r; });
  return { promise, resolve };
}

describe('ChatPanel — per-conversation concurrent pipelines', () => {
  let unmount: (() => void) | null = null;

  beforeEach(() => {
    vi.resetAllMocks();
    localStorage.clear();
    useAppStore.getState().resetAll();
    mockVectorstore.similaritySearch.mockResolvedValue([
      { id: 'doc-1', content: 'Relevant context about the question.', source: 'notes.txt', score: 0.9, metadata: {} },
    ]);
  });

  afterEach(() => {
    unmount?.();
    unmount = null;
  });

  function render(): void {
    act(() => {
      const container = document.createElement('div');
      document.body.appendChild(container);
      const root = createRoot(container);
      root.render(<ChatPanel onOpenData={() => {}} onOpenSettings={() => {}} />);
      unmount = () => {
        act(() => root.unmount());
        container.remove();
      };
    });
  }

  async function flush(n = 3) {
    for (let i = 0; i < n; i++) {
      await act(async () => {
        await new Promise<void>(r => setTimeout(r, 10));
      });
    }
  }

  function typeAndSubmit(text: string): void {
    const textarea = document.querySelector('textarea');
    expect(textarea, 'chat input textarea missing').not.toBeNull();
    const el = textarea as HTMLTextAreaElement;
    const nativeSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
    act(() => {
      nativeSetter?.call(el, text);
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    act(() => {
      el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    });
  }

  function convMessages(id: string): ChatMessage[] {
    return useAppStore.getState().conversations.find(c => c.id === id)?.messages ?? [];
  }

  it('keeps an in-flight pipeline routed to its owning conversation across a switch', async () => {
    const convA = useAppStore.getState().createConversation();
    const convB = useAppStore.getState().createConversation();
    act(() => {
      useAppStore.getState().switchConversation(convA);
    });
    render();

    // Registration order = call order: A's route is parked (deferred), and
    // every later invoke (HyDE, grade, eval) takes the default. A's
    // generate is stream #1.
    const routeA = deferredRoute();
    mockLLM.invoke
      .mockReturnValueOnce(routeA.promise)
      .mockResolvedValue({ content: EVAL_ANSWER });
    mockLLM.stream.mockResolvedValue({ content: 'Answer from conversation A' });

    typeAndSubmit('question A');
    await flush(2);

    // A's pipeline is mid-flight at the route step: user + streaming placeholder in A.
    expect(convMessages(convA)).toHaveLength(2);
    expect(convMessages(convA)[0].content).toBe('question A');
    expect(convMessages(convA)[1].streaming).toBe(true);

    // Switch to B mid-flight: A's placeholder must stay in A (not follow the
    // active conversation), and B must start empty.
    act(() => {
      useAppStore.getState().switchConversation(convB);
    });
    await flush(2);

    expect(convMessages(convA)).toHaveLength(2);
    expect(convMessages(convA)[1].streaming).toBe(true);
    expect(convMessages(convB)).toHaveLength(0);

    // A's pipeline completes while B is active: the answer lands in A.
    routeA.resolve({ content: ROUTE_ANSWER });
    await flush(10);

    expect(convMessages(convA)).toHaveLength(2);
    expect(convMessages(convA)[1].streaming).toBe(false);
    expect(convMessages(convA)[1].content).toBe('Answer from conversation A');
    expect(convMessages(convB)).toHaveLength(0);

    // Switching back to A shows the completed answer.
    act(() => {
      useAppStore.getState().switchConversation(convA);
    });
    await flush(2);
    expect(convMessages(convA)[1].content).toBe('Answer from conversation A');
  });

  it('aborts the previous pipeline gracefully when a new question is submitted in another conversation', async () => {
    const convA = useAppStore.getState().createConversation();
    const convB = useAppStore.getState().createConversation();
    act(() => {
      useAppStore.getState().switchConversation(convA);
    });
    render();

    // Registration order = call order: A's route parked, B's route settles
    // immediately (B completes), later invokes take the default. B's
    // generate is stream #1; A never generates (aborted at the boundary).
    const routeA = deferredRoute();
    const routeB = Promise.resolve({ content: ROUTE_ANSWER });
    mockLLM.invoke
      .mockReturnValueOnce(routeA.promise)
      .mockReturnValueOnce(routeB)
      .mockResolvedValue({ content: EVAL_ANSWER });
    mockLLM.stream.mockResolvedValueOnce({ content: 'Answer from conversation B' });

    typeAndSubmit('question A');
    await flush(2);
    expect(convMessages(convA)[1].streaming).toBe(true);

    act(() => {
      useAppStore.getState().switchConversation(convB);
    });
    typeAndSubmit('question B');
    await flush(2);

    // LLM calls are not abortable mid-call, so A's parked route completes;
    // the next step-boundary check catches the abort and finalizes A's
    // message gracefully instead of leaving it stuck streaming (which
    // would brick the conversation).
    routeA.resolve({ content: ROUTE_ANSWER });
    await flush(10);

    expect(convMessages(convA)).toHaveLength(2);
    expect(convMessages(convA)[1].streaming).toBe(false);
    expect(convMessages(convB)).toHaveLength(2);
    expect(convMessages(convB)[1].streaming).toBe(false);
    expect(convMessages(convB)[1].content).toBe('Answer from conversation B');
  });

  it('finalizes the message instead of leaving it stuck streaming when the pipeline errors', async () => {
    const convA = useAppStore.getState().createConversation();
    act(() => {
      useAppStore.getState().switchConversation(convA);
    });
    render();

    mockLLM.invoke.mockRejectedValue(new Error('provider unreachable'));

    typeAndSubmit('question A');
    await flush(10);

    // A failed route must not leave the placeholder streaming forever —
    // that would keep isRunning true and brick the conversation.
    const messages = convMessages(convA);
    expect(messages).toHaveLength(2);
    expect(messages[1].streaming).toBe(false);
    expect(messages[1].workflow?.error).toBeDefined();
  });
});
