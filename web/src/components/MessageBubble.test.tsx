import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { MessageBubble } from './MessageBubble';
import { useAppStore } from '../store';
import type { ChatMessage } from '../lib/types';
import { RagErrorCode } from '../lib/errors';

let root: ReturnType<typeof createRoot> | undefined;
const container = document.createElement('div');

afterEach(() => {
  act(() => root?.unmount());
  container.remove();
  localStorage.clear();
  useAppStore.getState().resetAll();
});

describe('saved answer sources', () => {
  it('keeps citations accessible after a real persistence round trip', async () => {
    useAppStore.getState().resetAll();
    const message: ChatMessage = {
      id: 'saved-answer', role: 'assistant', content: 'The policy allows remote work [1].', timestamp: Date.now(),
      workflow: {
        question: 'What is the remote work policy?', answer: 'The policy allows remote work [1].',
        routing: 'vectorstore', retryCount: 0, startedAt: 1, documents: [{ id: 'policy-1', content: 'Remote work is allowed.', source: 'handbook.txt' }],
        webResults: [], citations: [{ type: 'vectorstore', source: 'handbook.txt', excerpt: 'Remote work is allowed.' }],
        steps: [{ id: 'generate', node: 'generate', label: 'Generate answer', status: 'done', startedAt: 1, finishedAt: 2 }],
      },
    };
    useAppStore.getState().addMessage(message);
    await useAppStore.persist.rehydrate();
    const saved = useAppStore.getState().conversations.flatMap(c => c.messages).find(m => m.id === message.id);
    expect(saved?.workflow?.documents).toEqual([]);
    expect(saved?.workflow?.citations).toHaveLength(1);
    document.body.appendChild(container);
    root = createRoot(container);
    act(() => root?.render(<MessageBubble message={saved!} />));
    expect(container.textContent).toContain('handbook.txt');
    expect(container.textContent).toContain('Remote work is allowed.');
    const sources = Array.from(container.querySelectorAll('button')).find(button => button.textContent?.includes('Hide sources'));
    expect(sources).toBeDefined();
    act(() => sources!.click());
    expect(container.textContent).not.toContain('handbook.txt');
  });
});

describe('workflow error display', () => {
  function render(message: ChatMessage) {
    document.body.appendChild(container);
    root = createRoot(container);
    act(() => root?.render(<MessageBubble message={message} />));
  }

  it('does not badge a provider error as CORS just because the message mentions CORS', () => {
    render({
      id: 'local-fetch-error',
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      workflow: {
        question: 'Hello',
        documents: [],
        webResults: [],
        citations: [],
        retryCount: 0,
        steps: [],
        startedAt: 1,
        error: {
          code: RagErrorCode.PROVIDER_UNREACHABLE,
          message: 'Cannot connect to http://127.0.0.1:11434/v1. Check CORS settings allow this origin.',
          retryable: false,
        },
      },
    });

    expect(container.textContent).toContain('Check CORS settings allow this origin.');
    expect(container.textContent).not.toContain('CORS blocked');
  });

  it('badges typed CORS errors', () => {
    render({
      id: 'typed-cors-error',
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      workflow: {
        question: 'Hello',
        documents: [],
        webResults: [],
        citations: [],
        retryCount: 0,
        steps: [],
        startedAt: 1,
        error: {
          code: RagErrorCode.CORS_BLOCKED,
          message: 'Browser blocked request to OpenRouter (CORS).',
          retryable: false,
        },
      },
    });

    expect(container.textContent).toContain('CORS blocked');
    expect(container.textContent).toContain('Browser blocked request to OpenRouter');
  });
});
