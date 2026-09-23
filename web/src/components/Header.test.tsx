import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { Header } from './Header';
import { useAppStore } from '../store';
import type { PickedModels } from '../lib/models';
import type { ProviderKind } from '../lib/types';

const container = document.createElement('div');
let root: ReturnType<typeof createRoot>;

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  useAppStore.setState({ conversations: [], activeConversationId: null });
});

function renderHeader(provider: ProviderKind = 'openrouter') {
  document.body.appendChild(container);
  root = createRoot(container);
  const pickedModels: PickedModels = { chat: 'test-chat', embedding: 'test-embedding' };
  act(() =>
    root.render(
      <Header
        onOpenSettings={() => {}}
        onOpenData={() => {}}
        onToggleSidebar={() => {}}
        pickedModels={pickedModels}
        provider={provider}
      />
    )
  );
  return container;
}

describe('Header accessibility', () => {
  it('labels every icon-only control for screen readers', () => {
    const el = renderHeader();
    expect(el.querySelector('[aria-label="Toggle conversations sidebar"]')).not.toBeNull();
    expect(el.querySelector('[aria-label="Switch theme (currently system)"]')).not.toBeNull();
    expect(el.querySelector('[aria-label="Open data sandbox"]')).not.toBeNull();
    expect(el.querySelector('[aria-label="Open settings"]')).not.toBeNull();
  });

  it('hides the clear-chat control when the conversation is empty', () => {
    const el = renderHeader();
    expect(el.querySelector('[aria-label="Clear chat history"]')).toBeNull();
  });

  it('shows the clear-chat control once messages exist', () => {
    const el = renderHeader();
    act(() => {
      useAppStore.setState(state => ({
        conversations: [
          ...state.conversations,
          {
            id: 'test-conv',
            title: 'Test conversation',
            messages: [
              { id: 'm1', role: 'assistant' as const, content: 'hello', timestamp: Date.now() },
            ],
            createdAt: Date.now(),
            updatedAt: Date.now(),
          },
        ],
        activeConversationId: 'test-conv',
      }));
    });
    expect(el.querySelector('[aria-label="Clear chat history"]')).not.toBeNull();
  });
});
