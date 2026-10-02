import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ConversationSidebar } from './ConversationSidebar';
import { useAppStore, type Conversation } from '../store';

function renderSidebar(open: boolean, onClose = () => {}): () => void {
  let root: ReturnType<typeof createRoot> | null = null;
  const container = document.createElement('div');
  act(() => {
    document.body.appendChild(container);
    root = createRoot(container);
    root.render(<ConversationSidebar open={open} onClose={onClose} />);
  });
  return () => {
    act(() => root?.unmount());
    container.remove();
  };
}

function click(element: Element): void {
  act(() => {
    element.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
}

function makeConversation(overrides: Partial<Conversation> = {}): Conversation {
  return {
    id: 'conversation-1',
    title: 'Existing chat',
    messages: [],
    createdAt: 1,
    updatedAt: 1,
    ...overrides,
  };
}

beforeEach(() => {
  document.body.textContent = '';
  localStorage.clear();
  useAppStore.getState().resetAll();
});

describe('ConversationSidebar accessibility', () => {
  it('renders no dialog or controls while closed', () => {
    const unmount = renderSidebar(false);

    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(document.body.textContent).not.toContain('New chat');
    expect(document.body.textContent).not.toContain('Existing chat');
    unmount();
  });

  it('closes from an accessible button without creating or selecting a conversation', () => {
    const onClose = vi.fn();
    const createConversation = vi.fn(() => 'new-conversation');
    const switchConversation = vi.fn();
    act(() => {
      useAppStore.setState({
        conversations: [makeConversation()],
        activeConversationId: 'conversation-1',
        createConversation,
        switchConversation,
      });
    });

    const unmount = renderSidebar(true, onClose);
    const closeButton = document.querySelector('[aria-label="Close conversations"]');
    expect(closeButton).not.toBeNull();

    click(closeButton as HTMLButtonElement);

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(createConversation).not.toHaveBeenCalled();
    expect(switchConversation).not.toHaveBeenCalled();
    unmount();
  });
});
