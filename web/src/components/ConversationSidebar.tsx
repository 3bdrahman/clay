import { useState } from 'react';
import { useAppStore, type Conversation } from '../store';
import { useModalFocus } from '../hooks/useModalFocus';

interface Props {
  open: boolean;
  onClose: () => void;
}

export function ConversationSidebar({ open, onClose }: Props) {
  const conversations = useAppStore(s => s.conversations);
  const activeId = useAppStore(s => s.activeConversationId);
  const createConversation = useAppStore(s => s.createConversation);
  const deleteConversation = useAppStore(s => s.deleteConversation);
  const renameConversation = useAppStore(s => s.renameConversation);
  const switchConversation = useAppStore(s => s.switchConversation);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  const { dialogRef, stopBackdrop } = useModalFocus(open);

  function startEdit(c: Conversation) {
    setEditingId(c.id);
    setEditValue(c.title);
  }

  function commitEdit() {
    if (editingId) {
      const trimmed = editValue.trim();
      if (trimmed) renameConversation(editingId, trimmed);
    }
    setEditingId(null);
    setEditValue('');
  }

  function cancelEdit() {
    setEditingId(null);
    setEditValue('');
  }

  function handleNewChat() {
    createConversation();
    onClose();
  }

  function handleSelect(id: string) {
    if (editingId === id) return;
    switchConversation(id);
    onClose();
  }

  function handleDelete(id: string) {
    if (confirmDeleteId === id) {
      deleteConversation(id);
      setConfirmDeleteId(null);
    } else {
      setConfirmDeleteId(id);
    }
  }

  function formatTime(ts: number): string {
    const now = Date.now();
    const diff = now - ts;
    if (diff < 60_000) return 'just now';
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
    return new Date(ts).toLocaleDateString();
  }

  return (
    <>
      <div
        className={`fixed inset-0 bg-black/30 z-40 transition-opacity duration-200 ${
          open ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
      />
      <aside
        ref={dialogRef}
        className={`fixed left-0 top-0 bottom-0 w-64 bg-white dark:bg-ink-900 border-r border-ink-200 dark:border-ink-700 z-40 flex flex-col transition-transform duration-200 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
        role="dialog"
        aria-modal="true"
        aria-label="Conversations"
        onClick={stopBackdrop}
      >
        <div className="px-3 py-3 border-b border-ink-200 dark:border-ink-700">
          <button
            onClick={handleNewChat}
            type="button"
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-brand-600 hover:bg-brand-700 text-white text-sm font-medium transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-2">
          {conversations.length === 0 ? (
            <div className="text-center py-8 px-4">
              <p className="text-xs text-ink-400 dark:text-ink-500">No conversations yet</p>
            </div>
          ) : (
            <ul className="space-y-0.5">
              {conversations.map(c => {
                const isActive = c.id === activeId;
                const isEditing = editingId === c.id;
                const isConfirming = confirmDeleteId === c.id;

                if (isEditing) {
                  return (
                    <li key={c.id}>
                      <input
                        type="text"
                        value={editValue}
                        onChange={e => setEditValue(e.target.value)}
                        onBlur={commitEdit}
                        onKeyDown={e => {
                          if (e.key === 'Enter') commitEdit();
                          if (e.key === 'Escape') cancelEdit();
                        }}
                        autoFocus
                        className="w-full px-3 py-2 rounded-lg bg-ink-100 dark:bg-ink-800 text-sm border border-brand-500 outline-none"
                        placeholder="Conversation title"
                      />
                    </li>
                  );
                }

                return (
                  <li key={c.id} className="group flex items-center gap-1">
                    <div
                      onClick={() => handleSelect(c.id)}
                      onDoubleClick={() => startEdit(c)}
                      className={`flex-1 min-w-0 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                        isActive
                          ? 'bg-brand-50 dark:bg-brand-900/20 text-brand-700 dark:text-brand-300'
                          : 'hover:bg-ink-100 dark:hover:bg-ink-800 text-ink-700 dark:text-ink-300'
                      }`}
                    >
                      <div className="text-sm font-medium truncate">{c.title || 'New chat'}</div>
                      {c.messages.length > 0 && (
                        <div className="text-[10px] text-ink-400 dark:text-ink-500">
                          {c.messages.length} msg · {formatTime(c.updatedAt)}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => handleDelete(c.id)}
                      type="button"
                      title={isConfirming ? 'Click again to confirm' : 'Delete conversation'}
                      className={`flex-shrink-0 w-7 h-7 flex items-center justify-center rounded transition-colors ${
                        isConfirming
                          ? 'text-rose-600 bg-rose-50 dark:bg-rose-900/30'
                          : 'text-ink-400 dark:text-ink-500 hover:text-rose-500 opacity-0 group-hover:opacity-100'
                      }`}
                    >
                      {isConfirming ? (
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 13l3 3m0 0l3-3m-3 3V8m0 13a9 9 0 110-18 9 9 0 010 18z" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      )}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </aside>
    </>
  );
}