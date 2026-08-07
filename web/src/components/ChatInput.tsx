// ChatInput — sticky bottom input with submit/cancel

import { useEffect, useRef, useState, useId } from 'react';

interface Props {
  onSubmit: (text: string) => void;
  onCancel: () => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSubmit, onCancel, disabled, placeholder }: Props) {
  const [value, setValue] = useState('');
  const ref = useRef<HTMLTextAreaElement>(null);
  const helperTextId = useId();

  useEffect(() => {
    if (!disabled) ref.current?.focus();
  }, [disabled]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const isTextField =
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        (e.target instanceof HTMLElement && e.target.isContentEditable);
      if (e.key === '/' && !isTextField) {
        e.preventDefault();
        ref.current?.focus();
      } else if (e.key === 'Escape' && !disabled) {
        onCancel();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [disabled, onCancel]);

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSubmit(text);
    setValue('');
  };

  return (
    <div className="border-t border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-900 p-3">
      <div className="relative max-w-4xl mx-auto">
        <textarea
          ref={ref}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={placeholder || 'Ask anything about your data…'}
          disabled={disabled}
          rows={1}
          aria-label="Ask Clay a question"
          aria-busy={disabled}
          aria-describedby={helperTextId}
          className="w-full resize-none px-4 py-3 pr-24 rounded-2xl border border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800 text-sm text-ink-800 dark:text-ink-100 placeholder-ink-400 focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:focus:ring-brand-900 outline-none disabled:opacity-50"
          style={{ minHeight: 48, maxHeight: 160 }}
          onInput={e => {
            const el = e.currentTarget;
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 160) + 'px';
          }}
        />
        <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
          {disabled ? (
            <button
              onClick={onCancel}
              aria-label="Stop generation"
              className="px-3 py-1.5 rounded-lg bg-ink-100 dark:bg-ink-700 text-ink-700 dark:text-ink-200 hover:bg-ink-200 dark:hover:bg-ink-600 text-xs font-medium flex items-center gap-1.5"
            >
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
              Stop
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!value.trim()}
              aria-label="Send message"
              className="px-3 py-1.5 rounded-lg bg-brand-500 text-white hover:bg-brand-600 disabled:opacity-30 disabled:cursor-not-allowed text-xs font-medium flex items-center gap-1.5"
            >
              <span>Send</span>
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M2 21l23-9L2 3v7l15 2-15 2v7z" />
              </svg>
            </button>
          )}
        </div>
      </div>
      <div id={helperTextId} className="sr-only">
        Clay may make mistakes — verify important information. Press / to focus input, Escape to stop generation.
      </div>
      <div className="text-[10px] text-ink-400 dark:text-ink-500 text-center mt-1.5" aria-hidden="true">
        Clay may make mistakes — verify important information. Press / to focus, Esc to stop.
      </div>
    </div>
  );
}
