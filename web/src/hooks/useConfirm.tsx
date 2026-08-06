import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

interface ConfirmOptions {
  title: string;
  message?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  destructive?: boolean;
}

interface PendingConfirm extends ConfirmOptions {
  resolve: (ok: boolean) => void;
}

export function useConfirm(): readonly [
  confirm: (opts: ConfirmOptions) => Promise<boolean>,
  renderDialog: () => ReactNode,
] {
  const [pending, setPending] = useState<PendingConfirm | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  const confirm = useCallback((opts: ConfirmOptions) => {
    return new Promise<boolean>(resolve => {
      setPending({ ...opts, resolve });
    });
  }, []);

  const close = useCallback((ok: boolean) => {
    setPending(current => {
      current?.resolve(ok);
      return null;
    });
    previouslyFocused.current?.focus?.();
  }, []);

  useEffect(() => {
    if (!pending) return;
    previouslyFocused.current = (document.activeElement as HTMLElement | null) ?? null;
    const t = setTimeout(() => {
      const focusable = dialogRef.current?.querySelector<HTMLElement>(
        'button[data-autofocus="true"], button',
      );
      focusable?.focus();
    }, 0);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        close(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => {
      clearTimeout(t);
      window.removeEventListener('keydown', onKey);
    };
  }, [pending, close]);

  const renderDialog = (): ReactNode => {
    if (!pending) return null;
    return (
      <div
        className="fixed inset-0 z-[60] flex items-center justify-center p-4"
        onClick={() => close(false)}
        role="dialog"
        aria-modal="true"
        aria-label={pending.title}
      >
        <div className="absolute inset-0 bg-black/40 animate-fade-in" />
        <div
          ref={dialogRef}
          className="relative bg-white dark:bg-ink-900 rounded-xl shadow-2xl max-w-sm w-full p-5 animate-slide-up"
          onClick={e => e.stopPropagation()}
        >
          <h3 className="text-base font-semibold text-ink-900 dark:text-ink-50">
            {pending.title}
          </h3>
          {pending.message && (
            <p className="text-sm text-ink-600 dark:text-ink-300 mt-1.5">{pending.message}</p>
          )}
          <div className="flex justify-end gap-2 mt-4">
            <button
              type="button"
              onClick={() => close(false)}
              className="px-3 py-1.5 text-xs font-medium text-ink-700 dark:text-ink-200 border border-ink-200 dark:border-ink-700 rounded-lg hover:bg-ink-100 dark:hover:bg-ink-800"
            >
              {pending.cancelLabel ?? 'Cancel'}
            </button>
            <button
              type="button"
              data-autofocus="true"
              onClick={() => close(true)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg text-white ${
                pending.destructive
                  ? 'bg-rose-600 hover:bg-rose-700'
                  : 'bg-brand-600 hover:bg-brand-700'
              }`}
            >
              {pending.confirmLabel ?? 'Confirm'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  return [confirm, renderDialog] as const;
}
