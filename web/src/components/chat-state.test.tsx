import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ChatInput } from './ChatInput';

const container = document.createElement('div');
let root: ReturnType<typeof createRoot>;

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function renderInput(options: { disabled: boolean; isRunning: boolean; onConfigure?: () => void }) {
  document.body.appendChild(container);
  root = createRoot(container);
  const onCancel = vi.fn();
  const onSubmit = vi.fn();
  act(() => root.render(<ChatInput {...options} onSubmit={onSubmit} onCancel={onCancel} />));
  return { onCancel, onSubmit };
}

describe('ChatInput request state', () => {
  it('keeps drafts editable without showing Stop when services are unavailable', () => {
    // Given no running request and unavailable services.
    renderInput({ disabled: true, isRunning: false });
    // When the composer is rendered, then it is not a generation control.
    expect(container.querySelector('textarea')?.disabled).toBe(false);
    expect(container.querySelector('[aria-label="Stop generation"]')).toBeNull();
    expect(container.querySelector('textarea')?.getAttribute('aria-busy')).toBe('false');
  });

  it('opens configuration instead of offering a nonfunctional Stop button', () => {
    const onConfigure = vi.fn();
    renderInput({ disabled: true, isRunning: false, onConfigure });
    const button = container.querySelector<HTMLButtonElement>('[aria-label="Configure provider"]');
    act(() => button?.click());
    expect(onConfigure).toHaveBeenCalledOnce();
  });

  it('cancels an actual running request with Escape', () => {
    const { onCancel } = renderInput({ disabled: true, isRunning: true });
    act(() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })));
    expect(onCancel).toHaveBeenCalledOnce();
    expect(container.querySelector('[aria-label="Stop generation"]')).not.toBeNull();
  });

  it('does not cancel when idle', () => {
    const { onCancel } = renderInput({ disabled: false, isRunning: false });
    act(() => window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })));
    expect(onCancel).not.toHaveBeenCalled();
  });
});
