import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import { SettingsPanel } from './SettingsPanel';
import { useAppStore } from '../store';

function renderPanel(): () => void {
  let root: ReturnType<typeof createRoot> | null = null;
  const container = document.createElement('div');
  act(() => {
    document.body.appendChild(container);
    root = createRoot(container);
    root.render(
      <SettingsPanel
        open
        onClose={() => {}}
        refreshModels={async () => {}}
        pickedModels={{ chat: 'mock-model' }}
        resetAll={() => {}}
        clearSandboxData={() => {}}
      />,
    );
  });
  return () => {
    act(() => root?.unmount());
    container.remove();
  };
}

function budgetSlider(): HTMLInputElement {
  const slider = document.querySelector('input[type="range"][min="25000"]');
  expect(slider, 'analysis token budget slider missing').not.toBeNull();
  return slider as HTMLInputElement;
}

describe('SettingsPanel — analysis token budget', () => {
  let unmount: (() => void) | null = null;

  beforeEach(() => {
    localStorage.clear();
    useAppStore.getState().resetAll();
  });

  afterEach(() => {
    unmount?.();
    unmount = null;
  });

  it('shows the default budget initially', () => {
    unmount = renderPanel();
    expect(budgetSlider().value).toBe('100000');
  });

  it('updates the tool-loop token budget through the store', () => {
    unmount = renderPanel();
    const slider = budgetSlider();
    const nativeSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    act(() => {
      nativeSetter?.call(slider, '250000');
      slider.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(useAppStore.getState().settings.maxToolLoopTokens).toBe(250_000);
  });
});
