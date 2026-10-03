import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
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
    vi.unstubAllEnvs();
  });

  it('offers only OpenRouter and Local providers and keeps OpenRouter credentials in their field', () => {
    useAppStore.getState().updateSettings({ openrouterApiKey: 'saved-openrouter-key' });
    unmount = renderPanel();
    expect(document.body.textContent).toContain('OpenRouter');
    expect(document.body.textContent).toContain('Local');
    expect(document.body.textContent).not.toContain('NVIDIA NIM');
    expect(document.body.textContent).not.toContain('Groq');
    const input = document.querySelector<HTMLInputElement>('[aria-label="OpenRouter API key"]');
    expect(input).not.toBeNull();
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;
    act(() => {
      setter.call(input, 'sk-or-v1-updated');
      input!.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(useAppStore.getState().settings.openrouterApiKey).toBe('sk-or-v1-updated');
  });

  it('offers only the two approved free cloud models with no manual model input', () => {
    useAppStore.setState({ availableModels: [
      { id: 'some/paid-model', ownedBy: 'vendor', created: 0 },
      { id: 'nvidia/nemotron-3-super-120b-a12b:free', ownedBy: 'nvidia', created: 0, pricing: { prompt: '0', completion: '0', request: '0' }, supportedParameters: ['tools', 'tool_choice', 'response_format'] },
      { id: 'qwen/qwen3.8-27b:free', ownedBy: 'qwen', created: 0, pricing: { prompt: '0', completion: '0', request: '0' }, supportedParameters: ['tools', 'tool_choice', 'structured_outputs'] },
    ] });
    unmount = renderPanel();
    expect(document.querySelector('input[aria-label="Chat model"]')).toBeNull();
    const select = document.querySelector<HTMLSelectElement>('select[aria-label="Chat model"]');
    expect(select).not.toBeNull();
    expect(Array.from(select!.options).filter(option => option.value).map(option => option.value)).toEqual([
      'nvidia/nemotron-3-super-120b-a12b:free', 'qwen/qwen3.8-27b:free',
    ]);
    expect(document.body.textContent).not.toContain('some/paid-model');
  });

  it('does not render NIM relay configuration', () => {
    unmount = renderPanel();
    expect(document.querySelector<HTMLInputElement>('[aria-label="NIM relay base URL"]')).toBeNull();
    expect(document.body.textContent).not.toMatch(/relay/i);
    expect(document.body.textContent).not.toContain('NVIDIA');
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
