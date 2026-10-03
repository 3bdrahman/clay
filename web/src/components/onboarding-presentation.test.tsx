import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DataSandbox } from './DataSandbox';
import { ExampleQuestions } from './ExampleQuestions';
import { LandingHero } from './LandingHero';
import { WebSearchSetting } from './settings/WebSearchSetting';
import { useAppStore } from '../store';
import type { Settings } from '../lib/types';

function render(ui: React.ReactNode): () => void {
  let root: ReturnType<typeof createRoot> | null = null;
  const container = document.createElement('div');
  act(() => {
    document.body.appendChild(container);
    root = createRoot(container);
    root.render(ui);
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

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function currentSettings(overrides: Partial<Settings> = {}): Settings {
  return { ...useAppStore.getState().settings, ...overrides };
}

beforeEach(() => {
  localStorage.clear();
  document.body.textContent = '';
  useAppStore.getState().resetAll();
});

afterEach(() => {
  document.body.textContent = '';
});

describe('LandingHero onboarding presentation', () => {
  it('shows pending and success feedback when sample data is loaded', async () => {
    const sample = deferred<void>();
    const loadSample = vi.fn(() => sample.promise);
    const unmount = render(
      <LandingHero
        onLoadSample={loadSample}
        onAddData={() => {}}
        onOpenSettings={() => {}}
      />,
    );

    click(document.querySelector('button') as HTMLButtonElement);
    expect(loadSample).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).toContain('Loading sample data');

    await act(async () => {
      sample.resolve();
      await sample.promise;
    });
    expect(document.body.textContent).toContain('Sample data loaded');
    unmount();
  });

  it('does not offer web examples when search is unavailable and opens settings from the configure action', () => {
    const openSettings = vi.fn();
    useAppStore.getState().updateSettings({ webSearchProvider: 'none' });
    const unmount = render(
      <LandingHero
        onLoadSample={async () => {}}
        onAddData={() => {}}
        onOpenSettings={openSettings}
      />,
    );

    expect(document.body.textContent).not.toContain('Latest AI trends for business');
    expect(document.body.textContent).toContain('Web search is disabled');
    click(document.querySelector('[data-testid="landing-configure-web"]') as HTMLButtonElement);
    expect(openSettings).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('does not warn about unavailable search on the keyless default provider', () => {
    const unmount = render(
      <LandingHero
        onLoadSample={async () => {}}
        onAddData={() => {}}
        onOpenSettings={() => {}}
      />,
    );

    expect(document.querySelector('[data-testid="landing-configure-web"]')).toBeNull();
    expect(document.body.textContent).not.toContain('DuckDuckGo is unavailable');
    expect(document.body.textContent).not.toContain('Add a Serper API key');
    unmount();
  });

  it('offers stable web research examples when keyless search is available', () => {
    useAppStore.setState({
      sandboxDocuments: [{ id: 'doc-1', fileName: 'notes.txt', source: 'notes.txt', chunkCount: 1, loadedAt: 1 }],
    });
    const unmount = render(
      <LandingHero
        onLoadSample={async () => {}}
        onAddData={() => {}}
      />,
    );

    expect(document.body.textContent).toContain('What does independent web indexing mean for search quality?');
    expect(document.body.textContent).not.toContain('Latest AI trends for business');
    unmount();
  });

  it('states the model setup requirement and what context is sent to the LLM', () => {
    const unmount = render(
      <LandingHero
        onLoadSample={async () => {}}
        onAddData={() => {}}
      />,
    );

    expect(document.body.textContent).toContain('Configure an API key or local model before asking questions');
    expect(document.body.textContent).toContain('The LLM receives your question plus relevant document passages and dataset context');
    expect(document.body.textContent).not.toContain('Load Sample Data & Try It');
    unmount();
  });
});

describe('ExampleQuestions onboarding presentation', () => {
  it('replaces unavailable web prompts with an actionable settings control', () => {
    const openSettings = vi.fn();
    useAppStore.getState().updateSettings({ webSearchProvider: 'none' });
    const unmount = render(
      <ExampleQuestions
        onSelect={() => {}}
        onOpenSettings={openSettings}
      />,
    );

    expect(document.body.textContent).not.toContain('Latest AI trends');
    expect(document.body.textContent).toContain('Web search is disabled');
    click(document.querySelector('[data-testid="examples-configure-web"]') as HTMLButtonElement);
    expect(openSettings).toHaveBeenCalledTimes(1);
    unmount();
  });
});

describe('DataSandbox sample loading feedback', () => {
  it('clears the previous sample error when retrying and announces per-file failure detail', async () => {
    const first = Promise.reject(new Error('Sample data partially loaded. 1 OK, 1 failed: broken.csv (HTTP 404).'));
    const second = deferred<void>();
    const loadSampleData = vi
      .fn<() => Promise<void>>()
      .mockReturnValueOnce(first)
      .mockReturnValueOnce(second.promise);
    const unmount = render(
      <DataSandbox
        open
        onClose={() => {}}
        addFiles={async () => {}}
        loadSampleData={loadSampleData}
        clearSandboxData={() => {}}
        removeSandboxDocument={() => {}}
        removeSandboxDataset={() => {}}
      />,
    );

    const sampleButton = Array.from(document.querySelectorAll('button')).find(button =>
      button.textContent?.includes('sample dataset'),
    ) as HTMLButtonElement;

    await act(async () => {
      sampleButton.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await first.catch(() => undefined);
    });
    const alert = document.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('broken.csv (HTTP 404)');

    await act(async () => {
      sampleButton.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      await Promise.resolve();
    });
    expect(document.querySelector('[role="alert"]')).toBeNull();
    expect(document.body.textContent).toContain('Loading sample data');

    await act(async () => {
      second.resolve();
      await second.promise;
    });
    unmount();
  });
});

describe('WebSearchSetting accessibility and guidance', () => {
  it('offers Mwmbl as the keyless default and explains its public index limits', () => {
    const updateSettings = vi.fn();
    const unmount = render(
      <WebSearchSetting
        settings={currentSettings({ webSearchProvider: 'mwmbl' })}
        updateSettings={updateSettings}
      />,
    );

    expect(document.querySelector('select')?.getAttribute('aria-label')).toBe('Web search provider');
    expect(document.body.textContent).toContain('Mwmbl (no key)');
    expect(document.body.textContent).not.toContain('DuckDuckGo');
    expect(document.querySelector('input[type="password"]')).toBeNull();
    expect(document.body.textContent).toContain('public independent index');
    expect(document.body.textContent).toContain('coverage and freshness vary');
    unmount();
  });

  it('labels provider and key inputs and explains current unavailable configuration', () => {
    const updateSettings = vi.fn();
    const unmount = render(
      <WebSearchSetting
        settings={currentSettings({ webSearchProvider: 'serper', serperApiKey: '' })}
        updateSettings={updateSettings}
      />,
    );

    expect(document.querySelector('select')?.getAttribute('aria-label')).toBe('Web search provider');
    expect(document.querySelector('input[type="password"]')?.getAttribute('aria-label')).toBe('Serper API key');
    expect(document.body.textContent).toContain('Serper is selected, but an API key is required');
    unmount();
  });
});
