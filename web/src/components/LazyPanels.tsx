/* eslint-disable react-refresh/only-export-components */
import { lazy, Suspense } from 'react';

/**
 * Wraps a dynamic import with a retry mechanism. If the chunk fails to load
 * (e.g. because a new version was deployed and the old chunk is 404), this will
 * automatically trigger a hard reload of the page to fetch the new index.html.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function lazyWithRetry<T extends React.ComponentType<any>>(
  componentImport: () => Promise<{ default: T }>
) {
  return lazy(async () => {
    try {
      const component = await componentImport();
      window.sessionStorage.removeItem('chunk-failed-reload');
      return component;
    } catch (error: any) {
      const msg = error?.message || '';
      const isChunkLoadError = 
        error?.name === 'TypeError' || 
        msg.includes('dynamically imported module') || 
        msg.includes('fetch') ||
        msg.includes('load');
        
      if (isChunkLoadError) {
        if (!window.sessionStorage.getItem('chunk-failed-reload')) {
          window.sessionStorage.setItem('chunk-failed-reload', 'true');
          window.location.reload();
          // Return a never-resolving promise so React Suspense doesn't crash before reload
          return new Promise<{ default: T }>(() => {});
        }
      }
      throw error;
    }
  });
}

/**
 * Panel loading placeholder. Matches the backdrop-blurred, centered-modal
 * style used by SettingsPanel and DataSandbox so the user sees a consistent
 * loading affordance instead of a blank screen while the chunk loads.
 */
export function PanelFallback() {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      aria-busy="true"
      aria-live="polite"
      role="status"
    >
      <div className="bg-white dark:bg-ink-800 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] flex items-center justify-center p-12">
        <div className="flex flex-col items-center gap-3">
          <div
            className="w-8 h-8 border-4 border-brand-200 border-t-brand-600 rounded-full animate-spin"
            aria-hidden="true"
          />
          <p className="text-sm text-ink-500 dark:text-ink-400">Loading…</p>
        </div>
      </div>
    </div>
  );
}

/**
 * Lazy-loaded SettingsPanel. Source path is explicit (no extension) so it
 * matches every other import in the codebase — Vite resolves `./SettingsPanel`
 * to `./SettingsPanel.tsx` automatically.
 */
export const SettingsPanelLazy = lazyWithRetry(() =>
  import('./SettingsPanel').then((m) => ({ default: m.SettingsPanel })),
);

/**
 * Lazy-loaded DataSandbox. Same convention as SettingsPanelLazy.
 */
export const DataSandboxLazy = lazyWithRetry(() =>
  import('./DataSandbox').then((m) => ({ default: m.DataSandbox })),
);

/**
 * Lazy-loaded ChartRenderer. ChartRenderer uses a default export, so no
 * `.then` shim is needed — but the path style matches the others (no
 * extension, kebab-consistent root).
 */
export const ChartRendererLazy = lazyWithRetry(() => import('./ChartRenderer'));

/**
 * Re-export with Suspense wrappers already applied. Callers can drop these
 * in directly and get a consistent loading state without repeating the
 * <Suspense fallback={…}> ceremony at every site.
 */
export function SettingsPanelSuspense(props: React.ComponentProps<typeof SettingsPanelLazy>) {
  return (
    <Suspense fallback={<PanelFallback />}>
      <SettingsPanelLazy {...props} />
    </Suspense>
  );
}

export function DataSandboxSuspense(props: React.ComponentProps<typeof DataSandboxLazy>) {
  return (
    <Suspense fallback={<PanelFallback />}>
      <DataSandboxLazy {...props} />
    </Suspense>
  );
}
