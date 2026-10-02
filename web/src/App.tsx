import { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { ChatPanel } from './components/ChatPanel';
import { ConversationSidebar } from './components/ConversationSidebar';
import { useAppStore } from './store';
import { useClay } from './hooks/useClay';
import { SettingsPanelSuspense, DataSandboxSuspense } from './components/LazyPanels';
import { ErrorBoundary } from './components/ErrorBoundary';
import { initGlobalErrorHandler } from './lib/globalErrorHandler';

// Initialize global error handler
initGlobalErrorHandler();

export default function App() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [dataOpen, setDataOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const settings = useAppStore(s => s.settings);
  const clay = useClay();
  const { pickedModels, refreshModels, addFiles, loadSampleData, clearSandboxData, removeSandboxDocument, removeSandboxDataset, persistenceAvailable } = clay;
  const settingsProvider = settings.provider;
  const resetAll = useAppStore(s => s.resetAll);

  useEffect(() => {
    const root = document.documentElement;
    const apply = (theme: string) => {
      if (theme === 'dark') {
        root.classList.add('dark');
      } else if (theme === 'light') {
        root.classList.remove('dark');
      } else {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (prefersDark) {
          root.classList.add('dark');
        } else {
          root.classList.remove('dark');
        }
      }
    };
    apply(settings.theme);

    let handler: (() => void) | undefined;
    if (settings.theme === 'system') {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      handler = () => apply('system');
      mq.addEventListener('change', handler);
    }
    return () => {
      if (handler) {
        const mq = window.matchMedia('(prefers-color-scheme: dark)');
        mq.removeEventListener('change', handler);
      }
    };
  }, [settings.theme]);

  const openSettings = () => setSettingsOpen(true);
  const openData = () => setDataOpen(true);
  const toggleSidebar = () => setSidebarOpen(prev => !prev);

  return (
    <ErrorBoundary>
      <div className="h-screen flex flex-col bg-ink-50 dark:bg-ink-900">
        <Header
          onOpenSettings={openSettings}
          onOpenData={openData}
          onToggleSidebar={toggleSidebar}
          pickedModels={pickedModels}
          provider={settingsProvider}
        />
        {!persistenceAvailable && (
          <div
            role="status"
            className="px-4 py-2 text-center text-sm text-amber-800 bg-amber-50 border-b border-amber-200 dark:text-amber-200 dark:bg-amber-950/40 dark:border-amber-900"
          >
            Data may be lost on reload.
          </div>
        )}
        <div className="flex-1 flex min-h-0">
          <ConversationSidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
          <div className="flex-1 flex min-h-0">
            <ChatPanel clay={clay} onOpenData={openData} onOpenSettings={openSettings} />
          </div>
        </div>
        <SettingsPanelSuspense
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          refreshModels={refreshModels}
          pickedModels={pickedModels}
          resetAll={resetAll}
          clearSandboxData={clearSandboxData}
        />
        <DataSandboxSuspense
          open={dataOpen}
          onClose={() => setDataOpen(false)}
          addFiles={addFiles}
          loadSampleData={loadSampleData}
          clearSandboxData={clearSandboxData}
          removeSandboxDocument={removeSandboxDocument}
          removeSandboxDataset={removeSandboxDataset}
        />
      </div>
    </ErrorBoundary>
  );
}
