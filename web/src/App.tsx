import { lazy, Suspense, useEffect, useState } from 'react';
import { Header } from './components/Header';
import { ChatPanel } from './components/ChatPanel';
import { ConversationSidebar } from './components/ConversationSidebar';
import { useAppStore } from './store';
import { useClay } from './hooks/useClay';

const SettingsPanel = lazy(() => import('./components/SettingsPanel').then(m => ({ default: m.SettingsPanel })));
const DataSandbox = lazy(() => import('./components/DataSandbox').then(m => ({ default: m.DataSandbox })));

export default function App() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [dataOpen, setDataOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const settings = useAppStore(s => s.settings);
  const { pickedModels, refreshModels, addFiles, loadSampleData, clearSandboxData } = useClay();
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
    <div className="h-screen flex flex-col bg-ink-50 dark:bg-ink-900">
      <Header
        onOpenSettings={openSettings}
        onOpenData={openData}
        onToggleSidebar={toggleSidebar}
        pickedModels={pickedModels}
        provider={settingsProvider}
      />
      <div className="flex-1 flex min-h-0">
        <ConversationSidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        <div className="flex-1 flex min-h-0">
          <ChatPanel onOpenSettings={openSettings} onOpenData={openData} />
        </div>
      </div>
      <Suspense fallback={null}>
        <SettingsPanel
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          refreshModels={refreshModels}
          pickedModels={pickedModels}
          resetAll={resetAll}
          clearSandboxData={clearSandboxData}
        />
      </Suspense>
      <Suspense fallback={null}>
        <DataSandbox
          open={dataOpen}
          onClose={() => setDataOpen(false)}
          addFiles={addFiles}
          loadSampleData={loadSampleData}
          clearSandboxData={clearSandboxData}
        />
      </Suspense>
    </div>
  );
}
