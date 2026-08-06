import { useAppStore } from '../store';
import type { PickedModels } from '../lib/models';
import { useConfirm } from '../hooks/useConfirm';

interface Props {
  onOpenSettings: () => void;
  onOpenData: () => void;
  onToggleSidebar: () => void;
  pickedModels: PickedModels;
  provider: 'nim' | 'local';
}

export function Header({ onOpenSettings, onOpenData, onToggleSidebar, pickedModels, provider }: Props) {
  const settings = useAppStore(s => s.settings);
  const messageCount = useAppStore(s => {
    const conv = s.conversations.find(c => c.id === s.activeConversationId);
    return conv?.messages.length ?? 0;
  });
  const clearMessages = useAppStore(s => s.clearMessages);
  const availableModels = useAppStore(s => s.availableModels);
  const modelsLoading = useAppStore(s => s.modelsLoading);
  const sandboxDatasets = useAppStore(s => s.sandboxDatasets);
  const sandboxDocuments = useAppStore(s => s.sandboxDocuments);
  const [confirm, renderConfirmDialog] = useConfirm();

  const clearChat = async () => {
    const ok = await confirm({
      title: 'Clear chat history?',
      message: 'All messages in the current conversation will be removed.',
      confirmLabel: 'Clear chat',
      destructive: true,
    });
    if (ok) clearMessages();
  };

  const isLocal = provider === 'local';
  const hasKey = !isLocal && settings.apiKey.length > 0;
  const hasLocalModel = isLocal && !!pickedModels.answer;
  const connected = hasKey || hasLocalModel;
  const answerModel = pickedModels.answer ?? (isLocal ? 'No local answer model set' : availableModels[0]?.id ?? 'Loading models…');
  const providerLabel = isLocal ? 'Local server' : 'NVIDIA NIM';
  const providerDocsUrl = isLocal
    ? (settings.localServerUrl || '#')
    : 'https://docs.nvidia.com/nim/';
  const providerDocsLabel = isLocal ? 'Local server' : 'NVIDIA NIM docs';

  return (
    <header className="border-b border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-900 px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-3 min-w-0">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center text-white shadow-sm flex-shrink-0">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
         </svg>
       </div>
        <div className="min-w-0">
          <h1 className="text-base font-bold text-ink-900 dark:text-ink-50 leading-tight">Clay</h1>
          <div className="text-[10px] text-ink-500 dark:text-ink-400 flex items-center gap-1.5 truncate">
            <span
              className={`inline-block w-1.5 h-1.5 rounded-full ${
                connected
                  ? 'bg-emerald-500 animate-pulse'
                  : isLocal
                  ? 'bg-amber-500'
                  : 'bg-ink-300'
              }`}
            />
            <span>{providerLabel}</span>
            <span className="opacity-50">·</span>
            <span className="font-mono truncate" title={answerModel}>
              {modelsLoading && !isLocal ? 'Loading models…' : answerModel}
           </span>
            {messageCount > 0 && (
              <>
                <span className="opacity-50">·</span>
                <span>{messageCount} msg</span>
              </>
            )}
       </div>
     </div>
   </div>

      <div className="flex items-center gap-1.5">
        <button
          onClick={onToggleSidebar}
          className="px-2.5 py-1.5 text-xs text-ink-600 dark:text-ink-300 hover:bg-ink-100 dark:hover:bg-ink-800 rounded-lg transition"
          title="Conversations"
          type="button"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h7" />
          </svg>
        </button>
        {messageCount > 0 && (
          <button
            onClick={clearChat}
            className="px-2.5 py-1.5 text-xs text-ink-600 dark:text-ink-300 hover:bg-ink-100 dark:hover:bg-ink-800 rounded-lg transition"
            title="Clear chat"
            type="button"
            aria-label="Clear chat history"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
           </svg>
         </button>
        )}
        <a
          href={providerDocsUrl}
          target="_blank"
          rel="noreferrer"
          className="px-2.5 py-1.5 text-xs text-ink-600 dark:text-ink-300 hover:bg-ink-100 dark:hover:bg-ink-800 rounded-lg transition hidden sm:inline-flex items-center gap-1"
          title={providerDocsLabel}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
         </svg>
          {providerDocsLabel}
       </a>
        <button
          onClick={onOpenData}
          className="px-2.5 py-1.5 text-xs text-ink-700 dark:text-ink-200 hover:bg-ink-100 dark:hover:bg-ink-800 rounded-lg transition flex items-center gap-1.5"
          title="Data sandbox"
          type="button"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10a2 2 0 002 2h12a2 2 0 002-2V9a2 2 0 00-2-2h-5l-2-2H6a2 2 0 00-2 2zM4 13h16" />
         </svg>
          <span className="hidden sm:inline">Data</span>
          {(sandboxDatasets.length > 0 || sandboxDocuments.length > 0) && (
            <span className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full bg-brand-100 dark:bg-brand-900/40 text-brand-700 dark:text-brand-300 text-[10px] font-bold">
              {sandboxDatasets.length + sandboxDocuments.length}
           </span>
          )}
       </button>
        <button
          onClick={onOpenSettings}
          className="px-2.5 py-1.5 text-xs text-ink-700 dark:text-ink-200 hover:bg-ink-100 dark:hover:bg-ink-800 rounded-lg transition flex items-center gap-1.5"
          title="Settings"
          type="button"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
          <span className="hidden sm:inline">Settings</span>
      </button>
     </div>
     {renderConfirmDialog()}
   </header>
  );
}
