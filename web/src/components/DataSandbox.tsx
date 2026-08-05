import { useState, useRef, useCallback } from 'react';
import { useAppStore } from '../store';

interface Props {
  open: boolean;
  onClose: () => void;
  addFiles: (files: FileList | File[]) => Promise<void>;
  loadSampleData: () => Promise<void>;
  clearSandboxData: () => void;
}

export function DataSandbox({ open, onClose, addFiles, loadSampleData, clearSandboxData }: Props) {
  const sandboxDatasets = useAppStore(s => s.sandboxDatasets);
  const sandboxDocuments = useAppStore(s => s.sandboxDocuments);
  const sandboxProcessing = useAppStore(s => s.sandboxProcessing);
  const removeSandboxDataset = useAppStore(s => s.removeSandboxDataset);
  const removeSandboxDocument = useAppStore(s => s.removeSandboxDocument);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isWorking, setIsWorking] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const arr = Array.from(files);
      if (arr.length === 0) return;
      setIsWorking(true);
      setErrorMsg(null);
      try {
        await addFiles(files);
      } catch (e) {
        setErrorMsg(e instanceof Error ? e.message : String(e));
      } finally {
        setIsWorking(false);
      }
    },
    [addFiles],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
    },
    [handleFiles],
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handlePickFiles = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFiles(e.target.files);
        e.target.value = '';
      }
    },
    [handleFiles],
  );

  const handleRemoveDataset = (name: string) => {
    removeSandboxDataset(name);
  };

  const handleRemoveDocument = (fileName: string) => {
    removeSandboxDocument(fileName);
  };

  const handleLoadSample = async () => {
    setIsWorking(true);
    try {
      await loadSampleData();
    } finally {
      setIsWorking(false);
    }
  };

  const handleClearAll = () => {
    if (confirm('Clear all loaded data? This empties the sandbox.')) {
      clearSandboxData();
    }
  };

  if (!open) return null;

  const isEmpty = sandboxDatasets.length === 0 && sandboxDocuments.length === 0;

  return (
    <div className="fixed inset-0 z-50 flex" onClick={onClose}>
      <div className="absolute inset-0 bg-black/30 animate-fade-in" />
      <div
        className="relative ml-auto w-full max-w-lg bg-white dark:bg-ink-900 shadow-2xl overflow-y-auto animate-slide-up"
        onClick={e => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-white dark:bg-ink-900 border-b border-ink-200 dark:border-ink-700 px-6 py-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Data Sandbox</h2>
            <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-0.5">
                {isEmpty
                  ? 'Empty. Drop your CSVs, PDFs, or text files to query them.'
                  : `${sandboxDatasets.length} dataset${sandboxDatasets.length === 1 ? '' : 's'} · ${sandboxDocuments.length} document${sandboxDocuments.length === 1 ? '' : 's'}`}
           </p>
         </div>
          <button
            onClick={onClose}
            className="text-ink-500 hover:text-ink-800 dark:hover:text-ink-200"
            type="button"
            aria-label="Close data sandbox"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
       </div>

        <div className="px-6 py-4 space-y-5">
          {errorMsg && (
            <div className="text-xs text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-900/30 rounded px-3 py-2">
              {errorMsg}
           </div>
          )}

          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`rounded-xl border-2 border-dashed transition px-6 py-8 text-center ${
              isDragOver
                ? 'border-brand-500 bg-brand-50 dark:bg-brand-900/30'
                : 'border-ink-300 dark:border-ink-700 bg-ink-50 dark:bg-ink-800/40'
            }`}
          >
            <svg
              className="w-10 h-10 mx-auto text-ink-400 dark:text-ink-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
           </svg>
            <p className="text-sm font-medium text-ink-700 dark:text-ink-200 mt-3">
              {isDragOver ? 'Drop to load' : 'Drop files here'}
           </p>
            <p className="text-[11px] text-ink-500 dark:text-ink-400 mt-1">
              or
           </p>
            <button
              type="button"
              onClick={handlePickFiles}
              disabled={isWorking}
              className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 bg-brand-600 hover:bg-brand-700 text-white text-xs font-medium rounded-lg transition disabled:opacity-50"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
             </svg>
              Browse files
           </button>
              <p className="text-[10px] text-ink-400 dark:text-ink-500 mt-3">
              CSV → Arquero table · PDF / MD / TXT / JSON → chunked + embedded
            </p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".csv,.pdf,.md,.markdown,.txt,.text,.json"
              onChange={handleFileChange}
              className="hidden"
            />
         </div>

          {sandboxProcessing.length > 0 && (
            <div className="space-y-1.5">
              {sandboxProcessing.map(p => (
                <div
                  key={p.fileName}
                  className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg border border-ink-200 dark:border-ink-700 text-xs"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    {p.status === 'processing' || p.status === 'embedding' ? (
                      <svg className="w-3.5 h-3.5 animate-spin text-brand-500 flex-shrink-0" fill="none" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.25" />
                        <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
                     </svg>
                    ) : p.status === 'error' ? (
                      <svg className="w-3.5 h-3.5 text-rose-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                     </svg>
                    ) : (
                      <svg className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                     </svg>
                    )}
                    <span className="font-mono truncate">{p.fileName}</span>
                 </div>
                  <span
                    className={`text-[10px] font-semibold uppercase flex-shrink-0 ${
                      p.status === 'error'
                        ? 'text-rose-600 dark:text-rose-400'
                        : p.status === 'done'
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : 'text-brand-600 dark:text-brand-400'
                    }`}
                  >
                    {p.status === 'embedding' ? 'embedding' : p.status}
                 </span>
               </div>
              ))}
           </div>
          )}

          {sandboxDatasets.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
                  Datasets
               </div>
                <span className="text-[10px] text-ink-400">{sandboxDatasets.length}</span>
             </div>
              <ul className="space-y-1.5">
                {sandboxDatasets.map(d => (
                  <li
                    key={d.name}
                    className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg border border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <svg className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                       </svg>
                        <span className="font-mono text-xs font-semibold text-ink-800 dark:text-ink-100 truncate">
                          {d.name}
                       </span>
                     </div>
                      <div className="text-[10px] text-ink-500 dark:text-ink-400 mt-0.5 ml-5.5">
                        {d.rowCount} row{d.rowCount === 1 ? '' : 's'} · {d.columns.length} col{d.columns.length === 1 ? '' : 's'}
                     </div>
                   </div>
                    <button
                      onClick={() => handleRemoveDataset(d.name)}
                      className="text-ink-400 hover:text-rose-500 transition flex-shrink-0"
                      type="button"
                      aria-label={`Remove ${d.name}`}
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                     </svg>
                   </button>
                 </li>
                ))}
             </ul>
           </div>
          )}

          {sandboxDocuments.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold uppercase tracking-wide text-ink-500 dark:text-ink-400">
                  Documents
               </div>
                <span className="text-[10px] text-ink-400">{sandboxDocuments.length}</span>
             </div>
              <ul className="space-y-1.5">
                {sandboxDocuments.map(d => (
                  <li
                    key={d.id}
                    className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg border border-ink-200 dark:border-ink-700 bg-white dark:bg-ink-800"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <svg className="w-3.5 h-3.5 text-brand-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                       </svg>
                        <span className="font-mono text-xs font-semibold text-ink-800 dark:text-ink-100 truncate">
                          {d.fileName}
                       </span>
                     </div>
                      <div className="text-[10px] text-ink-500 dark:text-ink-400 mt-0.5 ml-5.5">
                        {d.chunkCount} chunk{d.chunkCount === 1 ? '' : 's'} embedded
                     </div>
                   </div>
                    <button
                      onClick={() => handleRemoveDocument(d.fileName)}
                      className="text-ink-400 hover:text-rose-500 transition flex-shrink-0"
                      type="button"
                      aria-label={`Remove ${d.fileName}`}
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                     </svg>
                   </button>
                 </li>
                ))}
             </ul>
           </div>
          )}

          {!isEmpty && (
            <div className="pt-3 border-t border-ink-200 dark:border-ink-700 flex gap-2">
              <button
                onClick={handleLoadSample}
                disabled={isWorking}
                className="flex-1 px-3 py-2 text-xs font-medium text-ink-700 dark:text-ink-200 border border-ink-200 dark:border-ink-700 rounded-lg hover:bg-ink-50 dark:hover:bg-ink-800 disabled:opacity-50"
                type="button"
              >
                Load sample data
             </button>
              <button
                onClick={handleClearAll}
                disabled={isWorking}
                className="flex-1 px-3 py-2 text-xs font-medium text-rose-600 dark:text-rose-400 border border-rose-200 dark:border-rose-800 rounded-lg hover:bg-rose-50 dark:hover:bg-rose-900/30 disabled:opacity-50"
                type="button"
              >
                Clear all
             </button>
           </div>
          )}

          {isEmpty && (
            <button
              onClick={handleLoadSample}
              disabled={isWorking}
              className="w-full px-3 py-2 text-xs font-medium text-ink-700 dark:text-ink-200 border border-dashed border-ink-300 dark:border-ink-700 rounded-lg hover:bg-ink-50 dark:hover:bg-ink-800 disabled:opacity-50"
              type="button"
            >
              Or load the small sample dataset to try things out
           </button>
          )}
       </div>
     </div>
   </div>
  );
}
