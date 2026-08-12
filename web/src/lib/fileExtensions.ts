export const SUPPORTED_FILE_EXTENSIONS = ['.csv', '.pdf', '.md', '.markdown', '.txt', '.text', '.json'] as const;
export const DOCUMENT_EXTENSIONS = ['.pdf', '.md', '.markdown', '.txt', '.text', '.json'] as const;
export const DATASET_EXTENSIONS = ['.csv'] as const;
export const ALL_EXTENSIONS = [...DATASET_EXTENSIONS, ...DOCUMENT_EXTENSIONS] as const;

// For accept attribute (comma-separated)
export const ACCEPT_EXTENSIONS = '.csv,.pdf,.md,.markdown,.txt,.text,.json';

// Helper functions
export function isSupportedExtension(filename: string): boolean {
  const lower = filename.toLowerCase();
  return SUPPORTED_FILE_EXTENSIONS.some(ext => lower.endsWith(ext));
}

export function isDatasetExtension(filename: string): boolean {
  const lower = filename.toLowerCase();
  return DATASET_EXTENSIONS.some(ext => lower.endsWith(ext));
}

export function isDocumentExtension(filename: string): boolean {
  const lower = filename.toLowerCase();
  return DOCUMENT_EXTENSIONS.some(ext => lower.endsWith(ext));
}