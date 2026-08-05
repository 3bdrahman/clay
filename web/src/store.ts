import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatMessage, Settings, ModelInfo } from './lib/types';
import { LOCAL_DEFAULT_BASE_URL } from './lib/providers';

export interface SandboxDataset {
  name: string;
  fileName: string;
  columns: string[];
  rowCount: number;
  loadedAt: number;
  csv?: string;
  isSample?: boolean;
}

export interface SandboxDocument {
  id: string;
  fileName: string;
  source: string;
  chunkCount: number;
  loadedAt: number;
  chunks?: Array<{ id: string; text: string; page?: number }>;
}

export interface SandboxProcessing {
  fileName: string;
  status: 'processing' | 'embedding' | 'done' | 'error';
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

interface AppState {
  settings: Settings;
  conversations: Conversation[];
  activeConversationId: string | null;
  isRunning: boolean;
  availableModels: ModelInfo[];
  modelsLoading: boolean;
  modelsError: string | null;
  modelsFetchedAt: number;
  sandboxDatasets: SandboxDataset[];
  sandboxDocuments: SandboxDocument[];
  sandboxProcessing: SandboxProcessing[];
  updateSettings: (patch: Partial<Settings>) => void;
  createConversation: () => string;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  switchConversation: (id: string) => void;
  addMessage: (msg: ChatMessage) => void;
  updateMessage: (id: string, updater: (msg: ChatMessage) => ChatMessage) => void;
  clearMessages: () => void;
  setRunning: (running: boolean) => void;
  setModels: (models: ModelInfo[]) => void;
  setModelsLoading: (loading: boolean) => void;
  setModelsError: (err: string | null) => void;
  setLocalCatalog: (models: ModelInfo[]) => void;
  addSandboxDataset: (d: SandboxDataset) => void;
  addSandboxDocument: (d: SandboxDocument) => void;
  removeSandboxDataset: (name: string) => void;
  removeSandboxDocument: (fileName: string) => void;
  setSandboxProcessing: (items: SandboxProcessing[]) => void;
  updateSandboxProcessingItem: (fileName: string, patch: Partial<SandboxProcessing>) => void;
  clearSandbox: () => void;
  resetAll: () => void;
}

const DEFAULT_SETTINGS: Settings = {
  provider: 'nim',
  apiKey: '',
  embeddingApiKey: '',
  webSearchProvider: 'duckduckgo',
  serperApiKey: '',
  temperature: 0,
  maxRetries: 3,
  theme: 'system',
  localServerUrl: LOCAL_DEFAULT_BASE_URL,
  localModels: {
    routing: '',
    codeGen: '',
    answer: '',
    eval: '',
    embedding: '',
  },
  localCatalog: [],
  localCatalogFetchedAt: 0,
};

function trimMessage(msg: ChatMessage): ChatMessage {
  if (!msg.workflow) return msg;
  const { documents: _documents, webResults: _webResults, dataAnalysis: _dataAnalysis, ...lightWorkflow } = msg.workflow;
  return { ...msg, workflow: { ...lightWorkflow, documents: [], webResults: [] } as ChatMessage['workflow'] };
}

function makeConversation(title = 'New chat'): Conversation {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    title,
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

function deriveTitle(msg: ChatMessage): string {
  if (msg.role !== 'user') return '';
  const text = msg.content.trim();
  if (!text) return '';
  return text.length > 40 ? text.slice(0, 37) + '…' : text;
}

export const useAppStore = create<AppState>()(
  persist(
    set => {
      const initialConv = makeConversation();
      return {
        settings: DEFAULT_SETTINGS,
        conversations: [initialConv],
        activeConversationId: initialConv.id,
      isRunning: false,
      availableModels: [],
      modelsLoading: false,
      modelsError: null,
      modelsFetchedAt: 0,
      sandboxDatasets: [],
      sandboxDocuments: [],
      sandboxProcessing: [],
      updateSettings: patch =>
        set(state => ({
          settings: { ...state.settings, ...patch },
        })),
      createConversation: () => {
        const conv = makeConversation();
        set(state => ({
          conversations: [conv, ...state.conversations],
          activeConversationId: conv.id,
          isRunning: false,
        }));
        return conv.id;
      },
      deleteConversation: id =>
        set(state => {
          const remaining = state.conversations.filter(c => c.id !== id);
          const nextActive =
            state.activeConversationId === id
              ? (remaining[0]?.id ?? null)
              : state.activeConversationId;
          return {
            conversations: remaining,
            activeConversationId: nextActive,
            isRunning: false,
          };
        }),
      renameConversation: (id, title) =>
        set(state => ({
          conversations: state.conversations.map(c =>
            c.id === id ? { ...c, title, updatedAt: Date.now() } : c,
          ),
        })),
      switchConversation: id =>
        set({ activeConversationId: id, isRunning: false }),
      addMessage: msg =>
        set(state => {
          const activeId = state.activeConversationId;
          if (!activeId) return {};
          const conversations = state.conversations.map(c => {
            if (c.id !== activeId) return c;
            const messages = [...c.messages, msg];
            let title = c.title;
            if (c.title === 'New chat' && msg.role === 'user' && msg.content.trim()) {
              title = deriveTitle(msg);
            }
            return { ...c, messages, title, updatedAt: Date.now() };
          });
          if (!state.conversations.some(c => c.id === activeId)) {
            const conv = makeConversation(deriveTitle(msg) || 'New chat');
            conv.messages = [msg];
            return { conversations: [conv, ...state.conversations], activeConversationId: conv.id };
          }
          return { conversations };
        }),
      updateMessage: (id, updater) =>
        set(state => {
          const activeId = state.activeConversationId;
          if (!activeId) return {};
          return {
            conversations: state.conversations.map(c =>
              c.id === activeId
                ? {
                    ...c,
                    messages: c.messages.map(m => (m.id === id ? updater(m) : m)),
                    updatedAt: Date.now(),
                  }
                : c,
            ),
          };
        }),
      clearMessages: () =>
        set(state => {
          const activeId = state.activeConversationId;
          if (!activeId) return {};
          return {
            conversations: state.conversations.map(c =>
              c.id === activeId ? { ...c, messages: [], title: 'New chat', updatedAt: Date.now() } : c,
            ),
          };
        }),
      setRunning: running => set({ isRunning: running }),
      setModels: models => set({ availableModels: models, modelsFetchedAt: Date.now() }),
      setModelsLoading: loading => set({ modelsLoading: loading }),
      setModelsError: err => set({ modelsError: err }),
      setLocalCatalog: models =>
        set(state => ({
          settings: {
            ...state.settings,
            localCatalog: models,
            localCatalogFetchedAt: Date.now(),
          },
        })),
      addSandboxDataset: d =>
        set(state => ({
          sandboxDatasets: [...state.sandboxDatasets.filter(x => x.name !== d.name), d],
        })),
      addSandboxDocument: d =>
        set(state => ({
          sandboxDocuments: [...state.sandboxDocuments.filter(x => x.id !== d.id), d],
        })),
      removeSandboxDataset: name =>
        set(state => ({
          sandboxDatasets: state.sandboxDatasets.filter(x => x.name !== name),
        })),
      removeSandboxDocument: fileName =>
        set(state => ({
          sandboxDocuments: state.sandboxDocuments.filter(x => x.fileName !== fileName),
        })),
      setSandboxProcessing: items => set({ sandboxProcessing: items }),
      updateSandboxProcessingItem: (fileName, patch) =>
        set(state => ({
          sandboxProcessing: state.sandboxProcessing.map(p =>
            p.fileName === fileName ? { ...p, ...patch } : p,
          ),
        })),
      clearSandbox: () =>
        set({ sandboxDatasets: [], sandboxDocuments: [], sandboxProcessing: [] }),
      resetAll: () => {
        const freshConv = makeConversation();
        set({
          settings: DEFAULT_SETTINGS,
          conversations: [freshConv],
          activeConversationId: freshConv.id,
          isRunning: false,
          availableModels: [],
          modelsLoading: false,
          modelsError: null,
          modelsFetchedAt: 0,
          sandboxDatasets: [],
          sandboxDocuments: [],
          sandboxProcessing: [],
        });
      },
      };
    },
    {
      name: 'clay-settings-v1',
      version: 3,
      migrate: (persistedState, _version) => {
        const state = (persistedState ?? {}) as Partial<{
          settings: Partial<Settings>;
          messages: ChatMessage[];
          conversations: Conversation[];
          activeConversationId: string | null;
          sandboxDatasets: SandboxDataset[];
          sandboxDocuments: SandboxDocument[];
        }>;
        const mergedSettings: Settings = {
          ...DEFAULT_SETTINGS,
          ...(state.settings ?? {}),
          localModels: {
            ...DEFAULT_SETTINGS.localModels,
            ...((state.settings as { localModels?: Partial<typeof DEFAULT_SETTINGS.localModels> } | undefined)?.localModels ?? {}),
          },
        };

        let conversations: Conversation[] = [];
        let activeId: string | null = null;

        if (Array.isArray(state.conversations) && state.conversations.length > 0) {
          // Already migrated (v3+): keep as-is, but sanitize messages
          const now = Date.now();
          conversations = state.conversations.map(c => ({
            id: c.id || crypto.randomUUID(),
            title: c.title || 'New chat',
            messages: Array.isArray(c.messages) ? c.messages.slice(-50).map(trimMessage) : [],
            createdAt: c.createdAt || now,
            updatedAt: c.updatedAt || now,
          }));
          activeId = state.activeConversationId
            ?? (conversations[0]?.id ?? null);
          if (activeId && !conversations.some(c => c.id === activeId)) {
            activeId = conversations[0]?.id ?? null;
          }
        } else if (Array.isArray(state.messages) && state.messages.length > 0) {
          const messages = state.messages.map(trimMessage);
          const title = messages.find(m => m.role === 'user')?.content.trim()?.slice(0, 40) || 'Migrated chat';
          const conv = makeConversation(title);
          conv.messages = messages;
          conversations = [conv];
          activeId = conv.id;
        }

        if (conversations.length === 0) {
          const conv = makeConversation();
          conversations = [conv];
          activeId = conv.id;
        }

        return {
          settings: mergedSettings,
          conversations,
          activeConversationId: activeId,
          availableModels: [],
          modelsLoading: false,
          modelsError: null,
          modelsFetchedAt: 0,
          sandboxDatasets: Array.isArray(state.sandboxDatasets) ? (state.sandboxDatasets as never) : [],
          sandboxDocuments: Array.isArray(state.sandboxDocuments)
            ? (state.sandboxDocuments as never)
            : [],
          sandboxProcessing: [],
        };
      },
      partialize: state => ({
        settings: state.settings,
        conversations: state.conversations.map(c => ({
          ...c,
          messages: c.messages.slice(-50).map(trimMessage),
        })),
        activeConversationId: state.activeConversationId,
        sandboxDatasets: state.sandboxDatasets,
        sandboxDocuments: state.sandboxDocuments,
      }),
    }
  )
);
