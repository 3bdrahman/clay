import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatMessage, Settings, ModelInfo } from './lib/types';
import { LOCAL_DEFAULT_BASE_URL } from './lib/providers';
import { migrateLegacyLocalModels, type LegacyLocalModelPicks } from './lib/localModelsMigrate';
import type { ProviderKind } from './lib/types';

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
  nimApiKey: '',
  openrouterApiKey: '',
  groqApiKey: '',
  togetherApiKey: '',
  openaiApiKey: '',
  anthropicApiKey: '',
  apiKey: '', // legacy field for migration
  embeddingApiKey: '',
  webSearchProvider: 'duckduckgo',
  serperApiKey: '',
  temperature: 0,
  maxRetries: 3,
  vectorstoreInitialK: 8,
  theme: 'system',
  localServerUrl: LOCAL_DEFAULT_BASE_URL,
  localModels: {
    chat: '',
    embeddings: '',
  },
  localCatalog: [],
  localCatalogFetchedAt: 0,
  pickedModelsOverride: {
    routing: '',
    codeGen: '',
    answer: '',
    eval: '',
    embedding: '',
  },
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
      return {
        settings: DEFAULT_SETTINGS,
        conversations: [],
        activeConversationId: null,
        isRunning: false,
        availableModels: [],
        modelsLoading: false,
        modelsError: null,
        modelsFetchedAt: 0,
        sandboxDatasets: [],
        sandboxDocuments: [],
        sandboxProcessing: [],
        updateSettings: patch =>
          set(state => {
            const next = { ...state.settings, ...patch };
            if (
              patch.provider !== undefined &&
              patch.provider !== state.settings.provider &&
              state.settings.provider === 'local' &&
              patch.provider !== 'local'
            ) {
              next.localCatalog = [];
              next.localCatalogFetchedAt = 0;
            }
            return { settings: next };
          }),
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
            let activeId = state.activeConversationId;
            // If no active conversation, create one
            if (!activeId || !state.conversations.some(c => c.id === activeId)) {
              const conv = makeConversation(msg.role === 'user' ? deriveTitle(msg) : 'New chat');
              conv.messages = [msg];
              return {
                conversations: [conv, ...state.conversations],
                activeConversationId: conv.id,
              };
            }
            const conversations = state.conversations.map(c => {
              if (c.id !== activeId) return c;
              const messages = [...c.messages, msg];
              let title = c.title;
              if (c.title === 'New chat' && msg.role === 'user' && msg.content.trim()) {
                title = deriveTitle(msg);
              }
              return { ...c, messages, title, updatedAt: Date.now() };
            });
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
      version: 5,
      migrate: (persistedState, _version) => {
        const state = (persistedState ?? {}) as Partial<{
          settings: Partial<Settings>;
          messages: ChatMessage[];
          conversations: Conversation[];
          activeConversationId: string | null;
          sandboxDatasets: SandboxDataset[];
          sandboxDocuments: SandboxDocument[];
        }>;
        const persistedLocalModels = (state.settings as {
          localModels?: Partial<LegacyLocalModelPicks>;
        } | undefined)?.localModels;
        const persistedSettings = state.settings ?? {};

        // Migrate legacy single apiKey to provider-specific key
        const legacyApiKey = persistedSettings.apiKey as string | undefined;
        const provider = (persistedSettings.provider as ProviderKind) ?? 'nim';
        const providerApiKeyField = {
          nim: 'nimApiKey',
          openrouter: 'openrouterApiKey',
          groq: 'groqApiKey',
          together: 'togetherApiKey',
          local: '',
        }[provider];

        const mergedSettings: Settings = {
          ...DEFAULT_SETTINGS,
          ...((persistedSettings as Omit<Partial<Settings>, 'localModels'> | undefined) ?? {}),
          localModels: migrateLegacyLocalModels(persistedLocalModels),
          pickedModelsOverride: persistedSettings.pickedModelsOverride ?? DEFAULT_SETTINGS.pickedModelsOverride,
          [providerApiKeyField]: legacyApiKey ?? '',
        };

        let conversations: Conversation[] = [];
        let activeId: string | null = null;

        if (Array.isArray(state.conversations) && state.conversations.length > 0) {
          // Already migrated (v3+): keep as-is, but sanitize messages and filter out empty conversations
          const now = Date.now();
          conversations = state.conversations
            .filter(c => Array.isArray(c.messages) && c.messages.length > 0)
            .map(c => ({
              id: c.id || crypto.randomUUID(),
              title: c.title || 'New chat',
              messages: c.messages.slice(-50).map(trimMessage),
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
          // Don't create a default conversation - start empty
          conversations = [];
          activeId = null;
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
        conversations: state.conversations
          .filter(c => c.messages.length > 0) // Only persist conversations with messages
          .map(c => ({
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
