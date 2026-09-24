// Core type definitions for Clay

import type { RagErrorCode } from './errors';

export interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: {
      type: 'object';
      properties: Record<string, unknown>;
      required: string[];
    };
  };
}

export interface LocalModelPicks {
  chat: string;
  embeddings: string;
}

export type ProviderKind = 'openrouter' | 'groq' | 'local';

export interface PickedModelsOverride {
  chatModel?: string;
  embedding?: string;
}

export interface Settings {
  provider: ProviderKind;
  // Provider API keys - each provider has its own key
  openrouterApiKey: string;
  groqApiKey: string;
  embeddingApiKey: string;
  // Legacy field for backward compat (migration)
  apiKey: string;
  webSearchProvider: 'serper' | 'duckduckgo' | 'none';
  serperApiKey: string;
  temperature: number;
  maxRetries: number;
  vectorstoreInitialK?: number;
  maxToolLoopTokens?: number; // cumulative tool-loop token budget per analysis (undefined = default)
  theme: 'light' | 'dark' | 'system';
  localServerUrl: string;
  localModels: LocalModelPicks;
  localCatalog: ModelInfo[];
  localCatalogFetchedAt: number;
  // User-overridable model selections (empty = auto-pick)
  // chatModel: single model for routing, codeGen, answer, eval
  // embedding: separate embedding model
  pickedModelsOverride: PickedModelsOverride;
}

export interface Document {
  id: string;
  content: string;
  source: string;
  page?: number;
  score?: number;
  metadata?: Record<string, unknown>;
}

export interface DatasetSummary {
  name: string;
  fileName?: string;
  rowCount: number;
  columns: string[];
  /** Optional sample rows for type detection */
  sampleRows?: Array<Record<string, unknown>>;
}

export interface DataAnalysisResult {
  type: 'data_analysis';
  question: string;
  code: string;
  explanation: string;
  resultType: 'table' | 'scalar' | 'chart' | 'error';
  result: unknown;
  chartConfig?: ChartConfig;
  attempts: number;
  durationMs: number;
  timestamp: number;
  insights?: Insight[];
  mode?: 'tools' | 'single-shot' | 'fallback';
  fallbackReason?: string;
  toolTrace?: Array<{ tool: string; calls: number; durationMs: number; tokensUsed: number }>;
  partial?: boolean;
}

export interface ChartConfig {
  type: 'bar' | 'line' | 'pie';
  title: string;
  xKey: string;
  yKeys: string[];
  data: Array<Record<string, unknown>>;
}

export interface WebResult {
  type: 'web_search';
  title: string;
  content: string;
  url?: string;
  score?: number;
}

export type SourceType = 'vectorstore' | 'python' | 'websearch';

export interface StepTrace {
  id: string;
  node: string;
  label: string;
  status: 'pending' | 'running' | 'done' | 'error' | 'skipped';
  startedAt?: number;
  finishedAt?: number;
  durationMs?: number;
  detail?: string;
  meta?: Record<string, unknown>;
  retries?: Array<{ attempt: number; error: string; delayMs: number }>;
  tokensUsed?: number;
}

export interface WorkflowState {
  question: string;
  routing?: SourceType;
  documents: Document[];
  webResults: WebResult[];
  dataAnalysis?: DataAnalysisResult;
  answer?: string;
  citations: Citation[];
  retryCount: number;
  steps: StepTrace[];
  startedAt: number;
  finishedAt?: number;
  error?: {
    code: RagErrorCode;
    message: string;
    step?: string;
    retryable: boolean;
  };
}

export interface Citation {
  source: string;
  page?: number;
  excerpt: string;
  type: 'vectorstore' | 'websearch' | 'python';
}

export interface ModelInfo {
  id: string;
  ownedBy: string;
  created: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  workflow?: WorkflowState;
  error?: string;
  streaming?: boolean;
}

export interface LLMRequest {
  system?: string;
  messages: Array<LLMMessage>;
  jsonMode?: boolean;
  temperature?: number;
  maxTokens?: number;
  model?: string;
  tools?: ToolDefinition[];
  toolChoice?: 'auto' | 'none';
}

export type LLMMessage =
  | { role: 'user'; content: string }
  | { role: 'system'; content: string }
  | { role: 'assistant'; content: string; toolCalls?: ToolCall[] }
  | { role: 'tool'; content: string; toolCallId: string };

export interface LLMResponse {
  content: string;
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
  };
  model?: string;
  toolCalls?: ToolCall[];
  finishReason?: string;
}

export interface EmbeddingRequest {
  input: string | string[];
  model?: string;
}

export interface EmbeddingResponse {
  embeddings: number[][];
  model?: string;
}

/** Structural + provenance metadata for a single chunk, used as the cache key and citation source. */
export interface ChunkMetadata {
  source: string;
  sourceHash: string;
  page?: number;
  heading?: string;
  charStart: number;
  charEnd: number;
  chunkIndex: number;
  tokenCount: number;
  modelId: string;
  updatedAt?: number;
}

/** Tunable parameters for the retrieval phase (top-k, score gate, MMR, hybrid fusion). */
export interface RetrievalConfig {
  topK: number;
  scoreThreshold: number;
  useMMR: boolean;
  mmrLambda: number;
  useHybrid: boolean;
  hybridAlpha: number;
}

/** LLM-as-judge relevance verdict for a single retrieved document. */
export interface GradeResult {
  docId: string;
  relevant: boolean;
  score?: number;
}

/** Per-question evaluation metrics with per-stage latency breakdown (v2 eval harness). */
export interface EvalResultV2 {
  questionId: string;
  nDCGAtK: number;
  MRR: number;
  recallAtK: number;
  latencyMs: number;
  stageLatencies: {
    retrieveMs: number;
    gradeMs: number;
    generateMs: number;
    evaluateMs: number;
  };
  routingCorrect: boolean;
  error?: string;
}

export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface Insight {
  finding: string;
  evidence: string;
  confidence: 'high' | 'medium' | 'low';
  implication?: string;
}

/** Default retrieval configuration: top-k 8, score gate 0.25, hybrid fusion on (alpha 0.5), MMR off. */
export const DEFAULT_RETRIEVAL_CONFIG: RetrievalConfig = {
  topK: 8,
  scoreThreshold: 0.25,
  useMMR: false,
  mmrLambda: 0.5,
  useHybrid: true,
  hybridAlpha: 0.5,
};
