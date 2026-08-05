// Core type definitions for Clay

export interface LocalModelPicks {
  routing: string;
  codeGen: string;
  answer: string;
  eval: string;
  embedding: string;
}

export type ProviderKind = 'nim' | 'local';

export interface Settings {
  provider: ProviderKind;
  apiKey: string;
  embeddingApiKey: string;
  webSearchProvider: 'serper' | 'duckduckgo' | 'none';
  serperApiKey: string;
  temperature: number;
  maxRetries: number;
  theme: 'light' | 'dark' | 'system';
  localServerUrl: string;
  localModels: LocalModelPicks;
  localCatalog: ModelInfo[];
  localCatalogFetchedAt: number;
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
  rowCount: number;
  columns: string[];
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
  error?: string;
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
  messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>;
  jsonMode?: boolean;
  temperature?: number;
  maxTokens?: number;
  model?: string;
}

export interface LLMResponse {
  content: string;
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
  };
  model?: string;
}

export interface EmbeddingRequest {
  input: string | string[];
  model?: string;
}

export interface EmbeddingResponse {
  embeddings: number[][];
  model?: string;
}
