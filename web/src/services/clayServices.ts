// clayServices — one seam for the service bundle.
//
// Assembles the full set of wired services (LLM, embeddings, vectorstore,
// web search, data analyzer) from settings + a model catalog + a dataset
// source. The app (useClay) and the eval runner are two adapters at this
// seam; they differ only in where the catalog and datasets come from.
// Owning the assembly here keeps the wiring in one place: entries and
// queries always embed under the same model, and every adapter receives
// the same client configuration.

import * as aq from 'arquero';
import type { ModelInfo, Settings } from '../lib/types';
import type { PickedModels } from '../lib/models';
import { resolveModels } from '../lib/models';
import { resolveProviderEndpoint } from '../lib/providers';
import type { LLMClient } from '../lib/llm';
import { createLLMClient } from '../lib/llm';
import { createEmbeddingsClient, getSharedEmbeddingCache } from '../lib/embeddings';
import type { EmbeddingsClient } from '../lib/embeddings';
import { createVectorStore } from '../lib/vectorstore';
import type { VectorStore } from '../lib/vectorstore';
import { createWebSearchClient } from '../lib/websearch';
import type { WebSearchClient } from '../lib/websearch';
import { EMBEDDING_MODEL_ID } from '../lib/embeddingModel';
import { createDataAnalyzer } from './analyzer';
import type { DataAnalyzer, DatasetMeta } from './analyzer';

export interface ClayServiceBundle {
  llm: LLMClient;
  embeddings: EmbeddingsClient;
  vectorstore: VectorStore;
  webSearch: WebSearchClient;
  analyzer: DataAnalyzer;
  pickedModels: PickedModels;
  /**
   * Tear down worker-backed services (embeddings, analyzer sandbox) once
   * pending work drains. Adapters call this when they replace the bundle so
   * recreated clients do not leak workers.
   */
  dispose: () => void;
}

export interface ClayServiceBundleInput {
  settings: Settings;
  /** Model catalog the picks resolve against — adapters fetch it their own way. */
  catalog: ModelInfo[];
  /** Analyzer dataset tables keyed by dataset name. The 'aq' namespace entry is added here. */
  analyzerTables: Map<string, unknown>;
  /** Column/row metadata for the analyzer datasets. */
  analyzerMetadata: DatasetMeta;
}

export function createClayServiceBundle(input: ClayServiceBundleInput): ClayServiceBundle {
  const { settings, catalog, analyzerTables, analyzerMetadata } = input;
  const endpoint = resolveProviderEndpoint(settings);

  // The fetched catalog is authoritative: local picks validate against it,
  // so an adapter can never pick a model that is missing from its own catalog.
  const { picked } = resolveModels({ ...settings, localCatalog: catalog }, catalog);

  // The shared session cache survives bundle recreation: re-creating the
  // bundle on settings/catalog/dataset changes must not drop cache warmth.
  const embeddings = createEmbeddingsClient({ cache: getSharedEmbeddingCache() });
  // Entries and queries embed under the same fixed local model — the model
  // never changes across adapters, so stored vectors always match queries.
  // MMR reranking ships off deliberately (useMMR defaults false): the dense
  // path already fetches 3x top-K candidates and the HyDE reranker supplies
  // diversity upstream, while MMR trades top-1 relevance for diversity —
  // near-duplicate chunks reinforce the same context and rarely hurt the
  // judge's grounded-answer check. Revisit when eval baseline data supports
  // the relevance/diversity tradeoff.
  const vectorstore = createVectorStore(embeddings, {
    embeddingModel: EMBEDDING_MODEL_ID,
  });
  const webSearch = createWebSearchClient(settings);
  const llm = createLLMClient({
    baseUrl: endpoint.baseUrl,
    apiKey: endpoint.apiKey,
    temperature: settings.temperature,
    providerLabel: endpoint.providerLabel,
  });

  const datasets = new Map<string, unknown>(analyzerTables);
  datasets.set('aq', aq);
  const analyzer = createDataAnalyzer({
    llm,
    datasets,
    metadata: analyzerMetadata,
    codeGenModel: picked.chat,
    maxToolLoopTokens: settings.maxToolLoopTokens,
  });

  const dispose = (): void => {
    embeddings.dispose?.();
    analyzer.dispose();
  };

  return { llm, embeddings, vectorstore, webSearch, analyzer, pickedModels: picked, dispose };
}
