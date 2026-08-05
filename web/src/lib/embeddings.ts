export type EmbeddingInputType = 'query' | 'passage';

export interface EmbedOptions {
  inputType?: EmbeddingInputType;
}

export interface EmbeddingsClient {
  embed(input: string | string[], opts?: EmbedOptions): Promise<number[][]>;
}

export class EmbeddingsConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EmbeddingsConfigError';
  }
}

export interface EmbeddingsClientConfig {
  baseUrl: string;
  apiKey: string;
  embeddingModel: string;
  providerLabel?: string;
}

// NVIDIA NIM asymmetric retrieval models (e.g. nv-embedqa-e5-v5,
// nv-embedqa-mistral-7b-v2) reject requests without `input_type`. Symmetric
// embedding models (e.g. llama-nemotron-embed-v1, nomic-embed-text) accept either.
const ASYMMETRIC_MODEL_PATTERNS: readonly RegExp[] = [
  /nv-embedqa-e5/,
  /nv-embedqa-mistral/,
];

function isAsymmetricModel(modelId: string): boolean {
  const lower = modelId.toLowerCase();
  return ASYMMETRIC_MODEL_PATTERNS.some(p => p.test(lower));
}

export function createEmbeddingsClient(config: EmbeddingsClientConfig): EmbeddingsClient {
  const baseUrl = config.baseUrl.replace(/\/+$/, '');
  const apiKey = config.apiKey;
  const embeddingModel = config.embeddingModel;
  const providerLabel = config.providerLabel ?? 'provider';

  async function embed(input: string | string[], opts?: EmbedOptions): Promise<number[][]> {
    if (!baseUrl) {
      throw new EmbeddingsConfigError(`${providerLabel} base URL is empty. Open Settings and configure it.`);
    }
    if (!embeddingModel) {
      throw new EmbeddingsConfigError(
        `${providerLabel} requires an embedding model. Pick one in Settings.`,
      );
    }

    const body: Record<string, unknown> = {
      model: embeddingModel,
      input: Array.isArray(input) ? input : [input],
    };
    if (isAsymmetricModel(embeddingModel)) {
      body.input_type = opts?.inputType ?? 'passage';
    }

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    const resp = await fetch(`${baseUrl}/embeddings`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`Embedding API error ${resp.status}: ${await resp.text()}`);

    const data = await resp.json();
    return (data.data || []).map((d: { embedding: number[] }) => d.embedding);
  }

  return { embed };
}
