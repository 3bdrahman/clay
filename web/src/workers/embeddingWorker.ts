// embeddingWorker — in-browser embeddings via transformers.js.
//
// Runs the fixed feature-extraction pipeline off the main thread so embedding
// a document never freezes the UI. The model (~23MB, q8-quantized) downloads
// from the HuggingFace CDN on first use and is cached by the browser after.
// Protocol: {type:'embed', id, texts} → {type:'result', id, embeddings} |
// {type:'error', id, message}.

import { pipeline, env, type FeatureExtractionPipeline } from '@huggingface/transformers';
import { EMBEDDING_MODEL_ID } from '../lib/embeddingModel';

env.allowLocalModels = false;

export interface EmbedWorkerRequest {
  type: 'embed';
  id: number;
  texts: string[];
}

export type EmbedWorkerResponse =
  | { type: 'result'; id: number; embeddings: number[][] }
  | { type: 'error'; id: number; message: string };

let extractor: FeatureExtractionPipeline | null = null;
let loading: Promise<FeatureExtractionPipeline> | null = null;

function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (extractor) return Promise.resolve(extractor);
  if (!loading) {
    loading = pipeline('feature-extraction', EMBEDDING_MODEL_ID, { dtype: 'q8' })
      .then((p) => {
        extractor = p;
        return p;
      })
      .catch((e: unknown) => {
        loading = null;
        throw e;
      });
  }
  return loading;
}

async function handleEmbed(msg: EmbedWorkerRequest): Promise<void> {
  try {
    const ex = await getExtractor();
    const output = await ex(msg.texts, { pooling: 'mean', normalize: true });
    const dim = output.dims[1] ?? 0;
    const data = output.data as Float32Array;
    const embeddings: number[][] = [];
    for (let i = 0; i < msg.texts.length; i += 1) {
      embeddings.push(Array.from(data.subarray(i * dim, (i + 1) * dim)));
    }
    self.postMessage({ type: 'result', id: msg.id, embeddings } satisfies EmbedWorkerResponse);
  } catch (e) {
    self.postMessage({
      type: 'error',
      id: msg.id,
      message: e instanceof Error ? e.message : String(e),
    } satisfies EmbedWorkerResponse);
  }
}

self.addEventListener('message', (ev: MessageEvent<EmbedWorkerRequest>) => {
  if (ev.data.type !== 'embed') return;
  void handleEmbed(ev.data);
});
