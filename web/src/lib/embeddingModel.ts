// The fixed local embedding model. Embeddings never leave the browser: the
// model (~23MB, q8) downloads from the HuggingFace CDN on first use and is
// cached by the browser afterwards. 384-dim, symmetric — no query/passage
// prefixes are needed.

export const EMBEDDING_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
