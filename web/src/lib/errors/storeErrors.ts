import { RagError, RagErrorCode } from './base';

// ============================================================================
// Vector Store Errors
// ============================================================================

export class VectorStoreCorruptedError extends RagError {
  constructor(reason: string, cause?: Error) {
    super({
      code: RagErrorCode.VECTOR_STORE_CORRUPTED,
      message: `Vector store corrupted: ${reason}. Try clearing data and re-indexing your documents.`,
      cause,
      retryable: false,
      context: { reason },
    });
  }
}

export class VectorStoreQuotaExceededError extends RagError {
  constructor(cause?: Error) {
    super({
      code: RagErrorCode.VECTOR_STORE_QUOTA_EXCEEDED,
      message: 'IndexedDB storage quota exceeded. Clear old documents or use a browser with higher storage limits.',
      cause,
      retryable: false,
      context: {},
    });
  }
}