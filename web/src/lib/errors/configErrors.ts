import { RagError, RagErrorCode } from './base';

// ============================================================================
// Configuration Errors
// ============================================================================

export class NoProviderError extends RagError {
  constructor(provider: string, cause?: Error) {
    const message =
      provider === 'local' || provider === 'ollama'
        ? 'No local server URL configured. Set the server URL in Settings.'
        : `No ${provider} API key configured. Add your API key in Settings.`;

    super({
      code: RagErrorCode.NO_PROVIDER_CONFIGURED,
      message,
      cause,
      retryable: false,
      provider,
      context: { provider },
    });
  }
}

export class ModelCatalogEmptyError extends RagError {
  constructor(provider: string, cause?: Error) {
    const message = `${provider} model catalog is empty. Check your API key and click Refresh in Settings.`;

    super({
      code: RagErrorCode.MODEL_CATALOG_EMPTY,
      message,
      cause,
      retryable: true,
      provider,
      context: { provider },
    });
  }
}

export class ModelNotFoundError extends RagError {
  constructor(modelId: string, availableModels: string[], cause?: Error) {
    const suggestions = availableModels.slice(0, 5).join(', ');
    const hint = availableModels.length > 0
      ? ` Available models: ${suggestions}${availableModels.length > 5 ? '...' : ''}`
      : '';

    super({
      code: RagErrorCode.MODEL_NOT_FOUND,
      message: `Model "${modelId}" not found in the catalog.${hint} Re-pick or refresh the catalog.`,
      cause,
      retryable: false,
      context: { modelId, availableCount: availableModels.length },
    });
  }
}

export class LocalServerUrlMissingError extends RagError {
  constructor(cause?: Error) {
    super({
      code: RagErrorCode.LOCAL_SERVER_URL_MISSING,
      message: 'Local server URL is required. Set the OpenAI-compatible server URL in Settings (e.g., http://localhost:11434/v1).',
      cause,
      retryable: false,
      provider: 'local',
      context: {},
    });
  }
}