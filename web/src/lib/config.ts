/**
 * Settings validation utilities for the Clay RAG pipeline.
 * Validates configuration at startup and before workflow execution.
 */

import type { Settings } from './types';
import {
  NoProviderError,
  EmbeddingModelMissingError,
  ModelCatalogEmptyError,
  ModelNotFoundError,
  LocalServerUrlMissingError,
  RagErrorCode,
} from './errors';

export interface ValidationResult {
  valid: boolean;
  errors: Array<{ code: RagErrorCode; message: string; providerKind?: 'local' }>;
  warnings: string[];
}

/**
 * Validates the complete settings object.
 * Returns validation result with errors and warnings.
 * Throws on first blocking error if throwOnError is true.
 */
export function validateSettings(
  settings: Settings,
  options: { throwOnError?: boolean } = {}
): ValidationResult {
  const errors: Array<{ code: RagErrorCode; message: string; providerKind?: 'local' }> = [];
  const warnings: string[] = [];

  // 1. Provider configuration
  if (settings.provider === 'local') {
    if (!settings.localServerUrl || !settings.localServerUrl.trim()) {
      const err = new LocalServerUrlMissingError();
      errors.push({ code: err.code, message: err.message, providerKind: 'local' });
    }

    // Local models must be picked
    if (!settings.localModels?.chat || !settings.localModels.chat.trim()) {
      const err = new ModelNotFoundError('chat', settings.localCatalog?.map((m) => m.id) ?? []);
      errors.push({ code: err.code, message: err.message, providerKind: 'local' });
    }

    if (!settings.localModels?.embeddings || !settings.localModels.embeddings.trim()) {
      const err = new EmbeddingModelMissingError();
      errors.push({ code: err.code, message: err.message, providerKind: 'local' });
    }
  } else {
    // API providers (openrouter, groq, together)
    const apiKeyField = {
      openrouter: 'openrouterApiKey',
      groq: 'groqApiKey',
      together: 'togetherApiKey',
    }[settings.provider];

    const apiKey = (settings as unknown as Record<string, string>)[apiKeyField];
    if (!apiKey || !apiKey.trim()) {
      const err = new NoProviderError(settings.provider);
      errors.push({ code: err.code, message: err.message });
    }

    if (!settings.embeddingApiKey || !settings.embeddingApiKey.trim()) {
      const err = new EmbeddingModelMissingError();
      errors.push({ code: err.code, message: err.message });
    }
  }

  // 2. Model catalog validation
  if (settings.provider === 'local') {
    if (!settings.localCatalog || settings.localCatalog.length === 0) {
      const err = new ModelCatalogEmptyError('local');
      warnings.push(err.message);
    } else {
      // Validate picked models exist in catalog
      const catalogIds = new Set(settings.localCatalog.map((m) => m.id));
      if (settings.localModels?.chat && !catalogIds.has(settings.localModels.chat)) {
        const err = new ModelNotFoundError(settings.localModels.chat, Array.from(catalogIds));
        errors.push({ code: err.code, message: err.message, providerKind: 'local' });
      }
      if (settings.localModels?.embeddings && !catalogIds.has(settings.localModels.embeddings)) {
        const err = new ModelNotFoundError(settings.localModels.embeddings, Array.from(catalogIds));
        errors.push({ code: err.code, message: err.message, providerKind: 'local' });
      }
    }
  }

  // 3. Web search configuration
  if (settings.webSearchProvider === 'serper' && (!settings.serperApiKey || !settings.serperApiKey.trim())) {
    warnings.push('Serper API key not configured. Web search will fall back to DuckDuckGo.');
  }

  const valid = errors.length === 0;

  if (options.throwOnError && !valid) {
    // Throw the first error
    const firstError = errors[0];
    throw createErrorFromCode(firstError.code, firstError.message);
  }

  return { valid, errors, warnings };
}

/**
 * Creates a RagError instance from code and message.
 * Used for throwing from validation.
 */
function createErrorFromCode(
  code: RagErrorCode,
  message: string
): Error {
  switch (code) {
    case RagErrorCode.NO_PROVIDER_CONFIGURED:
      return new NoProviderError('local', undefined);
    case RagErrorCode.EMBEDDING_MODEL_MISSING:
      return new EmbeddingModelMissingError();
    case RagErrorCode.MODEL_CATALOG_EMPTY:
      return new ModelCatalogEmptyError('local', undefined);
    case RagErrorCode.MODEL_NOT_FOUND:
      return new ModelNotFoundError('', []);
    case RagErrorCode.LOCAL_SERVER_URL_MISSING:
      return new LocalServerUrlMissingError();
    default:
      const err = new Error(message);
      err.name = 'RagError';
      return err;
  }
}

/**
 * Validates settings specifically for workflow execution.
 * More strict than general validation - throws on any blocking issue.
 */
export function validateSettingsForWorkflow(settings: Settings): void {
  validateSettings(settings, { throwOnError: true });
}

/**
 * Gets a user-friendly summary of settings status.
 */
export function getSettingsStatus(settings: Settings): {
  configured: boolean;
  provider: string;
  hasApiKey: boolean;
  hasEmbeddingKey: boolean;
  modelCount: number;
  issues: string[];
} {
  const result = validateSettings(settings);
  const issues = [...result.errors.map((e) => e.message), ...result.warnings];

  const providerNames: Record<string, string> = {
    openrouter: 'OpenRouter',
    groq: 'Groq',
    together: 'Together AI',
    local: 'Local Server',
  };

  const apiKeyField = {
    openrouter: 'openrouterApiKey',
    groq: 'groqApiKey',
    together: 'togetherApiKey',
    local: '',
  }[settings.provider];

  return {
    configured: result.valid,
    provider: providerNames[settings.provider] ?? settings.provider,
    hasApiKey: settings.provider === 'local' ? !!settings.localServerUrl : !!(settings as unknown as Record<string, string>)[apiKeyField],
    hasEmbeddingKey: settings.provider === 'local' ? !!settings.localModels?.embeddings : !!settings.embeddingApiKey,
    modelCount: settings.provider === 'local' ? settings.localCatalog?.length ?? 0 : 0,
    issues,
  };
}