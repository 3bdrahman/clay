/**
 * Settings validation utilities for the Clay RAG pipeline.
 * Validates configuration at startup and before workflow execution.
 */

import type { Settings } from './types';
import { getProviderConfig, resolveProviderEndpoint } from './providers';
import { inspectLocalServerUrl } from './localEndpoint';
import {
  NoProviderError,
  ModelCatalogEmptyError,
  ModelNotFoundError,
  LocalServerUrlMissingError,
  ProviderUnreachableError,
  RagErrorCode,
  type RagError,
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
  let firstError: RagError | undefined;
  const addError = (error: RagError, providerKind?: 'local') => {
    firstError ??= error;
    errors.push({ code: error.code, message: error.message, ...(providerKind ? { providerKind } : {}) });
  };

  // 1. Provider configuration
  if (settings.provider === 'local') {
    if (!settings.localServerUrl || !settings.localServerUrl.trim()) {
      const err = new LocalServerUrlMissingError();
      addError(err, 'local');
    } else {
      const endpoint = inspectLocalServerUrl(settings.localServerUrl);
      if (endpoint.error) addError(new ProviderUnreachableError('Local server', undefined, { message: endpoint.error, retryable: false }), 'local');
    }

    // Local models must be picked
    if (!settings.localModels?.chat || !settings.localModels.chat.trim()) {
      const err = new ModelNotFoundError('chat', settings.localCatalog?.map((m) => m.id) ?? []);
      addError(err, 'local');
    }
  } else {
    const endpoint = resolveProviderEndpoint(settings);
    if (!endpoint.apiKey) {
      const err = new NoProviderError(settings.provider);
      addError(err);
    }
  }

  // 2. Model catalog validation
  if (settings.provider === 'local') {
    if (!settings.localCatalog || settings.localCatalog.length === 0) {
      const err = new ModelCatalogEmptyError('local');
      warnings.push(err.message);
    } else {
      // Validate picked chat model exists in catalog
      const catalogIds = new Set(settings.localCatalog.map((m) => m.id));
      if (settings.localModels?.chat && !catalogIds.has(settings.localModels.chat)) {
        const err = new ModelNotFoundError(settings.localModels.chat, Array.from(catalogIds));
        addError(err, 'local');
      }
    }
  }

  // 3. Web search configuration
  if (settings.webSearchProvider === 'serper' && (!settings.serperApiKey || !settings.serperApiKey.trim())) {
    warnings.push('Add a Serper API key to enable web search.');
  }

  const valid = errors.length === 0;

  if (options.throwOnError && firstError) {
    throw firstError;
  }

  return { valid, errors, warnings };
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
  modelCount: number;
  issues: string[];
} {
  const result = validateSettings(settings);
  const issues = [...result.errors.map((e) => e.message), ...result.warnings];

  return {
    configured: result.valid,
    provider: getProviderConfig(settings.provider).displayName,
    hasApiKey: settings.provider === 'local' ? !!settings.localServerUrl.trim() : !!resolveProviderEndpoint(settings).apiKey,
    modelCount: settings.provider === 'local' ? settings.localCatalog?.length ?? 0 : 0,
    issues,
  };
}
