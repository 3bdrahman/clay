import providerModelsConfig from './providerModels.config.json';
import type { ModelInfo } from './types';

export type CloudProviderKind = 'openrouter' | 'nim';

interface ApprovedModelConfig {
  id: string;
  label: string;
}

const CLOUD_MODEL_CONFIG = providerModelsConfig as Record<CloudProviderKind, ApprovedModelConfig[]>;

export const OPENROUTER_FREE_ROUTING = {
  max_price: {
    prompt: 0,
    completion: 0,
    request: 0,
  },
} as const;

export function isApprovedCloudModel(provider: CloudProviderKind, id: string): boolean {
  return CLOUD_MODEL_CONFIG[provider].some((entry) => entry.id === id);
}

export function getApprovedCloudModelIds(provider: CloudProviderKind): string[] {
  return CLOUD_MODEL_CONFIG[provider].map(entry => entry.id);
}

export function getCloudModelLabel(provider: CloudProviderKind, id: string): string | undefined {
  return CLOUD_MODEL_CONFIG[provider].find((entry) => entry.id === id)?.label;
}

export function getCloudModelOptions(provider: CloudProviderKind, catalog: ModelInfo[]): ModelInfo[] {
  const byId = new Map(catalog.map((model) => [model.id, model]));
  return CLOUD_MODEL_CONFIG[provider].flatMap((entry) => {
    const model = byId.get(entry.id);
    if (!model) return [];
    if (provider === 'openrouter' && !isEligibleOpenRouterFreeModel(model)) return [];
    return [model];
  });
}

function isEligibleOpenRouterFreeModel(model: ModelInfo): boolean {
  if (!model.id.endsWith(':free')) return false;
  if (!isApprovedCloudModel('openrouter', model.id)) return false;
  if (!hasOpenRouterZeroPricing(model.pricing)) return false;
  return hasRequiredOpenRouterCapabilities(model.supportedParameters);
}

function hasOpenRouterZeroPricing(pricing: ModelInfo['pricing']): boolean {
  if (!pricing) return false;
  if (!isZeroPrice(pricing.prompt) || !isZeroPrice(pricing.completion)) {
    return false;
  }
  // Optional fees can be absent from the public catalog. Any fee it lists
  // must be zero; request-level price caps also enforce zero per-request cost.
  return Object.values(pricing).every((value) => isZeroPrice(value));
}

function isZeroPrice(value: string | undefined): boolean {
  if (value === undefined) return false;
  const normalized = value.trim();
  if (!normalized) return false;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) && parsed === 0;
}

function hasRequiredOpenRouterCapabilities(supportedParameters: string[] | undefined): boolean {
  if (!supportedParameters) return false;
  const parameters = new Set(supportedParameters);
  return parameters.has('tools')
    && parameters.has('tool_choice')
    && (parameters.has('response_format') || parameters.has('structured_outputs'));
}
