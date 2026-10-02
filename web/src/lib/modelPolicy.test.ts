import { describe, expect, it } from 'vitest';
import {
  getCloudModelLabel,
  getCloudModelOptions,
  isApprovedCloudModel,
  OPENROUTER_FREE_ROUTING,
} from './modelPolicy';
import type { ModelInfo } from './types';

const openRouterNemotron: ModelInfo = {
  id: 'nvidia/nemotron-3-super-120b-a12b:free',
  ownedBy: 'nvidia',
  created: 0,
  pricing: { prompt: '0', completion: '0', request: '0' },
  supportedParameters: ['tools', 'tool_choice', 'response_format', 'structured_outputs'],
};

const openRouterQwen: ModelInfo = {
  id: 'qwen/qwen3.8-27b:free',
  ownedBy: 'qwen',
  created: 0,
  pricing: { prompt: '0', completion: '0', request: '0' },
  supportedParameters: ['tools', 'tool_choice', 'structured_outputs'],
};

describe('modelPolicy', () => {
  it('externalizes exact approved cloud model ids and labels', () => {
    expect(isApprovedCloudModel('openrouter', 'nvidia/nemotron-3-super-120b-a12b:free')).toBe(true);
    expect(isApprovedCloudModel('openrouter', 'nvidia/nemotron-3-super-120b-a12b')).toBe(false);
    expect(isApprovedCloudModel('nim', 'nvidia/nemotron-3.5-lightning-30b-a3b')).toBe(true);
    expect(getCloudModelLabel('nim', 'nvidia/nemotron-3-super-120b-a12b')).toBe(
      'Nemotron 3 Super 120B',
    );
  });

  it('returns only eligible OpenRouter free models in configured order', () => {
    const options = getCloudModelOptions('openrouter', [
      openRouterQwen,
      { id: 'some/paid-model', ownedBy: 'vendor', created: 0 },
      openRouterNemotron,
    ]);
    expect(options.map((model) => model.id)).toEqual([
      'nvidia/nemotron-3-super-120b-a12b:free',
      'qwen/qwen3.8-27b:free',
    ]);
  });

  it('rejects OpenRouter models with nonzero or incomplete pricing metadata', () => {
    expect(getCloudModelOptions('openrouter', [
      { ...openRouterNemotron, pricing: { prompt: '0', completion: '0.000001', request: '0' } },
      { ...openRouterQwen, pricing: { prompt: '0' } },
    ])).toEqual([]);
  });

  it('accepts the live catalog shape with optional request pricing omitted', () => {
    const options = getCloudModelOptions('openrouter', [
      { ...openRouterNemotron, pricing: { prompt: '0', completion: '0' } },
      { ...openRouterQwen, pricing: { prompt: '0', completion: '0' } },
    ]);
    expect(options).toHaveLength(2);
    expect(OPENROUTER_FREE_ROUTING.max_price.request).toBe(0);
  });

  it('rejects a listed nonzero per-request charge', () => {
    expect(getCloudModelOptions('openrouter', [{
      ...openRouterNemotron, pricing: { prompt: '0', completion: '0', request: '0.01' },
    }])).toEqual([]);
  });

  it('rejects OpenRouter models without tool and structured-output capabilities', () => {
    expect(getCloudModelOptions('openrouter', [
      { ...openRouterNemotron, supportedParameters: ['tools', 'response_format'] },
      { ...openRouterQwen, supportedParameters: ['tools', 'tool_choice'] },
    ])).toEqual([]);
  });

  it('keeps NIM to the approved model ids that are present in the catalog', () => {
    const options = getCloudModelOptions('nim', [
      { id: 'nvidia/nemotron-3.5-lightning-30b-a3b', ownedBy: 'nvidia', created: 0 },
      { id: 'nvidia/not-approved', ownedBy: 'nvidia', created: 0 },
      { id: 'nvidia/nemotron-3-super-120b-a12b', ownedBy: 'nvidia', created: 0 },
    ]);
    expect(options.map((model) => model.id)).toEqual([
      'nvidia/nemotron-3-super-120b-a12b',
      'nvidia/nemotron-3.5-lightning-30b-a3b',
    ]);
  });

  it('exports the OpenRouter free routing guard used by request code', () => {
    expect(OPENROUTER_FREE_ROUTING).toEqual({
      max_price: { prompt: 0, completion: 0, request: 0 },
    });
  });
});
