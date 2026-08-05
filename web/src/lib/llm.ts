import type { LLMRequest, LLMResponse } from './types';

export interface LLMClient {
  invoke(req: LLMRequest): Promise<LLMResponse>;
  stream(req: LLMRequest, onToken: (token: string) => void, signal?: AbortSignal): Promise<LLMResponse>;
}

export class LLMConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LLMConfigError';
  }
}

export interface LLMClientConfig {
  baseUrl: string;
  apiKey: string;
  temperature?: number;
  providerLabel?: string;
}

export function createLLMClient(config: LLMClientConfig): LLMClient {
  const baseUrl = config.baseUrl.replace(/\/+$/, '');
  const apiKey = config.apiKey;
  const providerLabel = config.providerLabel ?? 'provider';
  const defaultTemperature = config.temperature ?? 0;

  if (!baseUrl) {
    throw new LLMConfigError(`${providerLabel} base URL is empty. Open Settings and configure it.`);
  }

  async function callOpenAICompatible(req: LLMRequest): Promise<LLMResponse> {
    const messages: Array<{ role: string; content: string }> = [];
    if (req.system) messages.push({ role: 'system', content: req.system });
    for (const m of req.messages) messages.push({ role: m.role, content: m.content });

    const body: Record<string, unknown> = {
      model: req.model,
      messages,
      temperature: req.temperature ?? defaultTemperature,
    };
    if (req.maxTokens) body.max_tokens = req.maxTokens;
    if (req.jsonMode) body.response_format = { type: 'json_object' };

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    const resp = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}: ${await resp.text()}`);

    const data = await resp.json();
    const choice = data.choices?.[0];
    if (!choice) throw new Error('No choices in response');

    return {
      content: choice.message?.content ?? '',
      usage: data.usage
        ? {
            promptTokens: data.usage.prompt_tokens,
            completionTokens: data.usage.completion_tokens,
            totalTokens: data.usage.total_tokens,
          }
        : undefined,
      model: data.model,
    };
  }

  async function streamOpenAICompatible(
    req: LLMRequest,
    onToken: (token: string) => void,
    signal?: AbortSignal,
  ): Promise<LLMResponse> {
    const messages: Array<{ role: string; content: string }> = [];
    if (req.system) messages.push({ role: 'system', content: req.system });
    for (const m of req.messages) messages.push({ role: m.role, content: m.content });

    const body: Record<string, unknown> = {
      model: req.model,
      messages,
      temperature: req.temperature ?? defaultTemperature,
      stream: true,
    };
    if (req.maxTokens) body.max_tokens = req.maxTokens;
    if (req.jsonMode) body.response_format = { type: 'json_object' };

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    };
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    const resp = await fetch(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal,
    });
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}: ${await resp.text()}`);

    const reader = resp.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let fullContent = '';
    let usage: LLMResponse['usage'] = undefined;
    let model: string | undefined;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            const choice = parsed.choices?.[0];
            if (!choice) continue;

            if (choice.delta?.content) {
              const token = choice.delta.content;
              fullContent += token;
              onToken(token);
            }

            if (choice.finish_reason) {
              usage = parsed.usage
                ? {
                    promptTokens: parsed.usage.prompt_tokens,
                    completionTokens: parsed.usage.completion_tokens,
                    totalTokens: parsed.usage.total_tokens,
                  }
                : undefined;
              model = parsed.model;
            }
          } catch {
            // Ignore parse errors for partial chunks
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    return { content: fullContent, usage, model };
  }

  return {
    async invoke(req: LLMRequest): Promise<LLMResponse> {
      return callOpenAICompatible(req);
    },
    async stream(req: LLMRequest, onToken: (token: string) => void, signal?: AbortSignal): Promise<LLMResponse> {
      return streamOpenAICompatible(req, onToken, signal);
    },
  };
}
