import type { LLMRequest, LLMResponse } from './types';
import {
  ProviderUnreachableError,
  InvalidApiKeyError,
  RateLimitError,
  StreamInterruptedError,
  TokenBudgetExceededError,
  GenerationFailedError,
  ModelNotFoundError,
  classifyError,
} from './errors';

export { ProviderUnreachableError } from './errors';

export interface LLMClientConfig {
  baseUrl: string;
  apiKey: string;
  temperature?: number;
  providerLabel?: string;
  timeoutMs?: number;
}

/**
 * Create an OpenAI-compatible LLM client.
 * Supports both invoke (non-streaming) and stream (token-by-token) modes.
 * @param config - Client configuration: baseUrl, apiKey, optional temperature, providerLabel
 * @returns LLMClient with invoke() and stream() methods
 * @throws Error if baseUrl is empty
 */
export function createLLMClient(config: LLMClientConfig): LLMClient {
  const baseUrl = config.baseUrl.replace(/\/+$/, '');
  const apiKey = config.apiKey;
  const providerLabel = config.providerLabel ?? 'provider';
  const defaultTemperature = config.temperature ?? 0;
  const timeoutMs = config.timeoutMs ?? 120000; // Default 2 minutes

  if (!baseUrl) {
    throw new ProviderUnreachableError(providerLabel, undefined, {
      isTimeout: false,
    });
  }

  function createAbortControllerWithTimeout(): { controller: AbortController; cleanup: () => void } {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    const cleanup = () => clearTimeout(timeoutId);
    return { controller, cleanup };
  }

  /**
   * Classifies HTTP response errors into typed RagErrors.
   */
  async function handleResponseError(resp: Response, _step: string): Promise<never> {
    const status = resp.status;
    const text = await resp.text().catch(() => '');

    if (status === 401 || status === 403) {
      throw new InvalidApiKeyError(providerLabel, status as 401 | 403);
    }

    if (status === 429) {
      const retryAfter = resp.headers.get('retry-after');
      const retryAfterMs = retryAfter ? parseInt(retryAfter, 10) * 1000 : undefined;
      throw new RateLimitError(providerLabel, retryAfterMs);
    }

    if (status === 404) {
      throw new ModelNotFoundError('unknown', [], new Error(`${status} ${resp.statusText}`));
    }

    if (status >= 500) {
      throw new ProviderUnreachableError(providerLabel, new Error(`${status} ${resp.statusText}: ${text}`), {
        retryable: true,
      });
    }

    // 400, 408, etc.
    throw new GenerationFailedError(providerLabel, new Error(`${status} ${resp.statusText}: ${text}`));
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

    const { controller, cleanup } = createAbortControllerWithTimeout();
    let resp: Response;
    try {
      resp = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (e) {
      cleanup();
      // Network error (DNS, connection refused, CORS, etc.)
      throw classifyError(e, providerLabel, 'invoke');
    }
    cleanup();

    if (!resp.ok) {
      await handleResponseError(resp, 'invoke');
    }

    let data: unknown;
    try {
      data = await resp.json();
    } catch {
      throw new GenerationFailedError(providerLabel, new Error('Invalid JSON response'));
    }

    const d = data as Record<string, unknown>;
    const choices = d.choices as Array<Record<string, unknown>> | undefined;
    if (!choices || choices.length === 0) {
      throw new GenerationFailedError(providerLabel, new Error('No choices in response'));
    }

    const choice = choices[0];
    const message = choice.message as Record<string, unknown> | undefined;
    const content = (message?.content as string) ?? '';

    // Check for token budget exceeded in response
    if (content.includes('token') && content.includes('budget') && content.includes('exceed')) {
      throw new TokenBudgetExceededError(0, 0, new Error(content));
    }

    return {
      content,
      usage: d.usage
        ? {
            promptTokens: (d.usage as Record<string, unknown>).prompt_tokens as number,
            completionTokens: (d.usage as Record<string, unknown>).completion_tokens as number,
            totalTokens: (d.usage as Record<string, unknown>).total_tokens as number,
          }
        : undefined,
      model: d.model as string | undefined,
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

    // Combine external signal with timeout signal
    const { controller, cleanup } = createAbortControllerWithTimeout();
    const combinedSignal = signal
      ? (() => {
          const combined = new AbortController();
          signal.addEventListener('abort', () => combined.abort());
          controller.signal.addEventListener('abort', () => combined.abort());
          return combined.signal;
        })()
      : controller.signal;

    let resp: Response;
    try {
      resp = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: combinedSignal,
      });
    } catch (e) {
      cleanup();
      // Network error or abort
      if (signal?.aborted || e instanceof DOMException) {
        throw new StreamInterruptedError(providerLabel, '', e instanceof Error ? e : new Error(String(e)));
      }
      throw classifyError(e, providerLabel, 'stream');
    } finally {
      cleanup();
    }

    if (!resp.ok) {
      await handleResponseError(resp, 'stream');
    }

    const reader = resp.body?.getReader();
    if (!reader) throw new GenerationFailedError(providerLabel, new Error('No response body'));

    const decoder = new TextDecoder();
    let fullContent = '';
    let usage: LLMResponse['usage'] = undefined;
    let model: string | undefined;

    try {
      while (true) {
        let readResult;
        try {
          readResult = await reader.read();
        } catch (e) {
          if (signal?.aborted) {
            throw new StreamInterruptedError(providerLabel, fullContent, e instanceof Error ? e : new Error(String(e)));
          }
          throw classifyError(e, providerLabel, 'stream-read');
        }

        const { done, value } = readResult;
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

    if (signal?.aborted) {
      throw new StreamInterruptedError(providerLabel, fullContent, new Error('Aborted'));
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

export interface LLMClient {
  invoke(req: LLMRequest): Promise<LLMResponse>;
  stream(req: LLMRequest, onToken: (token: string) => void, signal?: AbortSignal): Promise<LLMResponse>;
}