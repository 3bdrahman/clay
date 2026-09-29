import type { LLMRequest, LLMResponse } from './types';
import {
  ProviderUnreachableError,
  InvalidApiKeyError,
  RateLimitError,
  ProviderTimeoutError,
  StreamInterruptedError,
  GenerationFailedError,
  ModelNotFoundError,
  classifyError,
} from './errors';
import { buildMessages } from './llmMessages';

export interface StreamConfig {
  baseUrl: string;
  apiKey: string;
  providerLabel: string;
  defaultTemperature: number;
  timeoutMs: number;
}

function createAbortControllerWithTimeout(timeoutMs: number): {
  controller: AbortController;
  cleanup: () => void;
  didTimeout: () => boolean;
} {
  const controller = new AbortController();
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const cleanup = () => clearTimeout(timeoutId);
  return { controller, cleanup, didTimeout: () => timedOut };
}

async function handleResponseError(
  resp: Response,
  _step: string,
  providerLabel: string,
  modelId?: string,
): Promise<never> {
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
    throw new ModelNotFoundError(modelId ?? '(unspecified)', [], new Error(`${status} ${resp.statusText}`));
  }

  if (status >= 500) {
    throw new ProviderUnreachableError(providerLabel, new Error(`${status} ${resp.statusText}: ${text}`), {
      retryable: true,
    });
  }

  // 400, 408, etc.
  throw new GenerationFailedError(providerLabel, new Error(`${status} ${resp.statusText}: ${text}`));
}

/**
 * Stream an OpenAI-compatible chat completion with SSE parsing and abort handling.
 * @param config - Stream configuration (baseUrl, apiKey, providerLabel, defaultTemperature, timeoutMs)
 * @param req - The LLM request
 * @param onToken - Callback for each token received
 * @param signal - Optional external abort signal
 * @returns The complete LLM response
 */
export async function streamOpenAICompatible(
  config: StreamConfig,
  req: LLMRequest,
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<LLMResponse> {
  const { baseUrl, apiKey, providerLabel, defaultTemperature, timeoutMs } = config;

  const messages = buildMessages(req);

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

  // Each abort listener added here is detached in the finally below: a
  // tool loop makes many calls per question against one external signal,
  // and a listener left behind would pin every call's closure to it.
  const combinedController = new AbortController();
  const onExternalAbort = () => combinedController.abort();
  const onTimeoutAbort = () => combinedController.abort();
  const { controller, cleanup, didTimeout } = createAbortControllerWithTimeout(timeoutMs);
  controller.signal.addEventListener('abort', onTimeoutAbort);
  if (signal) {
    signal.addEventListener('abort', onExternalAbort);
  }

  // If external signal is already aborted, throw immediately — a live
  // combined signal does not fire listeners for an abort that already
  // happened before they were added.
  if (signal?.aborted) {
    cleanup();
    controller.signal.removeEventListener('abort', onTimeoutAbort);
    if (signal) {
      signal.removeEventListener('abort', onExternalAbort);
    }
    throw new StreamInterruptedError(providerLabel, '', new Error('Aborted'));
  }

  try {
    let resp: Response;
    try {
      resp = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: combinedController.signal,
      });
    } catch (e) {
      // Network error, user abort, or timeout
      if (signal?.aborted) {
        throw new StreamInterruptedError(providerLabel, '', e instanceof Error ? e : new Error(String(e)));
      }
      if (didTimeout()) {
        throw new ProviderTimeoutError(providerLabel, timeoutMs, e instanceof Error ? e : new Error(String(e)));
      }
      if (e instanceof DOMException) {
        throw new StreamInterruptedError(providerLabel, '', e instanceof Error ? e : new Error(String(e)));
      }
      throw classifyError(e, providerLabel, 'stream');
    }

    if (!resp.ok) {
      await handleResponseError(resp, 'stream', providerLabel, req.model);
    }

    const reader = resp.body?.getReader();
    if (!reader) throw new GenerationFailedError(providerLabel, new Error('No response body'));

    const decoder = new TextDecoder();
    let fullContent = '';
    let usage: LLMResponse['usage'] = undefined;
    let model: string | undefined;
    // A `data:` JSON line can split across network chunks; without carrying
    // the partial tail forward, the remainder (which no longer carries the
    // `data: ` prefix) is silently skipped and the token is lost.
    let sseBuffer = '';

    try {
      while (true) {
        let readResult;
        try {
          readResult = await reader.read();
        } catch (e) {
          if (signal?.aborted) {
            throw new StreamInterruptedError(providerLabel, fullContent, e instanceof Error ? e : new Error(String(e)));
          }
          if (didTimeout()) {
            // The timeout stays active through the read phase: a server
            // that stalls mid-stream surfaces as a typed timeout carrying
            // the budget in its message, instead of hanging silently
            // until the user presses Esc.
            throw new ProviderTimeoutError(providerLabel, timeoutMs, e instanceof Error ? e : new Error(String(e)));
          }
          throw classifyError(e, providerLabel, 'stream-read');
        }

        const { done, value } = readResult;
        if (done) break;

        sseBuffer += decoder.decode(value, { stream: true });
        const lines = sseBuffer.split('\n');
        sseBuffer = lines.pop() ?? '';

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
          } catch (e) {
            // A partial line is never handed to JSON.parse (the buffer keeps
            // it), so a parse failure here is a genuinely malformed chunk —
            // skipping is correct. DEV-only log so loud providers still surface.
            if (import.meta.env.DEV) {
              console.warn('[llm] stream(): partial SSE chunk parse skipped:', e);
            }
          }
        }
      }
      if (sseBuffer.startsWith('data: ') && !sseBuffer.includes('[DONE]')) {
        try {
          const parsed = JSON.parse(sseBuffer.slice(6).trim());
          const token = parsed.choices?.[0]?.delta?.content;
          if (token) {
            fullContent += token;
            onToken(token);
          }
        } catch (e) {
          if (import.meta.env.DEV) {
            console.warn('[llm] stream(): final SSE line parse skipped:', e);
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
  } finally {
    cleanup();
    controller.signal.removeEventListener('abort', onTimeoutAbort);
    if (signal) {
      signal.removeEventListener('abort', onExternalAbort);
    }
  }
}