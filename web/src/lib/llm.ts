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

  function createAbortControllerWithTimeout(): {
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

  /**
   * Classifies HTTP response errors into typed RagErrors.
   */
  async function handleResponseError(resp: Response, _step: string, modelId?: string): Promise<never> {
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

  async function sendChatRequest(req: LLMRequest, signal?: AbortSignal): Promise<LLMResponse> {
    const messages: Array<Record<string, unknown>> = [];
    if (req.system) messages.push({ role: 'system', content: req.system });
    for (const m of req.messages) {
      if (m.role === 'assistant' && m.toolCalls) {
        messages.push({
          role: 'assistant',
          content: m.content,
          tool_calls: m.toolCalls.map(c => ({
            id: c.id,
            type: 'function',
            function: { name: c.function.name, arguments: c.function.arguments },
          })),
        });
      } else if (m.role === 'tool') {
        messages.push({
          role: 'tool',
          content: m.content,
          tool_call_id: m.toolCallId,
        });
      } else {
        messages.push({ role: m.role, content: m.content });
      }
    }

    const body: Record<string, unknown> = {
      model: req.model,
      messages,
      temperature: req.temperature ?? defaultTemperature,
    };
    if (req.maxTokens) body.max_tokens = req.maxTokens;
    if (req.jsonMode && jsonModeSupported) body.response_format = { type: 'json_object' };
    if (req.tools) body.tools = req.tools;
    if (req.toolChoice) body.tool_choice = req.toolChoice;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    // Combine the external signal with the timeout signal (same pattern as
    // streamOpenAICompatible). Listeners are removed in the finally below —
    // a tool loop makes many calls per question against one external signal,
    // and an un-removed listener would pin every call's closure to it.
    const combinedController = new AbortController();
    const onExternalAbort = () => combinedController.abort();
    const onTimeoutAbort = () => combinedController.abort();
    const { controller, cleanup, didTimeout } = createAbortControllerWithTimeout();
    controller.signal.addEventListener('abort', onTimeoutAbort);
    if (signal) {
      signal.addEventListener('abort', onExternalAbort);
    }

    // If external signal is already aborted, throw immediately (matches stream behavior)
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
        throw classifyError(e, providerLabel, 'invoke');
      }

      if (!resp.ok) {
        await handleResponseError(resp, 'invoke', req.model);
      }

      // The timeout stays active through the body phase: a server that
      // accepts the headers but stalls the body would otherwise hang until
      // the user presses Esc.
      let data: unknown;
      try {
        data = await resp.json();
      } catch (e) {
        if (didTimeout()) {
          throw new ProviderTimeoutError(providerLabel, timeoutMs, e instanceof Error ? e : new Error(String(e)));
        }
        if (signal?.aborted || e instanceof DOMException) {
          throw new StreamInterruptedError(providerLabel, '', e instanceof Error ? e : new Error(String(e)));
        }
        if (import.meta.env.DEV) {
          console.warn('[llm] invoke(): response JSON parse failed:', e);
        }
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
      const finishReason = (choice.finish_reason as string | undefined) ?? (message?.finish_reason as string | undefined);

      const toolCalls = message?.tool_calls as Array<Record<string, unknown>> | undefined;
      const parsedToolCalls = toolCalls?.map(c => ({
        id: c.id as string,
        type: 'function' as const,
        function: {
          name: (c.function as Record<string, unknown>)?.name as string,
          arguments: (c.function as Record<string, unknown>)?.arguments as string,
        },
      }));

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
        toolCalls: parsedToolCalls,
        finishReason,
      };
    } finally {
      cleanup();
      controller.signal.removeEventListener('abort', onTimeoutAbort);
      if (signal) {
        signal.removeEventListener('abort', onExternalAbort);
      }
    }
  }

  let jsonModeSupported = true;

  async function callOpenAICompatible(req: LLMRequest, signal?: AbortSignal): Promise<LLMResponse> {
    try {
      return await sendChatRequest(req, signal);
    } catch (e) {
      // Local servers that reject the response_format param fail every
      // jsonMode call with a 400. Retry once without it and remember the
      // capability so subsequent calls skip the param.
      const isBadRequest = e instanceof GenerationFailedError &&
        e.cause instanceof Error && e.cause.message.startsWith('400');
      if (req.jsonMode && jsonModeSupported && isBadRequest) {
        jsonModeSupported = false;
        return sendChatRequest({ ...req, jsonMode: undefined }, signal);
      }
      throw e;
    }
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

    // Each abort listener added here is detached in the finally below: a
    // tool loop makes many calls per question against one external signal,
    // and a listener left behind would pin every call's closure to it.
    const combinedController = new AbortController();
    const onExternalAbort = () => combinedController.abort();
    const onTimeoutAbort = () => combinedController.abort();
    const { controller, cleanup, didTimeout } = createAbortControllerWithTimeout();
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
        await handleResponseError(resp, 'stream', req.model);
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

  return {
    async invoke(req: LLMRequest, signal?: AbortSignal): Promise<LLMResponse> {
      return callOpenAICompatible(req, signal);
    },
    async stream(req: LLMRequest, onToken: (token: string) => void, signal?: AbortSignal): Promise<LLMResponse> {
      return streamOpenAICompatible(req, onToken, signal);
    },
  };
}

export interface LLMClient {
  invoke(req: LLMRequest, signal?: AbortSignal): Promise<LLMResponse>;
  stream(req: LLMRequest, onToken: (token: string) => void, signal?: AbortSignal): Promise<LLMResponse>;
}