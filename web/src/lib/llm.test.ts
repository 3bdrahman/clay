import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest';
import { createLLMClient } from './llm';
import {
  InvalidApiKeyError,
  RateLimitError,
  ProviderUnreachableError,
  StreamInterruptedError,
  GenerationFailedError,
  ModelNotFoundError,
} from './errors';

describe('createLLMClient', () => {
  const mockFetch = vi.fn();
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
    globalThis.fetch = mockFetch;
  });

  afterAll(() => {
    globalThis.fetch = originalFetch;
  });

  it('throws ProviderUnreachableError when baseUrl is empty', () => {
    expect(() => createLLMClient({ baseUrl: '', apiKey: 'k', providerLabel: 'test' })).toThrow(ProviderUnreachableError);
  });

  it('does NOT throw when apiKey is empty (local servers do not require auth)', () => {
    expect(() => createLLMClient({ baseUrl: 'http://localhost:11434/v1', apiKey: '' })).not.toThrow();
  });

  it('targets the configured baseUrl (not NIM)', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: 'ok' } }] }),
    });

    const client = createLLMClient({ baseUrl: 'http://localhost:11434/v1', apiKey: '' });
    await client.invoke({ messages: [{ role: 'user', content: 'hi' }] });

    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:11434/v1/chat/completions',
      expect.objectContaining({ method: 'POST' }),
    );
  });

  it('omits Authorization header when apiKey is empty', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: 'ok' } }] }),
    });

    const client = createLLMClient({ baseUrl: 'http://localhost:11434/v1', apiKey: '' });
    await client.invoke({ messages: [{ role: 'user', content: 'hi' }] });

    const callHeaders = (mockFetch.mock.calls[0][1] as { headers: Record<string, string> }).headers;
    expect(callHeaders.Authorization).toBeUndefined();
  });

  it('sends Authorization when apiKey is provided', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: 'ok' } }] }),
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'nvapi-abc' });
    await client.invoke({ messages: [{ role: 'user', content: 'hi' }] });

    const callHeaders = (mockFetch.mock.calls[0][1] as { headers: Record<string, string> }).headers;
    expect(callHeaders.Authorization).toBe('Bearer nvapi-abc');
  });

  it('strips trailing slashes from baseUrl', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: 'ok' } }] }),
    });

    const client = createLLMClient({ baseUrl: 'http://localhost:11434/v1///', apiKey: '' });
    await client.invoke({ messages: [{ role: 'user', content: 'hi' }] });

    expect(mockFetch.mock.calls[0][0]).toBe('http://localhost:11434/v1/chat/completions');
  });

  it('invokes with correct payload', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: 'Hello!' } }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        model: 'test-model',
      }),
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'test-key', temperature: 0.3 });
    const resp = await client.invoke({
      system: 'You are helpful',
      messages: [{ role: 'user', content: 'Hi' }],
      temperature: 0.5,
      model: 'my-model',
    });

    expect(resp.content).toBe('Hello!');
    expect(resp.usage).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15 });
    expect(resp.model).toBe('test-model');

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.model).toBe('my-model');
    expect(body.temperature).toBe(0.5);
    expect(body.messages).toHaveLength(2);
    expect(body.messages[0]).toEqual({ role: 'system', content: 'You are helpful' });
    expect(body.messages[1]).toEqual({ role: 'user', content: 'Hi' });
  });

  it('uses config.temperature when not provided in request', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: 'Hi' } }] }),
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key', temperature: 0.7 });
    await client.invoke({ messages: [{ role: 'user', content: 'Hi' }] });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.temperature).toBe(0.7);
  });

  it('sets jsonMode response_format', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: '{"key": "value"}' } }] }),
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await client.invoke({
      messages: [{ role: 'user', content: 'Output JSON' }],
      jsonMode: true,
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.response_format).toEqual({ type: 'json_object' });
  });

  it('includes maxTokens when provided', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: 'Hi' } }] }),
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await client.invoke({
      messages: [{ role: 'user', content: 'Hi' }],
      maxTokens: 100,
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.max_tokens).toBe(100);
  });

  it('includes maxTokens in stream request when provided', async () => {
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"H"}}]}\n\n'));
        controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"i"}}]}\n\n'));
        controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        controller.close();
      },
    });
    mockFetch.mockResolvedValue({
      ok: true,
      body: mockStream,
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await client.stream({
      messages: [{ role: 'user', content: 'Hi' }],
      maxTokens: 50,
    }, () => {});

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.max_tokens).toBe(50);
  });

  // --- Error classification tests ---

  it('throws InvalidApiKeyError on 401 response', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      text: async () => 'Invalid API key',
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key', providerLabel: 'NVIDIA NIM' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow(InvalidApiKeyError);
    await expect(client.invoke({ messages: [] })).rejects.toThrow('Invalid API key for NVIDIA NIM');
  });

  it('throws InvalidApiKeyError on 403 response (forbidden)', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 403,
      statusText: 'Forbidden',
      text: async () => 'Forbidden',
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key', providerLabel: 'NVIDIA NIM' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow(InvalidApiKeyError);
    await expect(client.invoke({ messages: [] })).rejects.toThrow('API key rejected');
  });

  it('throws RateLimitError on 429 response', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 429,
      statusText: 'Too Many Requests',
      headers: new Map([['retry-after', '60']]),
      text: async () => 'Rate limited',
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key', providerLabel: 'NVIDIA NIM' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow(RateLimitError);
    await expect(client.invoke({ messages: [] })).rejects.toThrow('rate limit exceeded');
  });

  it('throws ProviderUnreachableError on 500 response', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      text: async () => 'Server error',
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key', providerLabel: 'NVIDIA NIM' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow(ProviderUnreachableError);
    await expect(client.invoke({ messages: [] })).rejects.toThrow('Cannot reach');
  });

  it('throws ProviderUnreachableError on 503 response (retryable)', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 503,
      statusText: 'Service Unavailable',
      text: async () => 'Unavailable',
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key', providerLabel: 'NVIDIA NIM' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow(ProviderUnreachableError);
    const error = await client.invoke({ messages: [] }).catch(e => e);
    expect(error.retryable).toBe(true);
  });

  it('throws ModelNotFoundError on 404 response', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      text: async () => 'Model not found',
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key', providerLabel: 'NVIDIA NIM' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow(ModelNotFoundError);
  });

  it('throws GenerationFailedError on 400 response', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      text: async () => 'Bad request',
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key', providerLabel: 'NVIDIA NIM' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow(GenerationFailedError);
  });

  it('throws ProviderUnreachableError on network error', async () => {
    mockFetch.mockRejectedValue(new TypeError('Failed to fetch'));

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow(ProviderUnreachableError);
  });

  it('throws StreamInterruptedError on AbortError during streaming', async () => {
    const abortError = new DOMException('Aborted', 'AbortError');
    mockFetch.mockRejectedValue(abortError);

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await expect(client.stream({ messages: [] }, () => {})).rejects.toThrow(StreamInterruptedError);
  });

  it('throws StreamInterruptedError when signal is aborted', async () => {
    const controller = new AbortController();
    controller.abort();

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await expect(client.stream({ messages: [] }, () => {}, controller.signal)).rejects.toThrow(StreamInterruptedError);
  });

  it('throws when no choices in response', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [] }),
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow(GenerationFailedError);
  });

  // --- Tool calling tests ---

  it('invoke parses tool_calls from a 200 response', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{
          message: {
            role: 'assistant',
            content: '',
            tool_calls: [{
              id: 'call_1',
              type: 'function',
              function: { name: 'profile_column', arguments: '{"dataset":"employees"}' }
            }],
            finish_reason: 'tool_calls'
          }
        }]
      }),
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    const resp = await client.invoke({
      model: 'm',
      messages: [{ role: 'user', content: 'q' }],
      tools: [{ type: 'function', function: { name: 'profile_column', description: 'd', parameters: {} } }],
      toolChoice: 'auto'
    });

    expect(resp.toolCalls?.[0].function.name).toBe('profile_column');
    expect(resp.toolCalls?.[0].id).toBe('call_1');
    expect(resp.finishReason).toBe('tool_calls');
  });

  it('invoke serializes tools and tool_choice into the request body', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: 'ok' } }] }),
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await client.invoke({
      model: 'm',
      messages: [{ role: 'user', content: 'q' }],
      tools: [{ type: 'function', function: { name: 'profile_column', description: 'd', parameters: {} } }],
      toolChoice: 'auto'
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.tools[0].function.name).toBe('profile_column');
    expect(body.tool_choice).toBe('auto');
  });

  it('invoke serializes tool-result messages with tool_call_id', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: 'ok' } }] }),
    });

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await client.invoke({
      model: 'm',
      messages: [
        { role: 'assistant', content: '', toolCalls: [{ id: 'call_1', type: 'function', function: { name: 'profile_column', arguments: '{}' } }] },
        { role: 'tool', content: '{"result": 1}', toolCallId: 'call_1' }
      ],
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.messages[0].tool_calls).toBeDefined();
    expect(body.messages[0].tool_calls[0].id).toBe('call_1');
    expect(body.messages[1].role).toBe('tool');
    expect(body.messages[1].tool_call_id).toBe('call_1');
  });

  it('invoke honors an external abort signal', async () => {
    const controller = new AbortController();
    controller.abort();

    const client = createLLMClient({ baseUrl: 'https://integrate.api.nvidia.com/v1', apiKey: 'key' });
    await expect(client.invoke({ messages: [{ role: 'user', content: 'hi' }] }, controller.signal)).rejects.toThrow();
  });

  it('does not discard a response for containing budget-related words', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { content: 'The token budget for exceeding limits is 100k.' } }],
      }),
    });

    const client = createLLMClient({ baseUrl: 'http://localhost:11434/v1', apiKey: '' });
    const resp = await client.invoke({ messages: [{ role: 'user', content: 'q' }] });
    expect(resp.content).toBe('The token budget for exceeding limits is 100k.');
  });

  it('does not drop a data line split across chunk boundaries', async () => {
    const line = 'data: {"choices":[{"delta":{"content":"SPLIT_TOKEN"}}]}\n';
    const splitAt = 30;
    const mockStream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(line.slice(0, splitAt)));
        controller.enqueue(new TextEncoder().encode(line.slice(splitAt)));
        controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        controller.close();
      },
    });
    mockFetch.mockResolvedValue({ ok: true, body: mockStream });

    const tokens: string[] = [];
    const client = createLLMClient({ baseUrl: 'http://localhost:11434/v1', apiKey: '' });
    await client.stream({ messages: [{ role: 'user', content: 'hi' }] }, (t: string) => tokens.push(t));

    expect(tokens.join('')).toContain('SPLIT_TOKEN');
  });

  it('retries a jsonMode request without response_format when the server rejects it with 400', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        text: async () => 'response_format not supported',
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ choices: [{ message: { content: '{"ok": true}' } }] }),
      });

    const client = createLLMClient({ baseUrl: 'http://localhost:11434/v1', apiKey: '' });
    const resp = await client.invoke({ messages: [{ role: 'user', content: 'q' }], jsonMode: true });

    expect(resp.content).toBe('{"ok": true}');
    expect(mockFetch).toHaveBeenCalledTimes(2);
    const secondBody = JSON.parse((mockFetch.mock.calls[1][1] as { body: string }).body);
    expect(secondBody.response_format).toBeUndefined();
  });

  it('remembers a jsonMode rejection and skips response_format on subsequent calls', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        text: async () => 'response_format not supported',
      })
      .mockResolvedValue({
        ok: true,
        json: async () => ({ choices: [{ message: { content: 'ok' } }] }),
      });

    const client = createLLMClient({ baseUrl: 'http://localhost:11434/v1', apiKey: '' });
    await client.invoke({ messages: [{ role: 'user', content: 'q1' }], jsonMode: true });
    await client.invoke({ messages: [{ role: 'user', content: 'q2' }], jsonMode: true });

    // First invoke: 400 + retry without jsonMode; second invoke: jsonMode skipped
    expect(mockFetch).toHaveBeenCalledTimes(3);
    const retryBody = JSON.parse((mockFetch.mock.calls[1][1] as { body: string }).body);
    expect(retryBody.response_format).toBeUndefined();
    const secondBody = JSON.parse((mockFetch.mock.calls[2][1] as { body: string }).body);
    expect(secondBody.response_format).toBeUndefined();
  });
});