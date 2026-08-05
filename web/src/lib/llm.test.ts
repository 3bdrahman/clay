import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest';
import { createLLMClient, LLMConfigError } from './llm';
import { NIM_BASE_URL } from './providers';

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

  it('throws LLMConfigError when baseUrl is empty', () => {
    expect(() => createLLMClient({ baseUrl: '', apiKey: 'k', providerLabel: 'test' })).toThrow(LLMConfigError);
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

    const client = createLLMClient({ baseUrl: NIM_BASE_URL, apiKey: 'nvapi-abc' });
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

    const client = createLLMClient({ baseUrl: NIM_BASE_URL, apiKey: 'test-key', temperature: 0.3 });
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

    const client = createLLMClient({ baseUrl: NIM_BASE_URL, apiKey: 'key', temperature: 0.7 });
    await client.invoke({ messages: [{ role: 'user', content: 'Hi' }] });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.temperature).toBe(0.7);
  });

  it('sets jsonMode response_format', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [{ message: { content: '{"key": "value"}' } }] }),
    });

    const client = createLLMClient({ baseUrl: NIM_BASE_URL, apiKey: 'key' });
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

    const client = createLLMClient({ baseUrl: NIM_BASE_URL, apiKey: 'key' });
    await client.invoke({
      messages: [{ role: 'user', content: 'Hi' }],
      maxTokens: 100,
    });

    const body = JSON.parse((mockFetch.mock.calls[0][1] as { body: string }).body);
    expect(body.max_tokens).toBe(100);
  });

  it('throws on API error', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      status: 429,
      statusText: 'Too Many Requests',
      text: async () => 'Rate limited',
    });

    const client = createLLMClient({ baseUrl: NIM_BASE_URL, apiKey: 'key' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow('429 Too Many Requests');
  });

  it('throws when no choices in response', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ choices: [] }),
    });

    const client = createLLMClient({ baseUrl: NIM_BASE_URL, apiKey: 'key' });
    await expect(client.invoke({ messages: [] })).rejects.toThrow('No choices in response');
  });
});
