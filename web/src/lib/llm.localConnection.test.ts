import { afterEach, expect, it, vi } from 'vitest';
import { createLLMClient } from './llm';

afterEach(() => vi.restoreAllMocks());

it('uses the same normalized base and browser permission options for local invoke and stream', async () => {
  const fetchMock = vi.spyOn(globalThis, 'fetch');
  const client = createLLMClient({ baseUrl: ' http://127.0.0.1:1234 ', apiKey: '', providerKind: 'local' });
  const request = { model: 'installed-model', messages: [{ role: 'user' as const, content: 'Hello' }] };
  fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ choices: [{ message: { content: 'Hello' } }] })));
  expect((await client.invoke(request)).content).toBe('Hello');
  fetchMock.mockResolvedValueOnce(new Response('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'));
  expect((await client.stream(request, () => {})).content).toBe('Hello');
  for (const [url, init] of fetchMock.mock.calls) {
    expect(url).toBe('http://127.0.0.1:1234/v1/chat/completions');
    expect(init).toMatchObject({ targetAddressSpace: 'loopback', mode: 'cors', credentials: 'omit', redirect: 'error' });
    expect(new Headers(init?.headers).get('Authorization')).toBeNull();
  }
});

it('reports local invoke and stream network failures without claiming a proven CORS block', async () => {
  vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));
  const client = createLLMClient({ baseUrl: 'http://127.0.0.1:1234/v1', apiKey: '', providerKind: 'local' });
  const request = { model: 'installed-model', messages: [] };
  await expect(client.invoke(request)).rejects.toMatchObject({ code: 'PROVIDER_UNREACHABLE', retryable: false, message: expect.stringContaining('local-network') });
  await expect(client.stream(request, () => {})).rejects.toMatchObject({ code: 'PROVIDER_UNREACHABLE', retryable: false, message: expect.stringContaining('local-network') });
});
