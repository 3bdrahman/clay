// @vitest-environment node
import { describe, expect, it, vi, afterEach } from 'vitest';
import worker from './nim-proxy';

const ALLOWED_ORIGIN = 'https://app.example';
const APPROVED_NIM_MODEL = 'nvidia/nemotron-3-super-120b-a12b';
const env = { ALLOWED_ORIGIN };

function request(path: string, init: RequestInit = {}): Request {
  return new Request(`https://relay.example${path}`, {
    ...init,
    headers: {
      Origin: ALLOWED_ORIGIN,
      Authorization: 'Bearer nvapi-user-key',
      ...(init.headers ?? {}),
    },
  });
}

async function json(response: Response): Promise<{ error: { code: string; message: string } }> {
  return response.json() as Promise<{ error: { code: string; message: string } }>;
}

function streamFromText(text: string): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('nim proxy worker', () => {
  it('forwards GET /v1/models to the fixed NVIDIA upstream with only caller bearer auth', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ data: [{ id: 'nvidia/llama' }] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );

    const response = await worker.fetch(
      request('/v1/models', {
        method: 'GET',
        headers: {
          Origin: ALLOWED_ORIGIN,
          Authorization: 'Bearer nvapi-user-key',
          'X-Forward-Me': 'no',
        },
      }),
      env,
    );

    expect(response.status).toBe(200);
    expect(response.headers.get('Access-Control-Allow-Origin')).toBe(ALLOWED_ORIGIN);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0]?.[0]).toBe('https://integrate.api.nvidia.com/v1/models');
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe('GET');
    expect(init.redirect).toBe('manual');
    const headers = init.headers as Headers;
    expect(headers.get('Authorization')).toBe('Bearer nvapi-user-key');
    expect(headers.get('X-Forward-Me')).toBeNull();
    expect(headers.get('Origin')).toBeNull();
    expect(init.body).toBeUndefined();
  });

  it('forwards POST /v1/chat/completions with an approved bounded JSON body and preserves Retry-After', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(streamFromText('data: {"choices":[]}\n\n'), {
        status: 429,
        headers: {
          'Content-Type': 'text/event-stream',
          'Retry-After': '7',
        },
      }),
    );

    const response = await worker.fetch(
      request('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: APPROVED_NIM_MODEL, messages: [{ role: 'user', content: 'hi' }], stream: true }),
      }),
      env,
    );

    expect(response.status).toBe(429);
    expect(response.headers.get('Retry-After')).toBe('7');
    expect(response.headers.get('Access-Control-Expose-Headers')).toContain('Retry-After');
    expect(response.headers.get('Content-Type')).toBe('text/event-stream');
    expect(await response.text()).toBe('data: {"choices":[]}\n\n');
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(ArrayBuffer);
    expect(new TextDecoder().decode(init.body as ArrayBuffer)).toContain(`"${APPROVED_NIM_MODEL}"`);
    const headers = init.headers as Headers;
    expect(headers.get('Authorization')).toBe('Bearer nvapi-user-key');
    expect(headers.get('Content-Type')).toBe('application/json');
  });

  it('handles CORS preflight only for the configured origin and allowed methods', async () => {
    const response = await worker.fetch(
      new Request('https://relay.example/v1/chat/completions', {
        method: 'OPTIONS',
        headers: {
          Origin: ALLOWED_ORIGIN,
          'Access-Control-Request-Method': 'POST',
        },
      }),
      env,
    );

    expect(response.status).toBe(204);
    expect(response.headers.get('Access-Control-Allow-Origin')).toBe(ALLOWED_ORIGIN);
    expect(response.headers.get('Access-Control-Allow-Methods')).toContain('POST');
    expect(response.headers.get('Access-Control-Allow-Headers')).toContain('Authorization');
  });

  it('rejects preflight requests for unsupported paths without contacting NVIDIA', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const response = await worker.fetch(
      new Request('https://relay.example/v1/files', {
        method: 'OPTIONS',
        headers: {
          Origin: ALLOWED_ORIGIN,
          'Access-Control-Request-Method': 'POST',
        },
      }),
      env,
    );

    expect(response.status).toBe(404);
    expect((await json(response)).error.code).toBe('route_not_found');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects disallowed origins without contacting NVIDIA', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const response = await worker.fetch(
      new Request('https://relay.example/v1/models', {
        method: 'GET',
        headers: {
          Origin: 'https://evil.example',
          Authorization: 'Bearer nvapi-user-key',
        },
      }),
      env,
    );

    expect(response.status).toBe(403);
    expect((await json(response)).error.code).toBe('origin_not_allowed');
    expect(response.headers.get('Access-Control-Allow-Origin')).toBe(ALLOWED_ORIGIN);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('requires a caller bearer key and never uses a server-side key', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');
    const envWithServerKey = { ...env, NIM_API_KEY: 'server-key-must-not-be-used' };

    const response = await worker.fetch(
      new Request('https://relay.example/v1/models', {
        method: 'GET',
        headers: { Origin: ALLOWED_ORIGIN },
      }),
      envWithServerKey,
    );

    expect(response.status).toBe(401);
    expect((await json(response)).error.code).toBe('authorization_required');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects unsupported paths, query strings, and methods without arbitrary forwarding', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const unsupportedPath = await worker.fetch(request('/v1/files', { method: 'GET' }), env);
    const queryAbuse = await worker.fetch(request('/v1/models?target=https://example.test', { method: 'GET' }), env);
    const wrongMethod = await worker.fetch(request('/v1/models', { method: 'POST' }), env);

    expect(unsupportedPath.status).toBe(404);
    expect(queryAbuse.status).toBe(400);
    expect(wrongMethod.status).toBe(405);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns a JSON upstream error when NVIDIA cannot be reached', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('network unavailable'));

    const response = await worker.fetch(request('/v1/models', { method: 'GET' }), env);

    expect(response.status).toBe(502);
    expect(response.headers.get('Cache-Control')).toBe('no-store');
    expect((await json(response)).error.code).toBe('upstream_unreachable');
  });

  it.each([
    ['OpenRouter-only free model', 'qwen/qwen3.8-27b:free'],
    ['unapproved paid model', 'nvidia/nemotron-3-ultra-550b-a55b'],
    ['arbitrary model id', 'nvidia/not-a-demo-model'],
  ])('rejects %s before contacting NVIDIA', async (_label, model) => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const response = await worker.fetch(
      request('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: 'hi' }] }),
      }),
      env,
    );

    expect(response.status).toBe(400);
    expect((await json(response)).error.code).toBe('model_not_approved');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects chat requests when model is missing or not a string', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const response = await worker.fetch(
      request('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 42, messages: [{ role: 'user', content: 'hi' }] }),
      }),
      env,
    );

    expect(response.status).toBe(400);
    expect((await json(response)).error.code).toBe('model_required');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects malformed JSON without contacting NVIDIA', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const response = await worker.fetch(
      request('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{"model":',
      }),
      env,
    );

    expect(response.status).toBe(400);
    expect((await json(response)).error.code).toBe('invalid_json');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns a clear CORS JSON 400 when request body reading fails', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');
    const brokenBody = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.error(new Error('stream broke'));
      },
    });

    const response = await worker.fetch(
      request('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: brokenBody,
        duplex: 'half',
      } as RequestInit),
      env,
    );

    expect(response.status).toBe(400);
    expect(response.headers.get('Access-Control-Allow-Origin')).toBe(ALLOWED_ORIGIN);
    expect((await json(response)).error.code).toBe('body_read_failed');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('enforces the streaming body size limit rather than trusting Content-Length alone', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');
    const oversizedBody = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(2 * 1024 * 1024));
        controller.enqueue(new Uint8Array(1));
        controller.close();
      },
    });

    const response = await worker.fetch(
      request('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': '12' },
        body: oversizedBody,
        duplex: 'half',
      } as RequestInit),
      env,
    );

    expect(response.status).toBe(413);
    expect((await json(response)).error.code).toBe('request_too_large');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
