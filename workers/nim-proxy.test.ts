import { describe, it, expect } from 'vitest';
import worker from './nim-proxy';

describe('nim-proxy relay', () => {
  it('answers preflights with permissive CORS headers', async () => {
    const req = new Request('https://clay-nim-proxy.example.workers.dev/nim-api/v1/models', { method: 'OPTIONS' });
    const resp = await worker.fetch(req, {});
    expect(resp.status).toBe(200);
    expect(resp.headers.get('Access-Control-Allow-Origin')).toBe('*');
    expect(resp.headers.get('Access-Control-Allow-Headers')).toContain('Authorization');
  });

  it('forwards requests to the NIM origin with the /nim-api prefix stripped', async () => {
    const calls: Array<{ url: string; headers: Headers }> = [];
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      calls.push({ url, headers: new Headers(init?.headers) });
      return new Response('{"data":[]}', { status: 200, headers: { 'content-type': 'application/json' } });
    }) as typeof fetch;

    try {
      const req = new Request('https://clay-nim-proxy.example.workers.dev/nim-api/v1/models', {
        headers: { authorization: 'Bearer nvapi-test' },
      });
      const resp = await worker.fetch(req, {});
      expect(calls).toHaveLength(1);
      expect(calls[0].url).toBe('https://integrate.api.nvidia.com/v1/models');
      expect(calls[0].headers.get('origin')).toBe('https://integrate.api.nvidia.com');
      expect(resp.status).toBe(200);
      expect(resp.headers.get('Access-Control-Allow-Origin')).toBe('*');
    } finally {
      globalThis.fetch = realFetch;
    }
  });

  it('passes non-preflight paths through unchanged when the prefix is absent', async () => {
    const calls: Array<{ url: string }> = [];
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      calls.push({ url });
      return new Response('{}', { status: 200 });
    }) as typeof fetch;

    try {
      const req = new Request('https://clay-nim-proxy.example.workers.dev/health');
      await worker.fetch(req, {});
      expect(calls).toHaveLength(1);
      expect(calls[0].url).toBe('https://integrate.api.nvidia.com/health');
    } finally {
      globalThis.fetch = realFetch;
    }
  });
});
