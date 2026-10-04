import { afterEach, expect, it, vi } from 'vitest';
import { listLocalCatalog } from './models';

afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

it('normalizes the server root and requests browser loopback access without cloud credentials', async () => {
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ data: [{ id: 'installed-model' }] })));
  const models = await listLocalCatalog(' http://127.0.0.1:1234 ', 'never-forward-this');
  expect(models[0].id).toBe('installed-model');
  expect(fetchMock).toHaveBeenCalledWith('http://127.0.0.1:1234/v1/models', expect.objectContaining({
    targetAddressSpace: 'loopback', mode: 'cors', credentials: 'omit', redirect: 'error',
  }));
  expect(JSON.stringify(fetchMock.mock.calls)).not.toContain('never-forward-this');
});

it.each(['', 'http://0.0.0.0:1234', 'http://user:secret@localhost:1234/v1', 'http://localhost:1234/v1/models'])('rejects invalid local base %s without contacting a default host', async value => {
  const fetchMock = vi.spyOn(globalThis, 'fetch');
  await expect(listLocalCatalog(value, '')).rejects.toMatchObject({ retryable: false });
  expect(fetchMock).not.toHaveBeenCalled();
});

it('reports network failure as ambiguous rather than asserting CORS was the cause', async () => {
  vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'));
  await expect(listLocalCatalog('http://127.0.0.1:1234/v1', '')).rejects.toMatchObject({
    code: 'PROVIDER_UNREACHABLE', retryable: false,
    message: expect.stringContaining('browser'),
  });
});

it.each(['<html>not an API</html>', '{"models":[]}', '{"data":[{"name":"wrong"}]}'])('rejects malformed local catalogs', async body => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(body));
  await expect(listLocalCatalog('http://127.0.0.1:1234/v1', '')).rejects.toMatchObject({
    code: 'PROVIDER_UNREACHABLE', retryable: false,
    message: expect.stringContaining('catalog'),
  });
});

it('bounds local discovery through response body reading', async () => {
  vi.useFakeTimers();
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, init) => ({
    ok: true,
    json: () => new Promise((_resolve, reject) => init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))),
  }) as Response);
  const pending = listLocalCatalog('http://127.0.0.1:1234/v1', '');
  const outcome = pending.catch(error => error);
  await vi.advanceTimersByTimeAsync(20_000);
  await expect(outcome).resolves.toMatchObject({ code: 'PROVIDER_TIMEOUT' });
  expect(vi.getTimerCount()).toBe(0);
});
