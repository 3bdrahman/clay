const NIM_API_ORIGIN = 'https://integrate.api.nvidia.com';
const PROXY_PATH_PREFIX = '/nim-api';

function applyCorsHeaders(headers: Headers): Headers {
  headers.set('Access-Control-Allow-Origin', '*');
  headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  return headers;
}

export default {
  async fetch(request: Request, _env: unknown): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname.startsWith(PROXY_PATH_PREFIX)
      ? url.pathname.slice(PROXY_PATH_PREFIX.length)
      : url.pathname;
    const targetUrl = `${NIM_API_ORIGIN}${path}${url.search}`;

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 200, headers: applyCorsHeaders(new Headers()) });
    }

    const headers = new Headers(request.headers);
    headers.set('origin', NIM_API_ORIGIN);

    const resp = await fetch(targetUrl, {
      method: request.method,
      headers,
      body: request.method !== 'GET' && request.method !== 'HEAD' ? request.body : undefined,
    });

    return new Response(resp.body, {
      status: resp.status,
      headers: applyCorsHeaders(new Headers(resp.headers)),
    });
  },
};
