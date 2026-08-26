export default {
  async fetch(request: Request, env: any): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname.replace('/nim-api', '');
    const targetUrl = `https://integrate.api.nvidia.com${path}${url.search}`;

    const headers = new Headers(request.headers);
    headers.set('origin', 'https://integrate.api.nvidia.com');
    headers.set('host', 'integrate.api.nvidia.com');

    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: 200,
        headers: corsHeaders(),
      });
    }

    const resp = await fetch(targetUrl, {
      method: request.method,
      headers,
      body: request.method !== 'GET' && request.method !== 'HEAD' ? request.body : undefined,
    });

    const respHeaders = new Headers(resp.headers);
    setCorsHeaders(respHeaders);

    return new Response(resp.body, {
      status: resp.status,
      headers: respHeaders,
    });
  },
};

function corsHeaders(): Headers {
  const h = new Headers();
  h.set('Access-Control-Allow-Origin', '*');
  h.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  h.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  return h;
}

function setCorsHeaders(h: Headers): void {
  h.set('Access-Control-Allow-Origin', '*');
  h.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  h.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}