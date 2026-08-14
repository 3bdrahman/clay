import type { Handler, HandlerEvent, HandlerContext } from '@netlify/functions';

const NIM_TARGET = 'https://integrate.api.nvidia.com';

const handler: Handler = async (event: HandlerEvent, _context: HandlerContext) => {
  const path = event.path.replace('/.netlify/functions/nim-proxy', '');
  const targetUrl = `${NIM_TARGET}${path}`;

  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(event.headers)) {
    if (value) headers[key] = value;
  }
  headers.origin = NIM_TARGET;
  headers.host = new URL(NIM_TARGET).host;

  const resp = await fetch(targetUrl, {
    method: event.httpMethod,
    headers,
    body: event.httpMethod !== 'GET' && event.httpMethod !== 'HEAD' ? event.body : undefined,
  });

  const respHeaders: Record<string, string> = {};
  for (const [key, value] of resp.headers.entries()) {
    if (key.toLowerCase() !== 'content-encoding' && key.toLowerCase() !== 'transfer-encoding') {
      respHeaders[key] = value;
    }
  }
  respHeaders['Access-Control-Allow-Origin'] = '*';
  respHeaders['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS';
  respHeaders['Access-Control-Allow-Headers'] = 'Content-Type, Authorization';

  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers: respHeaders, body: '' };
  }

  const body = await resp.text();
  return { statusCode: resp.status, headers: respHeaders, body };
};

export { handler };