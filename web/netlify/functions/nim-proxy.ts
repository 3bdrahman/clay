import type { Handler, HandlerEvent, HandlerContext } from '@netlify/functions';

const NIM_TARGET = 'https://integrate.api.nvidia.com';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

const handler: Handler = async (event: HandlerEvent, _context: HandlerContext) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers: CORS_HEADERS, body: '' };
  }

  let path = event.path;
  if (path.startsWith('/nim-api')) {
    path = path.replace('/nim-api', '');
  } else if (path.startsWith('/.netlify/functions/nim-proxy')) {
    path = path.replace('/.netlify/functions/nim-proxy', '');
  }

  if (!path.startsWith('/v1') && !path.startsWith('/v1/')) {
    path = `/v1${path}`;
  }

  const targetUrl = `${NIM_TARGET}${path}${event.rawQuery ? `?${event.rawQuery}` : ''}`;

  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(event.headers)) {
    if (value && key.toLowerCase() !== 'host' && key.toLowerCase() !== 'origin') {
      headers[key] = value;
    }
  }
  headers.origin = NIM_TARGET;
  headers.host = new URL(NIM_TARGET).host;

  const resp = await fetch(targetUrl, {
    method: event.httpMethod,
    headers,
    body: event.httpMethod !== 'GET' && event.httpMethod !== 'HEAD' ? event.body : undefined,
  });

  const respHeaders: Record<string, string> = { ...CORS_HEADERS };
  for (const [key, value] of resp.headers.entries()) {
    if (key.toLowerCase() !== 'content-encoding' &&
        key.toLowerCase() !== 'transfer-encoding' &&
        key.toLowerCase() !== 'access-control-allow-origin') {
      respHeaders[key] = value;
    }
  }

  const body = await resp.text();
  return { statusCode: resp.status, headers: respHeaders, body };
};

export { handler };