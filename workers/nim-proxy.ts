import { isApprovedCloudModel } from '../web/src/lib/modelPolicy';

const NIM_API_ORIGIN = 'https://integrate.api.nvidia.com';
const DEFAULT_ALLOWED_ORIGIN = 'https://3bdrahman.github.io';
const MAX_CHAT_COMPLETIONS_BODY_BYTES = 2 * 1024 * 1024;

const ROUTES = {
  models: '/v1/models',
  chatCompletions: '/v1/chat/completions',
} as const;

interface NimProxyEnv {
  ALLOWED_ORIGIN?: string;
}

interface Route {
  upstreamUrl: string;
  method: 'GET' | 'POST';
  needsJsonBody: boolean;
}

interface ChatCompletionBody {
  model: string;
}

const JSON_HEADERS = {
  'Content-Type': 'application/json; charset=utf-8',
  'Cache-Control': 'no-store',
} as const;

function getAllowedOrigin(env: NimProxyEnv): string {
  return env.ALLOWED_ORIGIN?.trim() || DEFAULT_ALLOWED_ORIGIN;
}

function corsHeaders(env: NimProxyEnv): Headers {
  const headers = new Headers();
  headers.set('Access-Control-Allow-Origin', getAllowedOrigin(env));
  headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  headers.set('Access-Control-Allow-Headers', 'Authorization, Content-Type');
  headers.set('Access-Control-Expose-Headers', 'Retry-After');
  headers.set('Vary', 'Origin');
  headers.set('Cache-Control', 'no-store');
  return headers;
}

function jsonResponse(env: NimProxyEnv, status: number, code: string, message: string): Response {
  const headers = corsHeaders(env);
  headers.set('Content-Type', JSON_HEADERS['Content-Type']);
  headers.set('Cache-Control', JSON_HEADERS['Cache-Control']);

  return new Response(JSON.stringify({ error: { code, message } }), {
    status,
    headers,
  });
}

function routeFor(request: Request, url: URL, env: NimProxyEnv): Route | Response {
  if (url.search !== '') {
    return jsonResponse(env, 400, 'query_not_supported', 'NIM relay endpoints do not accept URL query parameters.');
  }

  if (url.pathname === ROUTES.models) {
    if (request.method !== 'GET') {
      return jsonResponse(env, 405, 'method_not_allowed', 'Use GET /v1/models for the NIM model catalog.');
    }
    return {
      upstreamUrl: `${NIM_API_ORIGIN}${ROUTES.models}`,
      method: 'GET',
      needsJsonBody: false,
    };
  }

  if (url.pathname === ROUTES.chatCompletions) {
    if (request.method !== 'POST') {
      return jsonResponse(env, 405, 'method_not_allowed', 'Use POST /v1/chat/completions for NIM chat completions.');
    }
    return {
      upstreamUrl: `${NIM_API_ORIGIN}${ROUTES.chatCompletions}`,
      method: 'POST',
      needsJsonBody: true,
    };
  }

  return jsonResponse(env, 404, 'route_not_found', 'NIM relay supports only /v1/models and /v1/chat/completions.');
}

function rejectInvalidPreflightTarget(request: Request, env: NimProxyEnv): Response | null {
  const url = new URL(request.url);
  if (url.search !== '') {
    return jsonResponse(env, 400, 'query_not_supported', 'NIM relay endpoints do not accept URL query parameters.');
  }

  const requestedMethod = request.headers.get('Access-Control-Request-Method')?.toUpperCase();
  if (url.pathname === ROUTES.models) {
    if (requestedMethod !== undefined && requestedMethod !== 'GET') {
      return jsonResponse(env, 405, 'method_not_allowed', 'Use GET /v1/models for the NIM model catalog.');
    }
    return null;
  }

  if (url.pathname === ROUTES.chatCompletions) {
    if (requestedMethod !== undefined && requestedMethod !== 'POST') {
      return jsonResponse(env, 405, 'method_not_allowed', 'Use POST /v1/chat/completions for NIM chat completions.');
    }
    return null;
  }

  return jsonResponse(env, 404, 'route_not_found', 'NIM relay supports only /v1/models and /v1/chat/completions.');
}

function rejectIfOriginDisallowed(request: Request, env: NimProxyEnv): Response | null {
  const origin = request.headers.get('Origin');
  if (origin === getAllowedOrigin(env)) return null;
  return jsonResponse(env, 403, 'origin_not_allowed', 'This origin is not allowed to use the NIM relay.');
}

function rejectIfMissingBearer(request: Request, env: NimProxyEnv): Response | null {
  const authorization = request.headers.get('Authorization');
  if (authorization?.startsWith('Bearer ') && authorization.slice('Bearer '.length).trim() !== '') return null;
  return jsonResponse(env, 401, 'authorization_required', 'Pass your NVIDIA NIM API key as a Bearer token.');
}

function rejectIfUnsupportedContentType(request: Request, env: NimProxyEnv): Response | null {
  const contentType = request.headers.get('Content-Type');
  if (contentType?.toLowerCase().includes('application/json')) return null;
  return jsonResponse(env, 415, 'json_required', 'POST /v1/chat/completions requires an application/json request body.');
}

async function readBoundedBody(request: Request, env: NimProxyEnv): Promise<Uint8Array | Response> {
  const declaredLength = request.headers.get('Content-Length');
  if (declaredLength !== null) {
    const length = Number.parseInt(declaredLength, 10);
    if (!Number.isFinite(length) || length < 0) {
      return jsonResponse(env, 400, 'invalid_content_length', 'Content-Length must be a non-negative integer.');
    }
    if (length > MAX_CHAT_COMPLETIONS_BODY_BYTES) {
      return jsonResponse(env, 413, 'request_too_large', 'NIM chat completion requests must be 2 MiB or smaller.');
    }
  }

  if (request.body === null) {
    return jsonResponse(env, 400, 'body_required', 'POST /v1/chat/completions requires a JSON request body.');
  }

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > MAX_CHAT_COMPLETIONS_BODY_BYTES) {
      await reader.cancel();
      return jsonResponse(env, 413, 'request_too_large', 'NIM chat completion requests must be 2 MiB or smaller.');
    }
    chunks.push(value);
  }

  const body = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

function bodyReadFailureResponse(env: NimProxyEnv, error: unknown): Response {
  const message =
    error instanceof Error
      ? 'The NIM relay could not read the request body.'
      : 'The NIM relay received an unreadable request body.';
  return jsonResponse(env, 400, 'body_read_failed', message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseChatCompletionBody(body: Uint8Array, env: NimProxyEnv): ChatCompletionBody | Response {
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(body));
  } catch (error: unknown) {
    const message =
      error instanceof SyntaxError
        ? 'POST /v1/chat/completions requires a valid JSON request body.'
        : 'POST /v1/chat/completions request body could not be parsed as JSON.';
    return jsonResponse(env, 400, 'invalid_json', message);
  }

  if (!isRecord(parsed) || typeof parsed.model !== 'string' || parsed.model.trim() === '') {
    return jsonResponse(env, 400, 'model_required', 'POST /v1/chat/completions requires a string model field.');
  }

  if (!isApprovedCloudModel('nim', parsed.model)) {
    return jsonResponse(env, 400, 'model_not_approved', 'This NIM model is not approved for the public Clay demo.');
  }

  return { model: parsed.model };
}

function upstreamHeaders(request: Request, route: Route): Headers {
  const headers = new Headers();
  const authorization = request.headers.get('Authorization');
  if (authorization !== null) headers.set('Authorization', authorization);
  if (route.needsJsonBody) headers.set('Content-Type', 'application/json');
  return headers;
}

function bodyBytesToArrayBuffer(body: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(body.byteLength);
  new Uint8Array(buffer).set(body);
  return buffer;
}

function proxiedResponse(response: Response, env: NimProxyEnv): Response {
  const headers = corsHeaders(env);
  const contentType = response.headers.get('Content-Type');
  const retryAfter = response.headers.get('Retry-After');
  if (contentType !== null) headers.set('Content-Type', contentType);
  if (retryAfter !== null) headers.set('Retry-After', retryAfter);

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function upstreamFailureResponse(env: NimProxyEnv, error: unknown): Response {
  const message =
    error instanceof TypeError
      ? 'The NVIDIA NIM upstream endpoint could not be reached.'
      : 'The NVIDIA NIM upstream request failed before a response was received.';
  return jsonResponse(env, 502, 'upstream_unreachable', message);
}

async function handleOptions(request: Request, env: NimProxyEnv): Promise<Response> {
  const originRejection = rejectIfOriginDisallowed(request, env);
  if (originRejection !== null) return originRejection;

  const targetRejection = rejectInvalidPreflightTarget(request, env);
  if (targetRejection !== null) return targetRejection;

  const requestedMethod = request.headers.get('Access-Control-Request-Method');
  if (requestedMethod !== null && !['GET', 'POST'].includes(requestedMethod.toUpperCase())) {
    return jsonResponse(env, 405, 'method_not_allowed', 'NIM relay preflight allows only GET and POST.');
  }

  return new Response(null, {
    status: 204,
    headers: corsHeaders(env),
  });
}

export default {
  async fetch(request: Request, env: NimProxyEnv): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return handleOptions(request, env);
    }

    const originRejection = rejectIfOriginDisallowed(request, env);
    if (originRejection !== null) return originRejection;

    const url = new URL(request.url);
    const route = routeFor(request, url, env);
    if (route instanceof Response) {
      return route;
    }

    const authRejection = rejectIfMissingBearer(request, env);
    if (authRejection !== null) return authRejection;

    let body: Uint8Array | undefined;
    if (route.needsJsonBody) {
      const contentTypeRejection = rejectIfUnsupportedContentType(request, env);
      if (contentTypeRejection !== null) return contentTypeRejection;

      let boundedBody: Uint8Array | Response;
      try {
        boundedBody = await readBoundedBody(request, env);
      } catch (error: unknown) {
        return bodyReadFailureResponse(env, error);
      }
      if (boundedBody instanceof Response) return boundedBody;
      const parsedBody = parseChatCompletionBody(boundedBody, env);
      if (parsedBody instanceof Response) return parsedBody;
      body = boundedBody;
    }

    try {
      const upstreamResponse = await fetch(route.upstreamUrl, {
        method: route.method,
        redirect: 'manual',
        headers: upstreamHeaders(request, route),
        body: body === undefined ? undefined : bodyBytesToArrayBuffer(body),
      });
      return proxiedResponse(upstreamResponse, env);
    } catch (error: unknown) {
      return upstreamFailureResponse(env, error);
    }
  },
};
