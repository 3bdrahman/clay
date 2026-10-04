import { ProviderUnreachableError } from './errors/providerErrors';

export function inspectLocalServerUrl(value: string): { baseUrl: string; error: string | null } {
  const input = value.trim();
  const invalid = (error: string) => ({ baseUrl: '', error });
  if (!input) return invalid('Enter the local server URL, such as http://127.0.0.1:1234/v1.');
  if (!URL.canParse(input)) return invalid('Enter a complete HTTP or HTTPS server URL.');
  const url = new URL(input);
  if (!['http:', 'https:'].includes(url.protocol)) return invalid('The local server URL must use HTTP or HTTPS.');
  if (url.username || url.password || url.search || url.hash) {
    return invalid('Use an API base URL without embedded credentials, query parameters, or a fragment.');
  }
  if (url.hostname === '0.0.0.0' || url.hostname === '[::]') {
    return invalid('This is a server listening address. Use http://127.0.0.1 with the server’s port instead.');
  }
  const path = url.pathname.replace(/\/+$/, '');
  if (path.endsWith('/models') || path.endsWith('/chat/completions')) {
    return invalid('Enter the API base URL ending in /v1, without /models or /chat/completions.');
  }
  return { baseUrl: `${url.origin}${path || '/v1'}`, error: null };
}

export function normalizeLocalServerUrl(value: string): string {
  const endpoint = inspectLocalServerUrl(value);
  if (endpoint.error) {
    throw new ProviderUnreachableError('Local server', undefined, { message: endpoint.error, retryable: false });
  }
  return endpoint.baseUrl;
}

type LocalRequestOptions = Pick<RequestInit, 'mode' | 'credentials' | 'redirect'> & {
  // Ignored by browsers without Local Network Access support.
  targetAddressSpace?: 'loopback' | 'local';
};

export function getLocalRequestOptions(baseUrl: string): LocalRequestOptions {
  const hostname = new URL(baseUrl).hostname.toLowerCase();
  const options: LocalRequestOptions = { mode: 'cors', credentials: 'omit', redirect: 'error' };
  if (hostname === 'localhost' || hostname.endsWith('.localhost') || hostname === '[::1]' || /^127\.\d+\.\d+\.\d+$/.test(hostname)) {
    return { ...options, targetAddressSpace: 'loopback' };
  }
  const octets = hostname.split('.').map(Number);
  const isPrivateV4 = octets.length === 4 && octets.every(octet => Number.isInteger(octet) && octet >= 0 && octet <= 255)
    && (octets[0] === 10 || (octets[0] === 192 && octets[1] === 168)
      || (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31)
      || (octets[0] === 169 && octets[1] === 254));
  const isPrivateV6 = /^\[(?:fc|fd|fe[89ab])/i.test(hostname);
  if (isPrivateV4 || isPrivateV6 || hostname.endsWith('.local')) {
    return { ...options, targetAddressSpace: 'local' };
  }
  return options;
}

export function localConnectionError(baseUrl: string, cause: unknown): ProviderUnreachableError {
  const origin = typeof window !== 'undefined' ? window.location.origin : 'the Clay page origin';
  return new ProviderUnreachableError('Local server', cause instanceof Error ? cause : undefined, {
    message: `Cannot connect to ${baseUrl}. Check that the server is running, its CORS settings allow ${origin}, and this site has local-network access in your browser. A browser fetch error alone cannot identify which check failed.`,
    retryable: false,
  });
}
