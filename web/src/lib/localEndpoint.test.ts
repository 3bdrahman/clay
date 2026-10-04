import { describe, expect, it } from 'vitest';
import { getLocalRequestOptions, inspectLocalServerUrl, normalizeLocalServerUrl } from './localEndpoint';

describe('local endpoint boundary', () => {
  it.each([
    [' http://localhost:1234 ', 'http://localhost:1234/v1'],
    ['http://127.0.0.1:11434/v1///', 'http://127.0.0.1:11434/v1'],
    ['https://models.example/api/v1/', 'https://models.example/api/v1'],
  ])('normalizes %s consistently for discovery and chat', (input, expected) => {
    expect(normalizeLocalServerUrl(input)).toBe(expected);
  });

  it.each(['', 'localhost:1234', 'file:///models', 'http://0.0.0.0:1234/v1', 'http://user:password@localhost:1234/v1', 'http://localhost:1234/v1?key=secret', 'http://localhost:1234/v1#token', 'http://localhost:1234/v1/models', 'http://localhost:1234/v1/chat/completions'])('rejects unusable or ambiguous base URL %s', input => {
    expect(inspectLocalServerUrl(input).error).toBeTruthy();
    expect(() => normalizeLocalServerUrl(input)).toThrow();
  });

  it('requests loopback permission without sending cookies or custom request headers', () => {
    expect(getLocalRequestOptions('http://localhost:1234/v1')).toEqual({
      mode: 'cors', credentials: 'omit', redirect: 'error', targetAddressSpace: 'loopback',
    });
    expect(getLocalRequestOptions('http://127.0.0.1:11434/v1').targetAddressSpace).toBe('loopback');
  });

  it('uses the LAN address space only for recognized private addresses', () => {
    expect(getLocalRequestOptions('http://192.168.1.50:8000/v1').targetAddressSpace).toBe('local');
    expect(getLocalRequestOptions('https://public-model.example/v1').targetAddressSpace).toBeUndefined();
  });
});
