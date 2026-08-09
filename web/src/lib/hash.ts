/**
 * 32-bit FNV-1a hash. Deterministic, sync, zero-deps. Used as the canonical
 * text-hash for embedding-cache keys and chunk sourceHashes across the app.
 *
 * NOTE: This is FNV-1a (not SHA-256). FNV-1a is chosen because:
 *   - It is sync (no Web Crypto round-trip).
 *   - It is fast in the browser for short-to-medium strings.
 *   - It has zero dependencies.
 *
 * For stronger collision resistance (e.g. adversarial inputs), swap to
 * crypto.subtle.digest('SHA-256', …) — the function signature stays the same.
 */

export function hashText(text: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16);
}
