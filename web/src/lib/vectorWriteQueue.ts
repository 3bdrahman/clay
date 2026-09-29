/**
 * Module-level write coordinator.
 *
 * Why: `addEntries` is synchronous (returns void) but IndexedDB writes are
 * inherently async. Multiple VectorStore instances may exist in tests (and
 * could exist in production during HMR / store resets). When instance A
 * writes an entry and instance B is constructed and loads, B must see A's
 * write — but B's `load()` has no knowledge of A's in-flight IDB operations.
 *
 * Fix: every IDB write is registered on a shared writeQueue promise.
 * Every `load()` awaits writeQueue before reading. Once any `load()`
 * resolves, all writes issued before that load() call have landed in IDB.
 */
let writeQueue: Promise<void> = Promise.resolve();
let writeQueueFailed = false;
let writeQueueFailure: Error | null = null;

export function enqueueWrite(op: () => Promise<unknown>): void {
  writeQueue = writeQueue.then<void>(() => op().then(() => undefined, (e: unknown) => {
    writeQueueFailed = true;
    writeQueueFailure = e instanceof Error ? e : new Error(String(e));
    if (import.meta.env.DEV) console.error('[vectorstore] async op failed:', e);
  }));
}

export function _resetWriteQueue(): void {
  writeQueue = Promise.resolve();
  writeQueueFailed = false;
  writeQueueFailure = null;
}

export function getWriteQueueState(): { failed: boolean; failure: Error | null } {
  return { failed: writeQueueFailed, failure: writeQueueFailure };
}

export function awaitWriteQueue(): Promise<void> {
  return writeQueue;
}