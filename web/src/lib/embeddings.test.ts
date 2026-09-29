import { describe, it, expect, vi } from 'vitest';
import { createEmbeddingsClient, EMBEDDING_MODEL_ID } from './embeddings';
import { hashText } from './hash';
import type { EmbeddingWorkerLike, EmbedWorkerRequest, EmbedWorkerResponse } from './embeddings';
import { createEmbeddingCache } from './embeddingCache';

function fakeWorkerFactory() {
  const listeners: Array<(ev: MessageEvent) => void> = [];
  const errorListeners: Array<(ev: ErrorEvent) => void> = [];
  const sent: EmbedWorkerRequest[] = [];
  let terminateCalls = 0;
  const worker: EmbeddingWorkerLike = {
    postMessage: (msg) => {
      sent.push(msg);
    },
    addEventListener: (type, listener) => {
      if (type === 'error') errorListeners.push(listener as (ev: ErrorEvent) => void);
      else listeners.push(listener);
    },
    terminate: () => {
      terminateCalls += 1;
    },
  } as EmbeddingWorkerLike;
  return {
    worker,
    sent,
    respond: (resp: EmbedWorkerResponse) => {
      for (const listener of listeners) listener({ data: resp } as MessageEvent);
    },
    fail: (message: string) => {
      for (const listener of errorListeners) listener({ message } as ErrorEvent);
    },
    get terminateCalls() {
      return terminateCalls;
    },
  };
}

describe('createEmbeddingsClient', () => {
  it('embeds a single string through the worker', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const promise = client.embed('hello');
    expect(fake.sent).toHaveLength(1);
    expect(fake.sent[0].type).toBe('embed');
    expect(fake.sent[0].texts).toEqual(['hello']);

    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[1, 0]] });
    await expect(promise).resolves.toEqual([[1, 0]]);
  });

  it('embeds an array of strings in one request, order preserved', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const promise = client.embed(['a', 'b', 'c']);
    expect(fake.sent).toHaveLength(1);
    expect(fake.sent[0].texts).toEqual(['a', 'b', 'c']);

    fake.respond({
      type: 'result',
      id: fake.sent[0].id,
      embeddings: [[3, 0], [2, 0], [1, 0]],
    });
    await expect(promise).resolves.toEqual([[3, 0], [2, 0], [1, 0]]);
  });

  it('empty input never reaches the worker', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    await expect(client.embed([])).resolves.toEqual([]);
    expect(fake.sent).toHaveLength(0);
  });

  it('cache hit skips the worker; uncached inputs are embedded and written back', async () => {
    const fake = fakeWorkerFactory();
    const cache = createEmbeddingCache();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker, cache });

    const first = client.embed('hello');
    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[1, 0]] });
    await first;
    expect(cache.get(EMBEDDING_MODEL_ID, hashText('hello'))).toEqual([1, 0]);

    const second = client.embed('hello');
    expect(fake.sent).toHaveLength(1);
    await expect(second).resolves.toEqual([[1, 0]]);

    const third = client.embed(['hello', 'world']);
    expect(fake.sent).toHaveLength(2);
    expect(fake.sent[1].texts).toEqual(['world']);
    fake.respond({ type: 'result', id: fake.sent[1].id, embeddings: [[2, 0]] });
    await expect(third).resolves.toEqual([[1, 0], [2, 0]]);
  });

  it('interleaves concurrent requests by id', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const first = client.embed('one');
    const second = client.embed('two');
    expect(fake.sent).toHaveLength(2);

    fake.respond({ type: 'result', id: fake.sent[1].id, embeddings: [[22]] });
    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[11]] });

    await expect(first).resolves.toEqual([[11]]);
    await expect(second).resolves.toEqual([[22]]);
  });

  it('rejects with the worker error message', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const promise = client.embed('hello');
    fake.respond({ type: 'error', id: fake.sent[0].id, message: 'model download failed' });

    await expect(promise).rejects.toThrow('model download failed');
  });

  it('ignores malformed worker messages (untrusted boundary)', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const promise = client.embed('hello');
    for (const bad of [null, 'nope', { type: 'result' }, { type: 'result', id: 'x', embeddings: [] }]) {
      fake.respond(bad as unknown as EmbedWorkerResponse);
    }
    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[7]] });

    await expect(promise).resolves.toEqual([[7]]);
  });

  it('shared cache between two clients skips inference for identical text', async () => {
    const firstFake = fakeWorkerFactory();
    const secondFake = fakeWorkerFactory();
    const cache = createEmbeddingCache();
    const first = createEmbeddingsClient({ workerFactory: () => firstFake.worker, cache });
    const second = createEmbeddingsClient({ workerFactory: () => secondFake.worker, cache });

    const p1 = first.embed('shared');
    firstFake.respond({ type: 'result', id: firstFake.sent[0].id, embeddings: [[5]] });
    await p1;

    await expect(second.embed('shared')).resolves.toEqual([[5]]);
    expect(secondFake.sent).toHaveLength(0);
  });

  it('default internal cache: identical text skips the worker even without an injected cache', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const first = client.embed('repeat');
    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[9]] });
    await first;

    await expect(client.embed('repeat')).resolves.toEqual([[9]]);
    expect(fake.sent).toHaveLength(1);
  });

  it('the worker factory is the seam: spawns lazily on the first embed', async () => {
    const fake = fakeWorkerFactory();
    const workerFactory = vi.fn(() => fake.worker);
    const client = createEmbeddingsClient({ workerFactory });
    expect(workerFactory).not.toHaveBeenCalled();

    const promise = client.embed('hello');
    expect(workerFactory).toHaveBeenCalledTimes(1);
    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[1, 0]] });
    await expect(promise).resolves.toEqual([[1, 0]]);
  });

  it('a worker error event rejects the pending embed and the next embed spawns a fresh worker', async () => {
    const firstFake = fakeWorkerFactory();
    const secondFake = fakeWorkerFactory();
    let spawnCount = 0;
    const fakes = [firstFake, secondFake];
    const client = createEmbeddingsClient({ workerFactory: () => fakes[spawnCount++]!.worker });

    const promise = client.embed('doomed text');
    firstFake.fail('worker module failed to load');
    await expect(promise).rejects.toThrow('worker module failed to load');

    const retry = client.embed('retry text');
    secondFake.respond({ type: 'result', id: secondFake.sent[0].id, embeddings: [[3]] });
    await expect(retry).resolves.toEqual([[3]]);
  });

  it('a worker error event does not respawn immediately (no respawn storm)', () => {
    const fake = fakeWorkerFactory();
    const workerFactory = vi.fn(() => fake.worker);
    const client = createEmbeddingsClient({ workerFactory });
    const promise = client.embed('doomed text');
    fake.fail('worker module failed to load');
    void promise.catch(() => {});

    expect(workerFactory).toHaveBeenCalledTimes(1);
  });

  it('dispose terminates the worker once pending embeds drain', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const promise = client.embed('hello');
    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[1, 0]] });
    await promise;

    client.dispose();
    expect(fake.terminateCalls).toBe(1);
  });

  it('dispose with an in-flight embed waits for the drain before terminating', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const promise = client.embed('in-flight');
    client.dispose();
    expect(fake.terminateCalls).toBe(0);

    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[2, 0]] });
    await expect(promise).resolves.toEqual([[2, 0]]);
    expect(fake.terminateCalls).toBe(1);
  });

  it('dispose is a no-op when no worker was ever spawned', () => {
    const workerFactory = vi.fn(() => fakeWorkerFactory().worker);
    const client = createEmbeddingsClient({ workerFactory });
    client.dispose();
    expect(workerFactory).not.toHaveBeenCalled();
  });

  it('rejects when the worker returns an empty vector for a text', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const promise = client.embed(['a', 'b']);
    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[1], []] });
    await expect(promise).rejects.toThrow(/invalid vector/);
  });

  it('rejects when the worker returns non-finite vector values', async () => {
    const fake = fakeWorkerFactory();
    const client = createEmbeddingsClient({ workerFactory: () => fake.worker });

    const promise = client.embed('x');
    fake.respond({ type: 'result', id: fake.sent[0].id, embeddings: [[Number.NaN]] });
    await expect(promise).rejects.toThrow(/invalid vector/);
  });
});
