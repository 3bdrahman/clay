// Sandbox executor for LLM-generated analysis code. Single responsibility:
// turn generated source into a result value, or a typed CodeExecutionError.

import { CodeExecutionError } from '../lib/errors';

/**
 * SECURITY: Execute LLM-generated JavaScript in a constrained sandbox.
 *
 * ── Trust boundary ────────────────────────────────────────────────────────
 * The source of `code` is an LLM completion, NOT user-typed input. The LLM
 * has been system-prompted to emit only Arquero transformations, but the
 * underlying threat is prompt injection: a malicious document ingested via
 * the vectorstore (or a user-chosen CSV column header) could contain
 * instructions the LLM dutifully echoes into the generated code, after which
 * `new Function(...)` executes them in the host page context.
 *
 * ── Mitigations (current) ─────────────────────────────────────────────────
 *  - Strict mode (`"use strict"`) forbids accidental globals / `with` /
 *    undeclared assignments.
 *  - Limited scope. The only external references reachable from the
 *    generated code are:
 *        • the `aq` Arquero namespace (isolated per execution — see below)
 *        • `op` Arquero operators object (fresh clone — see below)
 *        • one parameter per loaded dataset table (Arquero `ColumnTable`)
 *        • the `result` slot reserved for the return value
 *    No `window`, `globalThis`, `document`, `fetch`, `eval`, `import`,
 *    `require`, `process`, or any DOM/network primitive is passed in. The
 *    generated code can still *reach* the global scope via property chains
 *    such as `(() => {}).constructor.constructor("...")()` — that is the
 *    inherent risk of `new Function` and the reason the document-ingestion
 *    path lives in the same browser tab rather than a worker.
 *
 *  - Input discipline. CSV column names are NOT injected as identifier names
 *    (they're accessed as `d['column name with spaces']`). The only
 *    identifierss derived from user-controlled data are dataset names, which
 *    are themselves produced by `deriveName()` in `services/files.ts` and
 *    stripped to `[a-zA-Z0-9_]`.
 *
 * ── Mitigations NOT applied (and why) ─────────────────────────────────────
 *  - Web Worker isolation. Moving execution to a Worker would buy true
 *    wall-clock isolation (no access to `window`, no synchronous DOM), at
 *    the cost of postMessage serialization of Arquero tables on every call.
 *    Tracked as a follow-up — this is the only call site that would move.
 *  - CSP `unsafe-eval` removal. The Vite dev build and the GitHub Pages
 *    deploy both rely on `new Function`; turning it off would also disable
 *    React refresh in dev. A proper sandbox (`quickjs-emscripten`,
 *    `proxy-tree-walker`) is the long-term fix and is incompatible with the
 *    current "no backend, no wasm" deploy profile.
 *  - Static code validation. We could AST-scan generated code for forbidden
 *    references before execution, but a determined prompt-injection can
 *    construct the same refs dynamically (`[][`constructor`]` etc.). The
 *    retry loop already rejects `SyntaxError` and timeouts; runtime failures
 *    return an error result to the UI instead of crashing the chat.
 *
 * ── Hardening applied here ────────────────────────────────────────────────
 * `op` is passed as a fresh shallow clone so generated code cannot mutate
 * the real `aq.op` and corrupt subsequent Arquero verbs. we do NOT freeze
 * `aq` because Arquero internally relies on the namespace being mutable
 * (attempted and reverted after test regression). Dataset tables are
 * Arquero `ColumnTable` instances (immutable by contract) and are re-read
 * from the `datasets` map on every execution, so generated code cannot
 * poison the in-memory store for the next question.
 *
 * @throws CodeExecutionError for syntax errors, runtime errors, timeouts
 */
export function createUserCodeExecutor(datasets: Map<string, unknown>): (code: string) => unknown {
  function executeUserCode(code: string): unknown {
    const aq = datasets.get('aq') as { op?: Record<string, unknown>; from?: unknown; fromCSV?: unknown } | undefined;
    const datasetsObj: Record<string, unknown> = {};
    for (const [name, table] of datasets) {
      if (name === 'aq') continue;
      datasetsObj[name] = table;
    }
    const opRaw = aq?.op || {};
    // Create a request-scoped aq wrapper to isolate the namespace per execution
    const aqRef = {
      from: aq?.from,
      fromCSV: aq?.fromCSV,
    };
    const opRef = { ...opRaw };
    const argNames = Object.keys(datasetsObj);
    const argValues = Object.values(datasetsObj);

    // eslint-disable-next-line @typescript-eslint/no-implied-eval, no-new-func
    const fn = new Function(
      ...argNames,
      'aq',
      'op',
      '"use strict"; let result; ' + code + '; return result;'
    );

    try {
      const result = fn(...argValues, aqRef, opRef);
      if (result && typeof result === 'object' && typeof (result as { objects?: () => unknown[] }).objects === 'function') {
        return (result as { objects: () => unknown[] }).objects();
      }
      return result;
    } catch (e) {
      const error = e instanceof Error ? e : new Error(String(e));

      const isSyntaxError = error instanceof SyntaxError ||
        error.name === 'SyntaxError' ||
        error.message.includes('SyntaxError') ||
        error.message.includes('Unexpected token') ||
        error.message.includes('Unexpected end of input');

      const isTimeout = error.name === 'TimeoutError' ||
        error.message.includes('timeout') ||
        error.message.includes('timed out');

      throw new CodeExecutionError(
        isSyntaxError ? 'Syntax error in generated code' :
        isTimeout ? 'Code execution timed out' :
        'Runtime error in generated code',
        error,
        {
          code,
          retryable: !isSyntaxError && !isTimeout,
        }
      );
    }
  }

  return executeUserCode;
}
