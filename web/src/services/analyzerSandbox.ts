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
 *  - Limited scope. The only external references PASSED to the generated
 *    code are:
 *        • the `aq` Arquero namespace (isolated per execution — see below)
 *        • `op` Arquero operators object (fresh clone — see below)
 *        • one parameter per loaded dataset table (Arquero `ColumnTable`)
 *        • the `result` slot reserved for the return value
 *    RESIDUAL RISK: `new Function`'s scope chain terminates at the page's
 *    global scope, so bare references (`fetch(...)`, `localStorage`,
 *    `document`, `eval`, `Function`, `import`) RESOLVE to page globals even
 *    though none are passed in — strict mode removes only accidental
 *    globals, not global reach. The static scan below rejects the known
 *    vectors before execution; a determined injection can still construct
 *    references dynamically (String.fromCharCode etc.), which is the
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
 *  - Full AST-level validation. A regex scan (applied — see
 *    `FORBIDDEN_CODE_PATTERNS`) rejects the known escape vectors before
 *    execution; an AST scan or a real sandbox (`quickjs-emscripten`,
 *    `proxy-tree-walker`) would narrow the remaining dynamic-construction
 *    evasion. The scan feeds its rejection back to the model as a retryable
 *    error result, bounded by the existing loop budgets; a false positive
 *    (a dataset column named exactly `window` or `constructor`) degrades to
 *    retries + salvage synthesis, visibly.
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
/**
 * Static scan of the generated source for the known escape vectors. `new
 * Function`'s scope chain terminates at the page's global scope, so bare
 * references (`fetch(...)`, `localStorage`) resolve to page globals — the
 * scan rejects them before execution. A determined injection can construct
 * references dynamically (String.fromCharCode etc.) and evade the scan; the
 * retry loop feeds the rejection back to the model for self-correction, and
 * both paths are bounded by the existing budgets.
 */
const FORBIDDEN_CODE_PATTERNS: ReadonlyArray<{ pattern: RegExp; label: string }> = [
  { pattern: /\.constructor/, label: 'constructor access' },
  { pattern: /__proto__/, label: 'prototype access' },
  { pattern: /\bglobalThis\b/, label: 'globalThis' },
  { pattern: /\bFunction\b/, label: 'Function' },
  { pattern: /\beval\s*\(/, label: 'eval' },
  { pattern: /\bimport\s*\(/, label: 'dynamic import' },
  { pattern: /\brequire\s*\(/, label: 'require' },
  { pattern: /\bprocess\./, label: 'process' },
  { pattern: /\bwindow\b/, label: 'window' },
  { pattern: /\bdocument\b/, label: 'document' },
  { pattern: /\blocalStorage\b/, label: 'localStorage' },
  { pattern: /\bfetch\s*\(/, label: 'fetch' },
  { pattern: /fromCSV\s*\(\s*['"`]https?:/, label: 'network fromCSV' },
];

function assertCodeSafe(code: string): void {
  for (const { pattern, label } of FORBIDDEN_CODE_PATTERNS) {
    if (pattern.test(code)) {
      throw new CodeExecutionError(
        `generated code references forbidden globals (${label})`,
        new Error(`forbidden: ${label}`),
        { code, retryable: true },
      );
    }
  }
}

export function createUserCodeExecutor(datasets: Map<string, unknown>): (code: string) => unknown {
  function executeUserCode(code: string): unknown {
    assertCodeSafe(code);
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
