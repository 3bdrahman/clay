// realmExecutor — the QuickJS realm that runs LLM-generated analysis code.
//
// SECURITY: the realm is the isolation boundary. A fresh QuickJS context per
// execution has NO page globals by construction (fetch, localStorage,
// document, Function, eval all resolve to undefined), so the prompt-injection
// escape class that `new Function` could not close is eliminated here rather
// than enumerated away. The static scan stays as fail-fast defense-in-depth.
//
// Contract: the generated source ends with the bare `result` assignment
// convention; result may be an Arquero table (its rows are unwrapped) or a
// plain value. Tables cannot cross the realm boundary — the raw CSV does,
// and the tables are re-created inside per execution, so generated code
// cannot poison the in-memory store or the realm's own aq namespace for the
// next question.
//
// Arquero's UMD bundle is eval'd into the realm: with no CommonJS present it
// assigns globalThis.aq. The realm has no Web APIs, so real UTF-8 codecs
// (pure JS, below) are provided first — Arquero references both TextDecoder
// and TextEncoder at load.
import { CodeExecutionError } from '../lib/errors';
import { newQuickJSWASMModuleFromVariant } from 'quickjs-emscripten-core';
import type { QuickJSContext, QuickJSWASMModule } from 'quickjs-emscripten-core';
import RELEASE_SYNC from '@jitl/quickjs-wasmfile-release-sync';

/** Wall-clock cap per generated-code execution; the interrupt handler trips it. */
export const EXECUTION_TIMEOUT_MS = 5_000;
/** QuickJS runtime memory cap per realm. */
export const MEMORY_LIMIT_BYTES = 256 * 1024 * 1024;

const SAFE_IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;

// The arquero UMD bundle, loaded lazily as a build-time-inlined string so the
// main bundle does not carry the ~1.4MB source. The promise is cached: the
// import resolves once per session.
let arqueroUmdPromise: Promise<string> | null = null;
export function loadArqueroUmd(): Promise<string> {
  if (arqueroUmdPromise === null) {
    arqueroUmdPromise = import('arquero/dist/arquero.min.js?raw').then(m => m.default);
  }
  return arqueroUmdPromise;
}

// The QuickJS engine: constructed from the release-sync variant explicitly so
// only that one wasm bundle ships (the barrel's getQuickJS pulls all four
// variants, ~4MB). One module instance is shared by every realm; contexts are
// per-execution. The promise is cached.
let quickJSModulePromise: Promise<QuickJSWASMModule> | null = null;
export function loadQuickJS(): Promise<QuickJSWASMModule> {
  if (quickJSModulePromise === null) {
    quickJSModulePromise = newQuickJSWASMModuleFromVariant(RELEASE_SYNC);
  }
  return quickJSModulePromise;
}

// Real UTF-8 codecs for the realm (QuickJS ships no Web APIs). decode is not
// called on the exposed fromCSV(string) surface, but the implementations are
// real rather than load-satisfying dummies.
const REALM_CODECS = `
globalThis.TextDecoder = class TextDecoder {
  decode(input) {
    if (input == null) return '';
    const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
    let out = '';
    let i = 0;
    while (i < bytes.length) {
      const b = bytes[i];
      if (b < 0x80) { out += String.fromCharCode(b); i += 1; }
      else if (b < 0xE0) { out += String.fromCharCode(((b & 0x1F) << 6) | (bytes[i + 1] & 0x3F)); i += 2; }
      else if (b < 0xF0) { out += String.fromCharCode(((b & 0x0F) << 12) | ((bytes[i + 1] & 0x3F) << 6) | (bytes[i + 2] & 0x3F)); i += 3; }
      else {
        out += String.fromCodePoint(((b & 0x07) << 18) | ((bytes[i + 1] & 0x3F) << 12) | ((bytes[i + 2] & 0x3F) << 6) | (bytes[i + 3] & 0x3F));
        i += 4;
      }
    }
    return out;
  }
};
globalThis.TextEncoder = class TextEncoder {
  encode(input) {
    const s = input == null ? '' : String(input);
    const out = [];
    for (const ch of s) {
      const cp = ch.codePointAt(0);
      if (cp < 0x80) out.push(cp);
      else if (cp < 0x800) out.push(0xC0 | (cp >> 6), 0x80 | (cp & 0x3F));
      else if (cp < 0x10000) out.push(0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F));
      else out.push(0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3F), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F));
    }
    return new Uint8Array(out);
  }
};
`;

interface RealmTable {
  name: string;
  csv: string;
}

interface RealmErrorCodeShape {
  kind: 'syntax' | 'timeout' | 'runtime';
  message: string;
}

export function isSafeRealmIdentifier(name: string): boolean {
  return SAFE_IDENTIFIER.test(name);
}

/** Serialize a host-side Arquero table to the CSV string the realm re-parses. */
export function tableToCsv(table: unknown): string {
  const toCsv = (table as { toCSV?: () => string }).toCSV;
  if (typeof toCsv !== 'function') {
    throw new CodeExecutionError(
      'dataset table is not an Arquero table (no toCSV)',
      new Error('no toCSV'),
      { retryable: false },
    );
  }
  return toCsv.call(table);
}

/**
 * Run generated code inside a fresh QuickJS realm. Synchronous work only —
 * the caller (the sandbox worker, or the client's main-thread path when no
 * Worker exists) supplies the already-loaded QuickJS module and arquero
 * source and handles disposal of the returned context.
 */
export function runInRealm(
  // The QuickJSContext constructor type from quickjs-emscripten-core.
  newContext: () => QuickJSContext,
  arqueroUmd: string,
  code: string,
  tables: RealmTable[],
): { rows: unknown; context: QuickJSContext } {
  const vm = newContext();
  vm.runtime.setMemoryLimit(MEMORY_LIMIT_BYTES);
  const start = Date.now();
  vm.runtime.setInterruptHandler(() => Date.now() - start > EXECUTION_TIMEOUT_MS);

  try {
    const codecsResult = vm.unwrapResult(vm.evalCode(REALM_CODECS));
    codecsResult.dispose();
    const aqResult = vm.unwrapResult(vm.evalCode(arqueroUmd));
    aqResult.dispose();

    for (const t of tables) {
      if (!isSafeRealmIdentifier(t.name)) {
        throw new CodeExecutionError(
          `dataset name "${t.name}" is not a safe realm identifier`,
          new Error('unsafe dataset name'),
          { code, retryable: false },
        );
      }
      const loadResult = vm.unwrapResult(
        vm.evalCode(`globalThis.${t.name} = aq.fromCSV(${JSON.stringify(t.csv)});`),
      );
      loadResult.dispose();
    }
    const opResult = vm.unwrapResult(vm.evalCode('globalThis.op = aq.op;'));
    opResult.dispose();

    // evalCode does not wrap in a function, so the generated code (which
    // ends with the bare `result` assignment contract) runs in an IIFE.
    const resultHandle = vm.unwrapResult(
      vm.evalCode(`(function(){ "use strict"; let result; ${code}; return result; })()`),
    );

    let rows: unknown;
    const objectsFn = vm.getProp(resultHandle, 'objects');
    if (vm.typeof(objectsFn) === 'function') {
      const rowsHandle = vm.unwrapResult(vm.callFunction(objectsFn, resultHandle));
      rows = vm.dump(rowsHandle);
      rowsHandle.dispose();
    } else {
      rows = vm.dump(resultHandle);
    }
    objectsFn.dispose();
    resultHandle.dispose();
    return { rows, context: vm };
  } catch (e) {
    throw classifyRealmError(e, start, code);
  }
}

export function classifyRealmError(e: unknown, startedAt: number, code: string): CodeExecutionError {
  const error = e instanceof Error ? e : new Error(String(e));
  const timedOut = Date.now() - startedAt > EXECUTION_TIMEOUT_MS;

  const isSyntaxError = error instanceof SyntaxError ||
    error.name === 'SyntaxError' ||
    error.message.includes('SyntaxError') ||
    error.message.includes('Unexpected token') ||
    error.message.includes('Unexpected end of input');

  const isTimeout = timedOut ||
    error.name === 'TimeoutError' ||
    error.message.includes('timeout') ||
    error.message.includes('timed out') ||
    error.message.includes('interrupted');

  return new CodeExecutionError(
    isSyntaxError ? 'Syntax error in generated code' :
    isTimeout ? 'Code execution timed out' :
    'Runtime error in generated code',
    error,
    {
      code,
      retryable: !isSyntaxError && !isTimeout,
    },
  );
}

export type { RealmErrorCodeShape };

// The sandbox worker's message protocol, shared with the client executor.
export interface SandboxWorkerRequest {
  type: 'execute';
  id: number;
  code: string;
  tables: Array<{ name: string; csv: string }>;
}

export type SandboxWorkerResponse =
  | { type: 'result'; id: number; rows: unknown }
  | { type: 'error'; id: number; kind: 'syntax' | 'timeout' | 'runtime'; message: string };
