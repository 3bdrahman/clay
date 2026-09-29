// Sandbox executor for LLM-generated analysis code. Single responsibility:
// turn generated source into a result value, or a typed CodeExecutionError.

import { CodeExecutionError } from '../lib/errors';
import {
  loadQuickJS,
  runInRealm,
  loadArqueroUmd,
  isSafeRealmIdentifier,
  tableToCsv,
  type SandboxWorkerRequest,
} from './realmExecutor';

/**
 * SECURITY: Execute LLM-generated JavaScript inside a QuickJS realm in a
 * dedicated Web Worker.
 *
 * ── Trust boundary ────────────────────────────────────────────────────────
 * The source of `code` is an LLM completion, NOT user-typed input. The LLM
 * has been system-prompted to emit only Arquero transformations, but the
 * underlying threat is prompt injection: a malicious document ingested via
 * the vectorstore (or a user-chosen CSV column header) could contain
 * instructions the LLM dutifully echoes into the generated code.
 *
 * ── Mitigations ───────────────────────────────────────────────────────────
 *  - Realm isolation by construction. Each execution runs in a FRESH QuickJS
 *    context with NO page globals: `fetch`, `localStorage`, `document`,
 *    `Function`, `eval` all resolve to undefined inside the realm. The
 *    escape-vector class that `new Function` could not close (its scope
 *    chain reaches the page's global scope) is eliminated rather than
 *    enumerated away. A fresh context per execution also means generated
 *    code cannot poison the realm's own `aq` namespace or leak globals into
 *    the next question's execution.
 *  - Static scan, defense-in-depth. The scan below rejects the known escape
 *    vectors BEFORE execution, so a rejected source fails fast with a
 *    retryable error fed back to the model instead of paying realm startup.
 *  - Off the UI thread. The realm runs in a dedicated Web Worker; the
 *    interpreter's ~4-5x cost on data-heavy code (measured: a 20k-row
 *    aggregate is 728ms interpreted vs 170ms JIT) never freezes the page.
 *  - Bounded execution. The realm's interrupt handler trips at
 *    EXECUTION_TIMEOUT_MS (an infinite loop in generated code is killed in
 *    milliseconds instead of hanging the tab forever) and the runtime memory
 *    limit bounds allocation.
 *  - Data crossing the boundary. Dataset tables cannot cross the realm
 *    boundary; the host serializes them with Arquero's own `toCSV` and the
 *    realm re-parses via `fromCSV`, so the tables the generated code sees
 *    are fresh per execution and the in-memory store cannot be poisoned.
 *    Dataset names are validated as safe realm identifiers before injection.
 *  - Worker death. The client listens for the worker's 'error' event,
 *    rejects pending executions, and respawns on the next call; a silent
 *    round-trip stall is rejected by the client-side timeout.
 *
 * ── Residual risk ─────────────────────────────────────────────────────────
 *  - The `toCSV`/`fromCSV` round-trip re-parses values (numbers, nulls,
 *    quoted strings); exotic column content could lose fidelity. The
 *    analyzer's normalized CSV parse path produces the tables, and the
 *    round-trip is verified by test.
 *  - A bug in the QuickJS interpreter or the in-realm codecs is contained by
 *    the per-execution context disposal.
 *  - The main-thread realm path runs inline when no Worker exists (the test
 *    harness); it is not used in production browsers.
 *
 * @throws CodeExecutionError for syntax errors, runtime errors, timeouts
 */

// (defense-in-depth) Static scan of the generated source for the known
// escape vectors. The QuickJS realm resolves bare references to undefined,
// so these cannot reach page globals any more — the scan still rejects them
// before execution to fail fast with a retryable error for the model.
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

const CLIENT_EXECUTION_TIMEOUT_MS = 15_000;

/**
 * Callable sandbox executor plus its teardown. `dispose` terminates the
 * worker once pending executions drain; a no-op when no worker was spawned.
 */
export type UserCodeExecutor = ((code: string) => Promise<unknown>) & {
  dispose: () => void;
};

export function createUserCodeExecutor(
  datasets: Map<string, unknown>,
  options?: { workerFactory?: () => Worker },
): UserCodeExecutor {
  const hasWorker = typeof Worker !== 'undefined';

  let worker: Worker | null = null;
  let doomed = false;
  let nextId = 1;
  const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

  function terminateIfDrained(): void {
    if (!doomed || pending.size > 0 || worker === null) return;
    const w = worker;
    worker = null;
    doomed = false;
    w.terminate();
  }

  function spawnWorker(): Worker {
    const w = options?.workerFactory
      ? options.workerFactory()
      : new Worker(new URL('../workers/analysisSandboxWorker.ts', import.meta.url), { type: 'module' });

    w.addEventListener('message', (ev: MessageEvent) => {
      const data: unknown = ev.data;
      if (typeof data !== 'object' || data === null) return;
      const msg = data as { type?: string; id?: number; rows?: unknown; kind?: string; message?: string };
      if (msg.type !== 'result' && msg.type !== 'error') return;
      const id = msg.id;
      if (typeof id !== 'number') return;
      const entry = pending.get(id);
      if (!entry) return;
      pending.delete(id);
      if (msg.type === 'result') {
        entry.resolve(msg.rows);
      } else {
        const message = msg.message ?? 'sandbox execution failed';
        entry.reject(new CodeExecutionError(message, new Error(message), {
          code: '',
          retryable: msg.kind === 'runtime',
        }));
      }
      terminateIfDrained();
    });

    // The worker 'error' event fires when the script fails to load — no
    // message is posted, so without this listener every pending execution
    // hangs. Respawning happens on the next execution.
    w.addEventListener('error', () => {
      const message = 'analysis sandbox worker failed to load';
      for (const entry of pending.values()) entry.reject(new Error(message));
      pending.clear();
      worker = null;
    });

    return w;
  }

  let mainThreadQuickJS: Awaited<ReturnType<typeof loadQuickJS>> | null = null;
  let mainThreadAqUmd: string | null = null;

  async function ensureMainThreadLoaded(): Promise<void> {
    if (mainThreadQuickJS === null) mainThreadQuickJS = await loadQuickJS();
    if (mainThreadAqUmd === null) mainThreadAqUmd = await loadArqueroUmd();
  }

  function collectTables(code: string): Array<{ name: string; csv: string }> {
    const tables: Array<{ name: string; csv: string }> = [];
    for (const [name, table] of datasets) {
      if (name === 'aq') continue;
      if (!isSafeRealmIdentifier(name)) {
        throw new CodeExecutionError(
          `dataset name "${name}" is not a safe realm identifier`,
          new Error('unsafe dataset name'),
          { code, retryable: false },
        );
      }
      tables.push({ name, csv: tableToCsv(table) });
    }
    return tables;
  }

  async function executeUserCode(code: string): Promise<unknown> {
    assertCodeSafe(code);

    if (!hasWorker) {
      // The main-thread realm path: environments without a Worker (the test
      // harness). The same realm logic runs inline.
      await ensureMainThreadLoaded();
      const { rows, context } = runInRealm(
        () => mainThreadQuickJS!.newContext(),
        mainThreadAqUmd!,
        code,
        collectTables(code),
      );
      context.dispose();
      return rows;
    }

    if (worker === null) worker = spawnWorker();
    const id = nextId++;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    const rows = await new Promise<unknown>((resolve, reject) => {
      pending.set(id, { resolve, reject });
      const request: SandboxWorkerRequest = { type: 'execute', id, code, tables: collectTables(code) };
      worker!.postMessage(request);
      timeoutId = setTimeout(() => {
        if (pending.delete(id)) {
          reject(new CodeExecutionError(
            'Code execution timed out',
            new Error('sandbox worker round-trip timed out'),
            { code, retryable: false },
          ));
          terminateIfDrained();
        }
      }, CLIENT_EXECUTION_TIMEOUT_MS);
    });
    if (timeoutId !== null) clearTimeout(timeoutId);
    return rows;
  }

  function dispose(): void {
    doomed = true;
    terminateIfDrained();
  }

  return Object.assign(executeUserCode, { dispose });
}
