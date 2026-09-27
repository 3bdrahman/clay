// analysisSandboxWorker — runs LLM-generated analysis code inside a QuickJS
// realm, off the UI thread. A fresh context per execution keeps generated
// code from poisoning the realm for the next question; the interrupt handler
// bounds wall-clock time and the runtime memory limit bounds allocation.
import { loadQuickJS, loadArqueroUmd, runInRealm } from '../services/realmExecutor';
import type { SandboxWorkerRequest, SandboxWorkerResponse } from '../services/realmExecutor';

let quickJS: Awaited<ReturnType<typeof loadQuickJS>> | null = null;
let arqueroUmd: string | null = null;

async function ensureLoaded(): Promise<void> {
  if (quickJS === null) quickJS = await loadQuickJS();
  if (arqueroUmd === null) arqueroUmd = await loadArqueroUmd();
}

self.onmessage = async (ev: MessageEvent<SandboxWorkerRequest>) => {
  const req = ev.data;
  if (req?.type !== 'execute') return;
  try {
    await ensureLoaded();
    const { rows, context } = runInRealm(() => quickJS!.newContext(), arqueroUmd!, req.code, req.tables);
    context.dispose();
    const resp: SandboxWorkerResponse = { type: 'result', id: req.id, rows };
    self.postMessage(resp);
  } catch (e) {
    const err = e instanceof Error ? e : new Error(String(e));
    const resp: SandboxWorkerResponse = {
      type: 'error',
      id: req.id,
      kind: err.message.includes('Syntax error')
        ? 'syntax'
        : err.message.includes('timed out')
        ? 'timeout'
        : 'runtime',
      message: err.message,
    };
    self.postMessage(resp);
  }
};
