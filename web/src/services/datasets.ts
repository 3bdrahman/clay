import * as aq from 'arquero';
import type { ColumnTable } from 'arquero';
import type { DatasetMeta } from '../services/analyzer';
import { isDatasetExtension } from '../lib/fileExtensions';
import { RagError, RagErrorCode } from '../lib/errors';
import { parseCsvNormalized } from './csvNormalize';

export interface SampleLoadResult {
  /** Dataset tables keyed by dataset name — real Arquero ColumnTables. */
  tables: Map<string, ColumnTable>;
  metadata: DatasetMeta;
  rawCsv: Record<string, string>;
  /** Arquero module namespace, exposed to LLM-generated code as the 'aq' variable. */
  arquero: typeof aq;
}

/**
 * Typed error raised when one or more per-file fetches/parse steps fail inside
 * `loadSampleDatasets`. Replaces the prior silent `catch e => console.warn`
 * partial-load behavior — see GH issue #16. The caller surfaces this to the
 * user via the existing error toast in DataSandbox.
 */
export class SampleDatasetLoadError extends RagError {
  public readonly failedFiles: ReadonlyArray<string>;
  public readonly succeededFiles: ReadonlyArray<string>;

  constructor(
    failed: ReadonlyArray<{ name: string; cause: Error }>,
    succeeded: ReadonlyArray<string>,
  ) {
    const failedNames = failed.map(f => f.name);
    const detail = failed
      .map(f => `${f.name} (${f.cause.message || f.cause.name || 'unknown error'})`)
      .join(', ');
    const all = succeeded.length === 0;
    const message = all
      ? `Sample data could not be loaded. Failed: ${detail}.`
      : `Sample data partially loaded. ${succeeded.length} OK, ${failed.length} failed: ${detail}. The failed datasets will be missing from the sandbox.`;

    super({
      code: RagErrorCode.UNKNOWN_ERROR,
      message,
      retryable: true,
      context: {
        failedCount: failed.length,
        succeededCount: succeeded.length,
        failedNames,
        succeededNames: succeeded,
      },
    });
    this.name = 'SampleDatasetLoadError';
    this.failedFiles = failedNames;
    this.succeededFiles = succeeded;
  }
}

// Get the base path for correct asset loading on GitHub Pages
const getBasePath = (): string => {
  const base = import.meta.env.BASE_URL || '/';
  return base.endsWith('/') ? base : base + '/';
};

export async function loadSampleDatasets(): Promise<SampleLoadResult> {
  const tables = new Map<string, ColumnTable>();
  const metadata: DatasetMeta = {};
  const rawCsv: Record<string, string> = {};

  const basePath = getBasePath();
  const indexResp = await fetch(`${basePath}data/datasets/index.json`, { cache: 'no-store' });
  if (!indexResp.ok) {
    throw new Error(
      `Sample dataset index unavailable (HTTP ${indexResp.status} ${indexResp.statusText}). ` +
        `Try uploading your own CSV instead.`,
    );
  }
  const idx = (await indexResp.json()) as { files?: string[] };
  const names = (idx.files || []).filter(n => isDatasetExtension(n));
  if (names.length === 0) {
    throw new Error('Sample dataset index is empty (no supported dataset files listed).');
  }

  const failures: Array<{ name: string; cause: Error }> = [];
  const successes: string[] = [];

  await Promise.all(
    names.map(async fileName => {
      const name = fileName.replace(/\.csv$/i, '');
      try {
        const resp = await fetch(`${basePath}data/datasets/${fileName}`, { cache: 'no-store' });
        if (!resp.ok) {
          throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
        }
        const text = await resp.text();
        const table = parseCsvNormalized(text);
        const columns = table.columnNames();
        const rowCount = table.numRows();
        tables.set(name, table);
        metadata[name] = { columns, rowCount };
        rawCsv[name] = text;
        successes.push(name);
      } catch (e) {
        const cause = e instanceof Error ? e : new Error(String(e));
        failures.push({ name, cause });
        if (import.meta.env.DEV) console.warn(`[clay] failed to load dataset ${name}:`, cause);
      }
    })
  );

  if (failures.length > 0) {
    throw new SampleDatasetLoadError(failures, successes);
  }

  return { tables, metadata, rawCsv, arquero: aq };
}

