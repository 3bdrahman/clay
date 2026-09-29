import type { ClayServices } from '../hooks/useClay';
import { processFile, embedDocumentChunks, existingSourceHashes, type ProcessedFile } from '../services/files';
import {
  registerSandboxTable,
  persistSandboxCsv,
} from '../services/sandboxTables';
import { loadSampleDatasets } from '../services/datasets';
import { useAppStore, type SandboxDataset, type SandboxProcessing } from '../store';

export interface SandboxIngestDeps {
  services: ClayServices | null;
  setSandboxProcessing: (items: SandboxProcessing[]) => void;
  updateSandboxProcessingItem: (fileName: string, update: Partial<SandboxProcessing>) => void;
  addSandboxDataset: (dataset: SandboxDataset) => void;
  addSandboxDocument: (document: { id: string; fileName: string; source: string; chunkCount: number; loadedAt: number }) => void;
  getSandboxProcessing: () => SandboxProcessing[];
  getSandboxDatasets: () => SandboxDataset[];
  setSandboxDatasets: (datasets: SandboxDataset[]) => void;
}

export async function addFiles(
  files: FileList | File[],
  deps: SandboxIngestDeps,
): Promise<void> {
  const arr = Array.from(files);
  if (arr.length === 0) return;
  const sv = deps.services;
  if (!sv) {
    throw new Error('Services not ready — add an API key first');
  }

  const initial: SandboxProcessing[] = arr.map(f => ({
    fileName: f.name,
    status: 'processing',
  }));
  deps.setSandboxProcessing([...deps.getSandboxProcessing(), ...initial]);

  for (const file of arr) {
    try {
      deps.updateSandboxProcessingItem(file.name, { status: 'processing' });
      const processed: ProcessedFile = await processFile(file);
      if (processed.error) {
        deps.updateSandboxProcessingItem(file.name, { status: 'error', error: processed.error });
        continue;
      }
      if (processed.dataset) {
        const csv = await file.text();
        registerSandboxTable(processed.dataset.name, processed.dataset.table);
        deps.addSandboxDataset({
          name: processed.dataset.name,
          fileName: file.name,
          columns: processed.dataset.columns,
          rowCount: processed.dataset.rowCount,
          loadedAt: Date.now(),
          csv,
          isSample: false,
        });
        await persistSandboxCsv(processed.dataset.name, csv);
        deps.updateSandboxProcessingItem(file.name, { status: 'done' });
        continue;
      }
      if (processed.document) {
        deps.updateSandboxProcessingItem(file.name, { status: 'embedding' });
        const hashes = await existingSourceHashes(sv.vectorstore, processed.document.source);
        if (hashes.has(processed.document.sourceHash)) {
          deps.updateSandboxProcessingItem(file.name, { status: 'done' });
          deps.addSandboxDocument({
            id: processed.document.source,
            fileName: file.name,
            source: processed.document.source,
            chunkCount: processed.document.chunks.length,
            loadedAt: Date.now(),
          });
          continue;
        }
        const embedded = await embedDocumentChunks(processed.document, sv.embeddings);
        sv.vectorstore.removeBySource(processed.document.source);
        sv.vectorstore.addEntries(embedded.map(e => ({
          id: e.id,
          text: e.text,
          source: e.source,
          page: e.page,
          embedding: e.embedding,
        })));
        deps.addSandboxDocument({
          id: processed.document.source,
          fileName: file.name,
          source: processed.document.source,
          chunkCount: processed.document.chunks.length,
          loadedAt: Date.now(),
        });
        deps.updateSandboxProcessingItem(file.name, { status: 'done' });
        continue;
      }
      deps.updateSandboxProcessingItem(file.name, { status: 'error', error: 'No content extracted' });
    } catch (e) {
      deps.updateSandboxProcessingItem(file.name, {
        status: 'error',
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  setTimeout(() => {
    useAppStore.setState(state => ({
      sandboxProcessing: state.sandboxProcessing.filter(
        p => p.status === 'processing' || p.status === 'embedding',
      ),
    }));
  }, 3000);
}

export async function loadSampleData(deps: SandboxIngestDeps): Promise<void> {
  const sample = await loadSampleDatasets();
  sample.tables.forEach((table, name) => {
    registerSandboxTable(name, table);
  });
  const newDatasets: SandboxDataset[] = [];
  sample.tables.forEach((table, name) => {
    const columns = table.columnNames();
    const rowCount = table.numRows();
    const originalCsv = sample.rawCsv[name];
    newDatasets.push({
      name,
      fileName: `sample/${name}.csv`,
      columns,
      rowCount,
      loadedAt: Date.now(),
      csv: originalCsv,
      isSample: true,
    });
  });
  deps.setSandboxDatasets([
    ...deps.getSandboxDatasets().filter(d => !newDatasets.some(n => n.name === d.name)),
    ...newDatasets,
  ]);
  await Promise.all(newDatasets.flatMap(d => (d.csv !== undefined ? [persistSandboxCsv(d.name, d.csv)] : [])));
}