import * as pdfjs from 'pdfjs-dist';
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = pdfjsWorker;

export interface ExtractedPage {
  pageNumber: number;
  text: string;
  heading?: string;
}

interface PdfTextItem {
  str: string;
  transform: number[];
  height: number;
  width: number;
}

function isTextItem(item: unknown): item is PdfTextItem {
  if (item === null || typeof item !== 'object') return false;
  const candidate = item as Record<string, unknown>;
  return (
    typeof candidate.str === 'string' &&
    Array.isArray(candidate.transform) &&
    candidate.transform.every(v => typeof v === 'number') &&
    typeof candidate.height === 'number' &&
    typeof candidate.width === 'number'
  );
}

function gatherTextItems(items: unknown[]): PdfTextItem[] {
  const result: PdfTextItem[] = [];
  for (const item of items) {
    if (isTextItem(item)) result.push(item);
  }
  return result;
}

function mode(values: number[]): number {
  if (values.length === 0) return 0;
  const counts = new Map<number, number>();
  let bestValue = values[0];
  let bestCount = 0;
  for (const v of values) {
    const next = (counts.get(v) ?? 0) + 1;
    counts.set(v, next);
    if (next > bestCount) {
      bestCount = next;
      bestValue = v;
    }
  }
  return bestValue;
}

function markHeading(str: string, height: number, modalHeight: number): string {
  if (modalHeight <= 0 || height < modalHeight * 1.5) return str;
  return height >= modalHeight * 3 ? `# ${str}` : `## ${str}`;
}

const HEADING_MARKER = /^#{1,2}\s+/;

function renderLine(items: PdfTextItem[], modalHeight: number): { text: string; firstHeading: string | null } {
  const sorted = [...items].sort((a, b) => a.transform[4] - b.transform[4]);
  let firstHeading: string | null = null;
  const parts = sorted.map(item => {
    const marked = markHeading(item.str, item.height, modalHeight);
    if (firstHeading === null && marked !== item.str) {
      firstHeading = marked.replace(HEADING_MARKER, '');
    }
    return marked;
  });
  const text = parts.join(' ').replace(/\s+/g, ' ').trim();
  return { text, firstHeading };
}

function renderPage(items: PdfTextItem[]): { text: string; heading?: string } {
  const modalHeight = mode(items.map(i => i.height));
  const lines = new Map<number, PdfTextItem[]>();
  for (const item of items) {
    const y = Math.round(item.transform[5]);
    const bucket = lines.get(y);
    if (bucket) bucket.push(item);
    else lines.set(y, [item]);
  }
  const sortedYs = [...lines.keys()].sort((a, b) => b - a);
  let pageHeading: string | null = null;
  const renderedLines: string[] = [];
  for (const y of sortedYs) {
    const lineItems = lines.get(y);
    if (!lineItems) continue;
    const { text, firstHeading } = renderLine(lineItems, modalHeight);
    if (firstHeading !== null && pageHeading === null) pageHeading = firstHeading;
    if (text.length > 0) renderedLines.push(text);
  }
  return { text: renderedLines.join('\n'), heading: pageHeading ?? undefined };
}

export async function extractPdfText(buffer: ArrayBuffer): Promise<ExtractedPage[]> {
  const loadingTask = pdfjs.getDocument({ data: buffer });
  const pdf = await loadingTask.promise;
  const pages: ExtractedPage[] = [];

  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const textItems = gatherTextItems(content.items);
    if (textItems.length === 0) continue;
    const { text, heading } = renderPage(textItems);
    if (text.trim().length > 0) {
      pages.push({
        pageNumber: i,
        text,
        ...(heading !== undefined ? { heading } : {}),
      });
    }
  }

  return pages;
}
