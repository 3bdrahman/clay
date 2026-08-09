/**
 * Derive natural-language example queries from the user's actual data,
 * so quick-start prompts always reference what's currently loaded instead
 * of bundled sample-CSV column names.
 */

export interface DatasetSummary {
  name: string;
  fileName: string;
  columns: string[];
  rowCount: number;
}

export interface DocumentSummary {
  fileName: string;
}

const NUMERIC_PATTERN =
  /price|amount|revenue|salary|count|total|budget|qty|quantity|score|rating|units|sales|spent|cost|fees/i;
const CATEGORICAL_PATTERN =
  /category|type|status|department|region|product|country|state|group|owner|priority|severity/i;
const TEMPORAL_PATTERN =
  /date|time|day|month|year|created|updated|timestamp/i;

export function deriveDataQueries(datasets: DatasetSummary[]): string[] {
  if (datasets.length === 0) return [];
  const first = datasets[0];
  const cols = first?.columns ?? [];
  const numeric = cols.find((c) => NUMERIC_PATTERN.test(c));
  const categorical = cols.find((c) => CATEGORICAL_PATTERN.test(c));
  const out: string[] = [];
  if (numeric && categorical) out.push(`Average ${numeric} by ${categorical}`);
  else if (categorical) out.push(`Count by ${categorical}`);
  if (numeric) out.push(`Total ${numeric}`);
  if (datasets.length > 1) out.push('Join these datasets');
  if (out.length === 0) {
    out.push(`Summarize ${first?.name ?? 'the dataset'}`);
    out.push(`Show the first rows of ${first?.name ?? 'the dataset'}`);
  }
  return out.slice(0, 3);
}

export function deriveDocumentQueries(docs: DocumentSummary[]): string[] {
  if (docs.length === 0) return [];
  const names = docs.map((d) => d.fileName);
  const sample = names.slice(0, 3).join(', ');
  return [
    `Summarize ${names.length === 1 ? names[0] : `${names.length} uploaded documents`}`,
    `Find action items${names.length > 1 ? ` across ${sample}` : ''}`,
    `Key themes in ${names.length === 1 ? names[0] : 'the uploaded files'}`,
  ];
}

export { NUMERIC_PATTERN, CATEGORICAL_PATTERN, TEMPORAL_PATTERN };
