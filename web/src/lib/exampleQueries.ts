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
  /** Optional sample rows for intelligent type detection (first 5 rows) */
  sampleRows?: Array<Record<string, unknown>>;
}

export interface DocumentSummary {
  fileName: string;
}

/**
 * Column type inferred from name patterns and/or value sampling.
 */
export type ColumnType = 'numeric' | 'categorical' | 'temporal' | 'unknown';

export interface ColumnInfo {
  name: string;
  type: ColumnType;
  confidence: number; // 0-1
  sampleValues?: unknown[];
}

/**
 * Confidence thresholds for column type detection.
 * These values are empirically chosen to balance precision/recall:
 * - HIGH (0.9): Strong explicit patterns (e.g., "price", "revenue")
 * - MEDIUM (0.7-0.85): Weaker patterns or partial value evidence
 * - LOW (0.6): Default fallback for string data
 * - THRESHOLD_0_5 (0.5): Minimum confidence to prefer value-based over name-based detection
 * - THRESHOLD_0_8 (0.8): Fraction of numeric/date-like values needed for high confidence
 * - THRESHOLD_0_7 (0.7): Fraction of date-like values for medium confidence
 * - CARDINALITY_MAX (20): Maximum unique values to consider categorical
 * - CARDINALITY_RATIO (0.5): Maximum ratio of unique/total values for categorical
 */
const CONFIDENCE = {
  HIGH: 0.9,
  MEDIUM_HIGH: 0.85,
  MEDIUM: 0.8,
  MEDIUM_LOW: 0.75,
  LOW: 0.7,
  FALLBACK: 0.6,
  ZERO: 0,
  THRESHOLD_0_5: 0.5,
  THRESHOLD_0_8: 0.8,
  THRESHOLD_0_7: 0.7,
  CARDINALITY_MAX: 20,
  CARDINALITY_RATIO: 0.5,
} as const;

/**
 * Detect column type from name patterns (fallback when no samples available).
 * Returns confidence based on pattern specificity.
 */
function detectTypeFromName(columnName: string): { type: ColumnType; confidence: number } {
  const lower = columnName.toLowerCase();

  // Numeric patterns - high confidence for explicit numeric terms
  if (/\b(price|amount|revenue|salary|count|total|budget|qty|quantity|score|rating|units|sales|spent|cost|fees|sum|avg|mean|min|max|balance|income|expense|profit|margin)\b/i.test(lower)) {
    return { type: 'numeric', confidence: CONFIDENCE.HIGH };
  }
  if (/\b(id|number|num|no|quantity|amount|value|rate|percentage|percent|ratio|index)\b/i.test(lower)) {
    return { type: 'numeric', confidence: CONFIDENCE.LOW };
  }

  // Temporal patterns
  if (/\b(date|time|day|month|year|created|updated|timestamp|datetime|year|quarter)\b/i.test(lower)) {
    return { type: 'temporal', confidence: CONFIDENCE.MEDIUM_HIGH };
  }
  if (/\b(at|on|since|until|from|to)\b/i.test(lower) && /\b(date|time)\b/i.test(lower)) {
    return { type: 'temporal', confidence: CONFIDENCE.LOW };
  }

  // Categorical patterns
  if (/\b(category|type|status|department|region|product|country|state|group|owner|priority|severity|name|label|tag|class|segment|tier)\b/i.test(lower)) {
    return { type: 'categorical', confidence: CONFIDENCE.MEDIUM_HIGH };
  }
  if (/\b(code|key|flag|indicator|level|grade|rank)\b/i.test(lower)) {
    return { type: 'categorical', confidence: CONFIDENCE.FALLBACK };
  }

  return { type: 'unknown', confidence: CONFIDENCE.ZERO };
}

/**
 * Detect column type from sample values (more accurate).
 * Analyzes first N non-null values to determine type.
 */
function detectTypeFromValues(values: unknown[]): { type: ColumnType; confidence: number } {
  const nonNull = values.filter(v => v !== null && v !== undefined && v !== '');
  if (nonNull.length === 0) return { type: 'unknown', confidence: CONFIDENCE.ZERO };

  // Check if all values are numeric
  const numericCount = nonNull.filter(v => typeof v === 'number' && Number.isFinite(v)).length;
  if (numericCount === nonNull.length) {
    return { type: 'numeric', confidence: 0.95 };
  }
  if (numericCount / nonNull.length > CONFIDENCE.THRESHOLD_0_8) {
    return { type: 'numeric', confidence: CONFIDENCE.MEDIUM };
  }

  // Check for date-like strings
  const dateLikeCount = nonNull.filter(v => {
    if (typeof v !== 'string') return false;
    // ISO date, US date, or timestamp patterns
    return /^\d{4}-\d{2}-\d{2}/.test(v) || 
           /^\d{2}\/\d{2}\/\d{4}/.test(v) ||
           /^\d{13}$/.test(v) || // millisecond timestamp
           /^\d{10}$/.test(v);   // second timestamp
  }).length;
  if (dateLikeCount === nonNull.length && nonNull.length > 0) {
    return { type: 'temporal', confidence: 0.9 };
  }
  if (dateLikeCount / nonNull.length > CONFIDENCE.THRESHOLD_0_7) {
    return { type: 'temporal', confidence: CONFIDENCE.MEDIUM_LOW };
  }

  // Check for low cardinality (categorical)
  const uniqueValues = new Set(nonNull.map(v => String(v)));
  if (uniqueValues.size <= Math.min(CONFIDENCE.CARDINALITY_MAX, nonNull.length * CONFIDENCE.CARDINALITY_RATIO) && uniqueValues.size > 1) {
    return { type: 'categorical', confidence: CONFIDENCE.MEDIUM };
  }

  // Default to categorical for string data with moderate cardinality
  if (nonNull.every(v => typeof v === 'string')) {
    return { type: 'categorical', confidence: CONFIDENCE.FALLBACK };
  }

  return { type: 'unknown', confidence: CONFIDENCE.ZERO };
}

/**
 * Infer column types for a dataset using name patterns and/or value sampling.
 * Prioritizes value-based detection when samples are available.
 */
export function inferColumnTypes(dataset: DatasetSummary): ColumnInfo[] {
  const cols = dataset.columns ?? [];
  const sampleRows = dataset.sampleRows ?? [];

  return cols.map(colName => {
    // Try value-based detection first if samples available
    if (sampleRows.length > 0) {
      const values = sampleRows.map(row => row[colName]);
      const valueResult = detectTypeFromValues(values);
      if (valueResult.confidence > 0.5) {
        return {
          name: colName,
          type: valueResult.type,
          confidence: valueResult.confidence,
          sampleValues: values.slice(0, 5),
        };
      }
    }

    // Fall back to name-based detection
    const nameResult = detectTypeFromName(colName);
    return {
      name: colName,
      type: nameResult.type,
      confidence: nameResult.confidence,
    };
  });
}

/**
 * Get columns of a specific type from a dataset.
 */
export function getColumnsByType(dataset: DatasetSummary, type: ColumnType): ColumnInfo[] {
  return inferColumnTypes(dataset).filter(c => c.type === type);
}

/**
 * Get the best column for a given type (highest confidence).
 */
export function getBestColumn(dataset: DatasetSummary, type: ColumnType): ColumnInfo | undefined {
  const cols = getColumnsByType(dataset, type);
  if (cols.length === 0) return undefined;
  return cols.reduce((best, current) => current.confidence > best.confidence ? current : best);
}

const DEFAULT_NUMERIC_PATTERN =
  /price|amount|revenue|salary|count|total|budget|qty|quantity|score|rating|units|sales|spent|cost|fees/i;
const DEFAULT_CATEGORICAL_PATTERN =
  /category|type|status|department|region|product|country|state|group|owner|priority|severity/i;
const DEFAULT_TEMPORAL_PATTERN =
  /date|time|day|month|year|created|updated|timestamp/i;

export function deriveDataQueries(datasets: DatasetSummary[]): string[] {
  if (datasets.length === 0) return [];
  const first = datasets[0];
  const cols = first?.columns ?? [];

  // Use intelligent type detection
  const numericCol = getBestColumn(first, 'numeric')?.name ?? 
    cols.find(c => DEFAULT_NUMERIC_PATTERN.test(c));
  const categoricalCol = getBestColumn(first, 'categorical')?.name ?? 
    cols.find(c => DEFAULT_CATEGORICAL_PATTERN.test(c));
  const temporalCol = getBestColumn(first, 'temporal')?.name ?? 
    cols.find(c => DEFAULT_TEMPORAL_PATTERN.test(c));

  const out: string[] = [];
  if (numericCol && categoricalCol) out.push(`Average ${numericCol} by ${categoricalCol}`);
  else if (categoricalCol) out.push(`Count by ${categoricalCol}`);
  if (numericCol) out.push(`Total ${numericCol}`);
  if (temporalCol && numericCol) out.push(`Trend of ${numericCol} over ${temporalCol}`);
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

export function deriveWebQueries(): string[] {
  // Dynamic web search questions - not tied to any specific dataset
  // These are general knowledge questions suitable for any user
  return [
    'What are the latest trends in AI for business?',
    'Compare cloud providers for ML workloads',
    'What are the best practices for vector search?',
  ];
}

// Re-export pattern constants for backward compatibility
export { DEFAULT_NUMERIC_PATTERN, DEFAULT_CATEGORICAL_PATTERN, DEFAULT_TEMPORAL_PATTERN };