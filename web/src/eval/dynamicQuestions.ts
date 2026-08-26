// Dynamic evaluation question generator
// Generates relevant evaluation questions based on actual loaded datasets and documents

import type { DatasetSummary, DocumentSummary } from '../lib/exampleQueries';
import type { EvalQuestion } from './runner';

/**
 * Generate data analysis questions based on actual dataset schemas.
 * Questions are derived from column names and types, not hardcoded assumptions.
 */
export function generateDataAnalysisQuestions(datasets: DatasetSummary[]): EvalQuestion[] {
  if (datasets.length === 0) return [];

  const questions: EvalQuestion[] = [];
  let questionIndex = 1;

  for (const ds of datasets) {
    const cols = ds.columns ?? [];
    if (cols.length === 0) continue;

    // Find numeric columns
    const numericCols = cols.filter(c => /\b(price|amount|revenue|salary|count|total|budget|qty|quantity|score|rating|units|sales|spent|cost|fees|sum|avg|mean|min|max)\b/i.test(c));
    const categoricalCols = cols.filter(c => /\b(category|type|status|department|region|product|country|state|group|owner|priority|severity|name|id)\b/i.test(c));
    const temporalCols = cols.filter(c => /\b(date|time|day|month|year|created|updated|timestamp)\b/i.test(c));

    // Generic questions that work with any dataset
    questions.push({
      id: `data-${String(questionIndex++).padStart(3, '0')}`,
      question: `How many rows are in ${ds.name}?`,
      category: 'data_analysis',
      expectedSource: 'python',
      expectedColumnIntent: ['count(*)', 'dataset:' + ds.name],
      goldenAnswer: `The query should count rows in ${ds.name}.`,
      minRelevantChunks: 0,
    });

    questions.push({
      id: `data-${String(questionIndex++).padStart(3, '0')}`,
      question: `What are the column names in ${ds.name}?`,
      category: 'data_analysis',
      expectedSource: 'python',
      expectedColumnIntent: ['columns', 'dataset:' + ds.name],
      goldenAnswer: `The query should list columns: ${cols.join(', ')}.`,
      minRelevantChunks: 0,
    });

    if (numericCols.length > 0) {
      const col = numericCols[0];
      questions.push({
        id: `data-${String(questionIndex++).padStart(3, '0')}`,
        question: `What is the average of ${col} in ${ds.name}?`,
        category: 'data_analysis',
        expectedSource: 'python',
        expectedColumnIntent: ['avg:numeric', 'column:' + col, 'dataset:' + ds.name],
        goldenAnswer: `The query should compute the average of ${col} from ${ds.name}.`,
        minRelevantChunks: 0,
      });

      questions.push({
        id: `data-${String(questionIndex++).padStart(3, '0')}`,
        question: `What is the total of ${col} in ${ds.name}?`,
        category: 'data_analysis',
        expectedSource: 'python',
        expectedColumnIntent: ['sum:numeric', 'column:' + col, 'dataset:' + ds.name],
        goldenAnswer: `The query should sum the ${col} column from ${ds.name}.`,
        minRelevantChunks: 0,
      });
    }

    if (categoricalCols.length > 0) {
      const col = categoricalCols[0];
      questions.push({
        id: `data-${String(questionIndex++).padStart(3, '0')}`,
        question: `Show the distribution of ${col} in ${ds.name}`,
        category: 'data_analysis',
        expectedSource: 'python',
        expectedColumnIntent: ['group-by:categorical', 'count', 'column:' + col, 'dataset:' + ds.name],
        goldenAnswer: `The query should group by ${col} and count occurrences.`,
        minRelevantChunks: 0,
      });
    }

    if (numericCols.length > 0 && categoricalCols.length > 0) {
      const numCol = numericCols[0];
      const catCol = categoricalCols[0];
      questions.push({
        id: `data-${String(questionIndex++).padStart(3, '0')}`,
        question: `Average ${numCol} by ${catCol} in ${ds.name}`,
        category: 'data_analysis',
        expectedSource: 'python',
        expectedColumnIntent: ['avg:numeric', 'group-by:categorical', 'column:' + numCol, 'column:' + catCol, 'dataset:' + ds.name],
        goldenAnswer: `The query should compute mean of ${numCol} per ${catCol} group.`,
        minRelevantChunks: 0,
      });
    }

    if (temporalCols.length > 0 && numericCols.length > 0) {
      const tempCol = temporalCols[0];
      const numCol = numericCols[0];
      questions.push({
        id: `data-${String(questionIndex++).padStart(3, '0')}`,
        question: `Aggregate ${numCol} over time using ${tempCol} in ${ds.name}`,
        category: 'data_analysis',
        expectedSource: 'python',
        expectedColumnIntent: ['group-by:temporal', 'sum:numeric', 'column:' + numCol, 'column:' + tempCol, 'dataset:' + ds.name],
        goldenAnswer: `The query should group by time period and sum ${numCol}.`,
        minRelevantChunks: 0,
      });
    }

    // Limit questions per dataset to keep eval manageable
    if (questions.length > 10) break;
  }

  // If no dataset-specific questions generated, add generic ones
  if (questions.length === 0 && datasets.length > 0) {
    const first = datasets[0];
    questions.push({
      id: `data-${String(questionIndex++).padStart(3, '0')}`,
      question: `Summarize ${first.name}`,
      category: 'data_analysis',
      expectedSource: 'python',
      expectedColumnIntent: ['summarize', 'dataset:' + first.name],
      goldenAnswer: `The query should provide a summary of ${first.name}.`,
      minRelevantChunks: 0,
    });
  }

  return questions.slice(0, 12); // Cap at 12 data questions
}

/**
 * Generate document analysis questions based on actual uploaded documents.
 */
export function generateDocumentQuestions(docs: DocumentSummary[]): EvalQuestion[] {
  if (docs.length === 0) return [];

  const questions: EvalQuestion[] = [];
  const names = docs.map(d => d.fileName);
  let questionIndex = 1;

  questions.push({
    id: `docs-${String(questionIndex++).padStart(3, '0')}`,
    question: `Summarize ${names.length === 1 ? names[0] : `${names.length} uploaded documents`}`,
    category: 'documents',
    expectedSource: 'vectorstore',
    goldenAnswer: `Answer should provide a summary based on the uploaded document${names.length > 1 ? 's' : ''}.`,
    minRelevantChunks: 1,
  });

  questions.push({
    id: `docs-${String(questionIndex++).padStart(3, '0')}`,
    question: `Find action items${names.length > 1 ? ` across ${names.slice(0, 3).join(', ')}` : ''}`,
    category: 'documents',
    expectedSource: 'vectorstore',
    goldenAnswer: `Answer should identify action items from the document${names.length > 1 ? 's' : ''}.`,
    minRelevantChunks: 1,
  });

  questions.push({
    id: `docs-${String(questionIndex++).padStart(3, '0')}`,
    question: `Key themes in ${names.length === 1 ? names[0] : 'the uploaded files'}`,
    category: 'documents',
    expectedSource: 'vectorstore',
    goldenAnswer: `Answer should identify key themes from the document${names.length > 1 ? 's' : ''}.`,
    minRelevantChunks: 1,
  });

  // Add document-specific questions
  for (const doc of docs.slice(0, 3)) {
    questions.push({
      id: `docs-${String(questionIndex++).padStart(3, '0')}`,
      question: `What does ${doc.fileName} say about key topics?`,
      category: 'documents',
      expectedSource: 'vectorstore',
      goldenAnswer: `Answer should cite content from ${doc.fileName}.`,
      minRelevantChunks: 1,
    });
  }

  return questions.slice(0, 8); // Cap at 8 document questions
}

/**
 * Generate web search questions - general knowledge questions.
 */
export function generateWebSearchQuestions(): EvalQuestion[] {
  return [
    {
      id: 'web-001',
      question: 'What are the latest trends in AI for business?',
      category: 'web_search',
      expectedSource: 'websearch',
      goldenAnswer: 'Answer should cite web search results about AI trends.',
      minRelevantChunks: 0,
    },
    {
      id: 'web-002',
      question: 'Compare cloud providers for machine learning workloads',
      category: 'web_search',
      expectedSource: 'websearch',
      goldenAnswer: 'Answer should cite web search results comparing cloud providers.',
      minRelevantChunks: 0,
    },
    {
      id: 'web-003',
      question: 'What are the best practices for vector search?',
      category: 'web_search',
      expectedSource: 'websearch',
      goldenAnswer: 'Answer should cite web search results about vector search best practices.',
      minRelevantChunks: 0,
    },
  ];
}

/**
 * Generate a complete evaluation question set based on actual loaded data.
 * This replaces the static questions.json with dynamic, data-aware questions.
 */
export function generateEvalQuestions(
  datasets: DatasetSummary[],
  documents: DocumentSummary[]
): EvalQuestion[] {
  const dataQuestions = generateDataAnalysisQuestions(datasets);
  const docQuestions = generateDocumentQuestions(documents);
  const webQuestions = generateWebSearchQuestions();

  return [...dataQuestions, ...docQuestions, ...webQuestions];
}