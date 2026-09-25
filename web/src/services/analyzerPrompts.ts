/**
 * Analyzer prompt builders — extracted from analyzer.ts for modularity.
 * These functions build the prompts used by the data analyzer's LLM interactions.
 */

import type { DatasetMeta } from './analyzer';

export interface BuildPromptParams {
  question: string;
  relevant: string[];
  metadata: DatasetMeta;
}

export interface BuildRetryPromptParams {
  question: string;
  lastError: string | null;
  datasetNames: string[];
}

export interface BuildSystemPromptParams {
  // No params needed currently, but keeping the pattern for consistency
}

export interface BuildInitialUserMessageParams {
  question: string;
  relevant: string[];
  metadata: DatasetMeta;
  previousContext?: string;
}

/**
 * Build the initial prompt for single-shot analysis.
 * Lists relevant datasets with their columns and row counts.
 */
export function buildPrompt({ question, relevant, metadata }: BuildPromptParams): string {
  const datasetInfo = relevant
    .map(name => {
      const meta = metadata[name];
      return `- ${name} (${meta?.rowCount || '?'} rows): columns = ${JSON.stringify(meta?.columns || [])}`;
    })
    .join('\n');
  return `Question: ${question}

Datasets available as Arquero tables (variable name = dataset name):
${datasetInfo}

Generate JavaScript using the Arquero library (available as 'aq'). Datasets are loaded as variables named after their dataset name and are real Arquero tables.

Example patterns:
- Filter: employees.filter(d => d.department === 'Engineering')
- Group count: employees.groupby('department').count()
- Sort top N: projects.orderby('budget_usd', 'desc').limit(5)
- Aggregate: projects.groupby('status').rollup({ total: d => op.sum(d.budget_usd) })
- Join: feedback.join(projects, ['project_id'])

Return JSON with literal text inside the code block (no outer braces):
{"code": "<your JavaScript code, ending with result = ...>", "explanation": "<brief explanation of the analysis>"}`;
}

/**
 * Build the retry prompt for single-shot analysis when code execution fails.
 * Lists available dataset names (excluding 'aq' namespace).
 * 
 * @param lastError - The error message from the failed attempt, or null if unknown
 */
export function buildRetryPrompt({ question, lastError, datasetNames }: BuildRetryPromptParams): string {
  const errorText = lastError
    ? `The previous code failed with: ${lastError}`
    : 'The previous code failed.';
  return `${errorText}

Question: ${question}

Generate FIXED JavaScript code using Arquero (loaded as 'aq'). Available datasets: ${datasetNames.join(', ')}.

Common pitfalls to avoid:
- Don't use pandas syntax (no .iloc, no pd, no Python f-strings)
- Use Arquero verbs: .filter(), .groupby(), .count(), .orderby(), .limit(), .rollup(), .join()
- Access columns as d.columnName or d['column name with spaces']
- Store result in a variable named result
- Return small JSON-safe results (objects, arrays of objects, or primitives)

Return JSON: {"code": "...", "explanation": "..."}`;
}

/**
 * Build the system prompt for the agentic tool loop.
 * Instructs the model to use tools to inspect data before reasoning.
 */
export function buildSystemPrompt(): string {
  return `You are a senior data analyst. You have tools to inspect the actual data — begin EVERY analysis by calling list_datasets, then profile_column on the relevant columns BEFORE reasoning. Quantify every claim with numbers from tool results. Verify patterns against the actual data (use filter_sample to inspect rows). Compute correlations where meaningful. State limitations and confidence honestly — never overclaim causation from correlation. When the evidence is sufficient, STOP calling tools and return the final synthesis as JSON with this exact structure:
{
  "answer": "your final answer text",
  "insights": [
    {"finding": "...", "evidence": "...", "confidence": "high|medium|low", "implication": "..."}
  ],
  "chart": {"type": "bar|line|pie", "title": "...", "xKey": "...", "yKeys": [...], "data": [...]}
}

The chart field is optional. Include it only when a visualization adds value.

CRITICAL: Your FIRST response MUST begin with "PLAN:" followed by one line describing your analysis strategy (what you will inspect and in what order). EVERY SUBSEQUENT response MUST begin with "REFLECTION:" followed by one line on what the results showed and what you will do next. This is required for every iteration.`;
}

/**
 * Build the initial user message for the tool loop.
 * Includes dataset info and optional previous analysis context.
 */
export function buildInitialUserMessage({ question, relevant, metadata, previousContext }: BuildInitialUserMessageParams): string {
  const datasetInfo = relevant
    .map(name => {
      const meta = metadata[name];
      return `- ${name} (${meta?.rowCount || '?'} rows): columns = ${JSON.stringify(meta?.columns || [])}`;
    })
    .join('\n');
  let message = `Question: ${question}

Available datasets:
${datasetInfo}

Use the tools to explore the data and answer the question.`;
  if (previousContext) {
    message = `Previous analysis context (from the last data question):\n${previousContext}\n\n---\n\n${message}`;
  }
  return message;
}