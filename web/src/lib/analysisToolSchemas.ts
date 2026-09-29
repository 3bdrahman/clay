import type { ToolDefinition } from './types';

export const MAX_FILTER_LIMIT = 50;
export const DEFAULT_FILTER_LIMIT = 5;

export const TOOL_SCHEMAS: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'list_datasets',
      description: 'List all available datasets with their row counts and column names',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'profile_column',
      description: 'Get statistical profile of a single column (numeric or categorical)',
      parameters: {
        type: 'object',
        properties: {
          dataset: { type: 'string', description: 'Dataset name' },
          column: { type: 'string', description: 'Column name' },
        },
        required: ['dataset', 'column'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'aggregate',
      description: 'Group rows by a column and compute aggregate measures',
      parameters: {
        type: 'object',
        properties: {
          dataset: { type: 'string', description: 'Dataset name' },
          groupBy: { type: 'string', description: 'Column to group by' },
          measures: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                column: { type: 'string' },
                fn: { type: 'string', enum: ['sum', 'mean', 'count', 'min', 'max'] },
                as: { type: 'string' },
              },
              required: ['column', 'fn'],
            },
            minItems: 1,
          },
        },
        required: ['dataset', 'groupBy', 'measures'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'filter_sample',
      description: 'Filter rows and return a sample (default 5, max 50)',
      parameters: {
        type: 'object',
        properties: {
          dataset: { type: 'string', description: 'Dataset name' },
          where: {
            type: 'object',
            properties: {
              column: { type: 'string' },
              op: { type: 'string', enum: ['eq', 'contains', 'gt', 'lt'] },
              value: {},
            },
            required: ['column', 'op', 'value'],
          },
          limit: { type: 'number', minimum: 1, maximum: MAX_FILTER_LIMIT },
        },
        required: ['dataset'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'correlate',
      description: 'Compute correlation between two numeric columns',
      parameters: {
        type: 'object',
        properties: {
          dataset: { type: 'string', description: 'Dataset name' },
          columnA: { type: 'string', description: 'First column' },
          columnB: { type: 'string', description: 'Second column' },
          method: { type: 'string', enum: ['pearson', 'spearman'], default: 'pearson' },
        },
        required: ['dataset', 'columnA', 'columnB'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_code',
      description: 'Execute arbitrary Arquero code in the sandbox',
      parameters: {
        type: 'object',
        properties: {
          code: { type: 'string', description: 'Arquero code to execute' },
        },
        required: ['code'],
      },
    },
  },
];