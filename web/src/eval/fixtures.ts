// Shared eval fixtures — one definition for the golden-set inputs so the
// eval test file and the golden-summary gate cannot drift apart.
import type { Settings } from '../lib/types';
import { generateEvalQuestions } from './dynamicQuestions';
import type { EvalQuestion } from './types';

export const TEST_SETTINGS: Settings = {
  provider: 'openrouter',
  openrouterApiKey: import.meta.env.VITE_EVAL_API_KEY ?? '',
  groqApiKey: '',
  apiKey: '',
  webSearchProvider: 'duckduckgo',
  serperApiKey: '',
  temperature: 0,
  maxRetries: 3,
  theme: 'system',
  localServerUrl: '',
  localModels: { chat: '' },
  localCatalog: [],
  localCatalogFetchedAt: 0,
  pickedModelsOverride: {
    chatModel: '',
  },
};

export const TEST_DATASETS = [
  {
    name: 'employees',
    fileName: 'employees.csv',
    columns: ['id', 'name', 'department', 'salary', 'hire_date'],
    rowCount: 100,
    sampleRows: [
      { id: 1, name: 'Alice', department: 'Engineering', salary: 120000, hire_date: '2020-01-15' },
      { id: 2, name: 'Bob', department: 'Sales', salary: 90000, hire_date: '2019-03-22' },
    ],
  },
  {
    name: 'projects',
    fileName: 'projects.csv',
    columns: ['id', 'name', 'budget', 'status', 'start_date'],
    rowCount: 50,
    sampleRows: [
      { id: 1, name: 'Project Alpha', budget: 500000, status: 'active', start_date: '2023-01-01' },
    ],
  },
];

export const TEST_DOCUMENTS = [
  { fileName: 'handbook.pdf' },
  { fileName: 'benefits.md' },
];

export const TEST_QUESTIONS: EvalQuestion[] = generateEvalQuestions(TEST_DATASETS, TEST_DOCUMENTS);
