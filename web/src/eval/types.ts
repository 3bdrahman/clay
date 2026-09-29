// Eval types — shared type definitions for the eval suite.

export interface EvalQuestion {
  id: string;
  question: string;
  category: 'data_analysis' | 'documents' | 'web_search';
  expectedSource: 'python' | 'vectorstore' | 'websearch';
  expectedColumnIntent?: string[];
  goldenAnswer: string;
  minRelevantChunks: number;
}

export interface EvalResult {
  questionId: string;
  question: string;
  category: string;
  expectedSource: string;
  actualSource: string | undefined;
  routingCorrect: boolean;
  retrievedChunks: number;
  relevantChunks: number;
  recallAtK: number;
  answer: string;
  latencyMs: number;
  error?: string;
  answerScore?: number; // Lexical overlap score 0-1
  judgeScore?: number; // LLM-as-judge score 0-1
}

export interface EvalSummary {
  total: number;
  passed: number;
  failed: number;
  routingAccuracy: number;
  avgRecallAtK: number;
  avgLatencyMs: number;
  avgAnswerScore: number;
  avgJudgeScore: number;
  byCategory: Record<string, { total: number; passed: number; routingAccuracy: number }>;
  results: EvalResult[];
}