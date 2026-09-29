import { RagError, RagErrorCode } from './base';

/**
 * Error thrown by analysis tools when a tool-specific failure occurs.
 * Wraps the tool name and problem description for debugging.
 */
export class AnalysisToolError extends RagError {
  constructor(tool: string, problem: string, cause?: Error) {
    super({
      code: RagErrorCode.UNKNOWN_ERROR,
      message: `Analysis tool "${tool}" failed: ${problem}`,
      cause,
      retryable: false,
      context: { tool, problem },
    });
  }
}