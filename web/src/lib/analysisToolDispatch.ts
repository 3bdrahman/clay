import type { AnalysisToolContext } from './types';
import { AnalysisToolError } from './errors/analysisErrors';
import { listDatasetsTool } from './analysisTools';
import { profileColumnTool } from './analysisTools';
import { aggregateTool } from './analysisTools';
import { filterSampleTool } from './analysisTools';
import { correlateTool } from './analysisTools';
import { runCodeTool } from './analysisTools';

export async function executeToolCall(
  ctx: AnalysisToolContext,
  name: string,
  args: Record<string, unknown>,
  executor?: (code: string) => Promise<unknown>
): Promise<unknown> {
  switch (name) {
    case 'list_datasets':
      return listDatasetsTool(ctx);
    case 'profile_column': {
      if (!args.dataset || typeof args.dataset !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'profile_column requires dataset string');
      }
      if (!args.column || typeof args.column !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'profile_column requires column string');
      }
      return profileColumnTool(ctx, { dataset: args.dataset, column: args.column });
    }
    case 'aggregate': {
      if (!args.dataset || typeof args.dataset !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'aggregate requires dataset string');
      }
      if (!args.groupBy || typeof args.groupBy !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'aggregate requires groupBy string');
      }
      if (!Array.isArray(args.measures) || args.measures.length === 0) {
        throw new AnalysisToolError('executeToolCall', 'aggregate requires non-empty measures array');
      }
      return aggregateTool(ctx, {
        dataset: args.dataset,
        groupBy: args.groupBy,
        measures: args.measures as Array<{ column: string; fn: 'sum' | 'mean' | 'count' | 'min' | 'max'; as?: string }>,
      });
    }
    case 'filter_sample': {
      if (!args.dataset || typeof args.dataset !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'filter_sample requires dataset string');
      }
      return filterSampleTool(ctx, {
        dataset: args.dataset,
        where: args.where as { column: string; op: 'eq' | 'contains' | 'gt' | 'lt'; value: unknown } | undefined,
        limit: typeof args.limit === 'number' ? args.limit : undefined,
      });
    }
    case 'correlate': {
      if (!args.dataset || typeof args.dataset !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'correlate requires dataset string');
      }
      if (!args.columnA || typeof args.columnA !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'correlate requires columnA string');
      }
      if (!args.columnB || typeof args.columnB !== 'string') {
        throw new AnalysisToolError('executeToolCall', 'correlate requires columnB string');
      }
      return correlateTool(ctx, {
        dataset: args.dataset,
        columnA: args.columnA,
        columnB: args.columnB,
        method: args.method as 'pearson' | 'spearman' | undefined,
      });
    }
    case 'run_code': {
      if (!args.code || typeof args.code !== 'string' || args.code.trim() === '') {
        throw new AnalysisToolError('executeToolCall', 'run_code requires non-empty code string');
      }
      if (!executor) {
        throw new AnalysisToolError('executeToolCall', 'run_code requires executor function');
      }
      return runCodeTool({ code: args.code }, executor);
    }
    default:
      throw new AnalysisToolError('executeToolCall', `unknown tool: ${name}`);
  }
}