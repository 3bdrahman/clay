// ResultRenderer — renders analysis results (tables, charts, JSON, primitives)

import { Suspense } from 'react';
import type { DataAnalysisResult } from '../lib/types';

import { ChartRendererLazy, PanelFallback } from './LazyPanels';

const ChartRenderer = ChartRendererLazy;

export function ResultRenderer({ result, chartConfig }: { result: unknown; chartConfig?: DataAnalysisResult['chartConfig'] }) {
  if (chartConfig) {
    return (
      <Suspense fallback={<PanelFallback />}>
        <ChartRenderer config={chartConfig} />
      </Suspense>
    );
  }
  if (Array.isArray(result)) {
    return (
      <div className="overflow-x-auto">
        <table className="text-[11px] w-full">
          <tbody>
            {result.slice(0, 20).map((row, i) => (
              <tr key={i} className="border-b border-ink-100 dark:border-ink-700">
                {typeof row === 'object' && row !== null ? (
                  Object.entries(row as Record<string, unknown>).map(([k, v]) => (
                    <td key={k} className="px-2 py-1 font-mono">
                      <span className="text-ink-500">{k}:</span> {String(v)}
                    </td>
                  ))
                ) : (
                  <td className="px-2 py-1 font-mono">{String(row)}</td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
        {result.length > 20 && <div className="text-[10px] text-ink-400 mt-1">+ {result.length - 20} more rows</div>}
      </div>
    );
  }
  if (result && typeof result === 'object') {
    return (
      <pre className="text-[11px] font-mono bg-ink-50 dark:bg-ink-900 p-2 rounded overflow-x-auto">
        {JSON.stringify(result, null, 2)}
      </pre>
    );
  }
  return <div className="text-sm font-mono">{String(result)}</div>;
}