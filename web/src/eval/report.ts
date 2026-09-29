// Eval report formatting.

import type { EvalSummary } from './types';

export function formatReport(summary: EvalSummary): string {
  const lines: string[] = [];
  lines.push('# Clay Eval Report');
  lines.push('');
  lines.push(`**Total Questions:** ${summary.total}`);
  lines.push(`**Passed:** ${summary.passed} / ${summary.total} (${((summary.passed / summary.total) * 100).toFixed(1)}%)`);
  lines.push(`**Routing Accuracy:** ${(summary.routingAccuracy * 100).toFixed(1)}%`);
  lines.push(`**Avg Recall@K:** ${(summary.avgRecallAtK * 100).toFixed(1)}%`);
  lines.push(`**Avg Answer Score:** ${(summary.avgAnswerScore * 100).toFixed(1)}%`);
  lines.push(`**Avg Judge Score:** ${(summary.avgJudgeScore * 100).toFixed(1)}%`);
  lines.push(`**Avg Latency:** ${summary.avgLatencyMs.toFixed(0)}ms`);
  lines.push('');

  lines.push('## By Category');
  lines.push('');
  for (const [cat, stats] of Object.entries(summary.byCategory)) {
    lines.push(`- **${cat}**: ${stats.passed}/${stats.total} passed, routing ${(stats.routingAccuracy * 100).toFixed(1)}%`);
  }
  lines.push('');

  lines.push('## Per-Question Results');
  lines.push('');
  for (const r of summary.results) {
    const status = r.error ? '❌ ERROR' : (r.routingCorrect && r.recallAtK >= 0.5 ? '✅ PASS' : '❌ FAIL');
    lines.push(`### ${r.questionId} ${status}`);
    lines.push(`- **Question**: ${r.question}`);
    lines.push(`- **Category**: ${r.category}`);
    lines.push(`- **Expected Source**: ${r.expectedSource}`);
    lines.push(`- **Actual Source**: ${r.actualSource ?? '—'}`);
    lines.push(`- **Routing**: ${r.routingCorrect ? '✅' : '❌'}`);
    lines.push(`- **Retrieved Chunks**: ${r.retrievedChunks}`);
    lines.push(`- **Relevant Chunks**: ${r.relevantChunks}`);
    lines.push(`- **Recall@K**: ${(r.recallAtK * 100).toFixed(1)}%`);
    lines.push(`- **Answer Score**: ${((r.answerScore ?? 0) * 100).toFixed(1)}%`);
    if (r.judgeScore !== undefined) lines.push(`- **Judge Score**: ${(r.judgeScore * 100).toFixed(1)}%`);
    lines.push(`- **Latency**: ${r.latencyMs}ms`);
    if (r.error) lines.push(`- **Error**: ${r.error}`);
    lines.push(`- **Answer Preview**: ${r.answer.slice(0, 200)}...`);
    lines.push('');
  }

  return lines.join('\n');
}