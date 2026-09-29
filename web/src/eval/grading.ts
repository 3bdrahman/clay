// Eval grading — LLM-as-judge scoring.

/**
 * Score an answer against a golden answer using LLM-as-judge.
 * Returns a score 0-1 and a one-line rationale.
 */
export async function scoreWithJudge(
  llm: {
    invoke: (req: {
      system?: string;
      messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>;
      jsonMode?: boolean;
      temperature?: number;
      model?: string;
    }) => Promise<{ content: string; usage?: { totalTokens?: number } }>;
  },
  model: string,
  question: string,
  generated: string,
  golden: string
): Promise<{ score: number; rationale: string }> {
  const prompt = `Question: ${question}

Golden Answer: ${golden}

Generated Answer: ${generated}

Score the generated answer on factual coverage of the golden answer (0.0 to 1.0).
Consider: Does the generated answer contain the key facts from the golden answer? Are there hallucinations or missing critical information?
Return JSON: {"score": 0.0-1.0, "rationale": "one-line explanation"}`;

  try {
    const resp = await llm.invoke({
      system: 'You are an expert evaluator. Score factual coverage accurately and concisely.',
      messages: [{ role: 'user', content: prompt }],
      jsonMode: true,
      temperature: 0,
      model,
    });

    const parsed = JSON.parse(resp.content || '{}');
    const score = Math.max(0, Math.min(1, Number.isFinite(parsed.score) ? parsed.score : 0));
    const rationale = String(parsed.rationale ?? '').slice(0, 200);
    return { score, rationale };
  } catch (e) {
    if (import.meta.env.DEV) {
      console.warn('[eval] Judge scoring failed:', e);
    }
    return { score: 0, rationale: 'Judge scoring failed' };
  }
}