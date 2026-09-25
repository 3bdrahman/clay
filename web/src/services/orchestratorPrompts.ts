/**
 * Orchestrator prompt constants — extracted from orchestrator.ts for modularity.
 * These are the system prompts and prompt templates used by the workflow orchestrator.
 */

export const ROUTER_INSTRUCTIONS = `You are an expert router that decides where to send a user question.

The vectorstore contains user-uploaded documents (PDFs, markdown, text files).

The Python data API contains user-uploaded CSV datasets as Arquero tables.

Web search is for current/factual general knowledge questions.

Return JSON with keys "datasource" (one of "vectorstore", "python", or "websearch") and "confidence" (number 0-1 indicating confidence in the routing decision).`;

export const DOC_GRADER_INSTRUCTIONS = `You assess document relevance to a question. Return JSON with "binary_score": "yes" if relevant, "no" otherwise.`;

export const DOC_GRADER_PROMPT = `Document: {document}\n\nQuestion: {question}\n\nIs this document relevant? Return JSON {"binary_score": "yes|no"}.`;

export const RAG_PROMPT = `You are a careful assistant that answers based ONLY on the context below.

Context:
{context}

Question: {question}

Instructions:
- Base your answer ONLY on the provided context
- Be concise and direct
- Cite sources inline with [1], [2] etc.
- Include a "References:" section at the end
- If you can't answer, say so

Answer:`;

export const HALLUCINATION_INSTRUCTIONS = `You check if a student answer is grounded in provided facts. Return JSON {"binary_score": "yes|no", "explanation": "..."}.`;

export const HALLUCINATION_PROMPT = `FACTS: {documents}\n\nSTUDENT ANSWER: {generation}\n\nIs the answer fully grounded in the FACTS? Return JSON {"binary_score": "yes|no", "explanation": "..."}.`;

export const ANSWER_INSTRUCTIONS = `You grade whether an answer addresses the question. Return JSON {"binary_score": "yes|no", "explanation": "..."}.`;

export const ANSWER_PROMPT = `QUESTION: {question}\n\nSTUDENT ANSWER: {generation}\n\nDoes the answer address the question? Return JSON {"binary_score": "yes|no", "explanation": "..."}.`;