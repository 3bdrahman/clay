import type { LLMRequest } from './types';

/**
 * Build the OpenAI-compatible messages array from an LLMRequest.
 * Shared by both invoke (non-streaming) and stream paths to avoid duplication.
 */
export function buildMessages(req: LLMRequest): Array<Record<string, unknown>> {
  const messages: Array<Record<string, unknown>> = [];
  if (req.system) messages.push({ role: 'system', content: req.system });
  for (const m of req.messages) {
    if (m.role === 'assistant' && m.toolCalls) {
      messages.push({
        role: 'assistant',
        content: m.content,
        tool_calls: m.toolCalls.map(c => ({
          id: c.id,
          type: 'function',
          function: { name: c.function.name, arguments: c.function.arguments },
        })),
      });
    } else if (m.role === 'tool') {
      messages.push({
        role: 'tool',
        content: m.content,
        tool_call_id: m.toolCallId,
      });
    } else {
      messages.push({ role: m.role, content: m.content });
    }
  }
  return messages;
}