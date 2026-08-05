// Demo LLM Client — provides simulated responses for portfolio/demo mode
// This allows the app to be fully functional without an API key

import type { LLMClient } from './llm';
import type { LLMRequest, LLMResponse } from './types';

const DEMO_RESPONSES: Record<string, { content: string; delay?: number }> = {
  // Router responses
  '{"datasource":"vectorstore"}': { content: '{"datasource":"vectorstore"}' },
  '{"datasource":"python"}': { content: '{"datasource":"python"}' },
  '{"datasource":"websearch"}': { content: '{"datasource":"websearch"}' },
  
  // Document grader responses
  '{"binary_score":"yes"}': { content: '{"binary_score":"yes"}' },
  '{"binary_score":"no"}': { content: '{"binary_score":"no"}' },
  
  // Evaluation responses
  '{"binary_score":"yes","explanation":"The answer is grounded in the provided facts."}': { 
    content: '{"binary_score":"yes","explanation":"The answer is grounded in the provided facts."}' 
  },
  '{"binary_score":"no","explanation":"The answer contains hallucinated information."}': { 
    content: '{"binary_score":"no","explanation":"The answer contains hallucinated information."}' 
  },
  '{"binary_score":"yes","explanation":"The answer directly addresses the question."}': { 
    content: '{"binary_score":"yes","explanation":"The answer directly addresses the question."}' 
  },
  '{"binary_score":"no","explanation":"The answer does not fully address the question."}': { 
    content: '{"binary_score":"no","explanation":"The answer does not fully address the question."}' 
  },
};

const DEMO_ANSWERS: Record<string, string> = {
  'average salary': "Based on the data, the average salary by department is:\n\n- Engineering: $110,000\n- Sales: $80,000\n- Marketing: $90,000\n\n[1] Source: employees dataset",
  'project status': "Project distribution by status:\n\n- Active: 2 projects\n- Completed: 1 project\n\n[1] Source: projects dataset",
  'feedback': "Average rating per project:\n\n- Project A: 4.5/5\n- Project B: 4.2/5\n- Project C: 4.8/5\n\n[1] Source: feedback dataset",
  'total budget': "Total budget by project status:\n\n- Active: $90,000\n- Completed: $30,000\n\n[1] Source: projects dataset",
  'employee count': "Total employees: 4\n\n[1] Source: employees dataset",
  'document summary': "Based on your uploaded documents, here are the key themes:\n\n1. **Project Planning** - Multiple documents discuss project timelines and milestones\n2. **Team Structure** - Information about department organization and roles\n3. **Budget Allocation** - Financial planning across projects\n\nThe documents contain meeting notes, project proposals, and team updates.",
  'web search': "Based on web search results, here are the latest trends:\n\n1. **AI in Business** - Generative AI adoption is accelerating across enterprises\n2. **Cloud ML Platforms** - AWS, Azure, and GCP are expanding their ML offerings\n3. **Vector Databases** - Growing adoption for RAG applications\n\n[1] Source: Web search results",
  'default': "I'm running in demo mode without an API key. In this mode, I can show you how Clay works with sample data.\n\nTry asking:\n- \"Average salary by department\" (data analysis)\n- \"Summarize my documents\" (vector search - after uploading PDFs)\n- \"What are the latest AI trends?\" (web search)\n\nTo use the full AI capabilities, add a NVIDIA NIM API key in Settings.",
};

function matchDemoResponse(input: string): string {
  const lower = input.toLowerCase();
  
  // Check for exact matches first
  for (const [key, val] of Object.entries(DEMO_RESPONSES)) {
    if (lower.includes(key.toLowerCase().slice(0, 50))) {
      return val.content;
    }
  }
  
  // Check demo answers
  for (const [key, val] of Object.entries(DEMO_ANSWERS)) {
    if (lower.includes(key)) {
      return val;
    }
  }
  
  return DEMO_ANSWERS.default;
}

export function createDemoLLMClient(): LLMClient {
  return {
    async invoke(req: LLMRequest): Promise<LLMResponse> {
      // Simulate network delay
      await new Promise(r => setTimeout(r, 300 + Math.random() * 200));
      
      const userContent = req.messages.find(m => m.role === 'user')?.content || '';
      const systemPrompt = req.system || '';
      const matched = matchDemoResponse(userContent);
      
      // Special handling for different task types
      if (req.jsonMode) {
        // Try to return valid JSON
        try {
          JSON.parse(matched);
          return { content: matched, usage: { promptTokens: 100, completionTokens: 50, totalTokens: 150 }, model: 'demo-model' };
        } catch {
          // Return appropriate JSON based on system prompt
          if (systemPrompt.includes('router') || systemPrompt.includes('Router')) {
            return { content: '{"datasource":"vectorstore"}', usage: { promptTokens: 100, completionTokens: 20, totalTokens: 120 }, model: 'demo-model' };
          }
          if (systemPrompt.includes('relevance') || systemPrompt.includes('grade')) {
            return { content: '{"binary_score":"yes"}', usage: { promptTokens: 100, completionTokens: 20, totalTokens: 120 }, model: 'demo-model' };
          }
          if (systemPrompt.includes('hallucination')) {
            return { content: '{"binary_score":"yes","explanation":"The answer is grounded in the provided facts."}', usage: { promptTokens: 100, completionTokens: 30, totalTokens: 130 }, model: 'demo-model' };
          }
          if (systemPrompt.includes('grade whether')) {
            return { content: '{"binary_score":"yes","explanation":"The answer directly addresses the question."}', usage: { promptTokens: 100, completionTokens: 30, totalTokens: 130 }, model: 'demo-model' };
          }
          if (systemPrompt.includes('data analyst') || systemPrompt.includes('Arquero')) {
            if (userContent.includes('salary') || userContent.includes('average')) {
              const code = "result = employees.groupby('department').rollup({ avg_salary: d => op.mean(d.salary_usd) })";
              return { content: JSON.stringify({ code, explanation: "Average salary by department" }), usage: { promptTokens: 200, completionTokens: 80, totalTokens: 280 }, model: 'demo-model' };
            }
            if (userContent.includes('project') && userContent.includes('status')) {
              const code = "result = projects.groupby('status').count()";
              return { content: JSON.stringify({ code, explanation: "Project count by status" }), usage: { promptTokens: 200, completionTokens: 80, totalTokens: 280 }, model: 'demo-model' };
            }
            if (userContent.includes('budget')) {
              const code = "result = projects.groupby('status').rollup({ total_budget: d => op.sum(d.budget_usd) })";
              return { content: JSON.stringify({ code, explanation: "Total budget by project status" }), usage: { promptTokens: 200, completionTokens: 80, totalTokens: 280 }, model: 'demo-model' };
            }
            if (userContent.includes('feedback') || userContent.includes('rating')) {
              const code = "result = feedback.groupby('project_id').rollup({ avg_rating: d => op.mean(d.rating) })";
              return { content: JSON.stringify({ code, explanation: "Average rating per project" }), usage: { promptTokens: 200, completionTokens: 80, totalTokens: 280 }, model: 'demo-model' };
            }
            if (userContent.includes('count') || userContent.includes('how many')) {
              const code = "result = employees.count()";
              return { content: JSON.stringify({ code, explanation: "Total employee count" }), usage: { promptTokens: 200, completionTokens: 50, totalTokens: 250 }, model: 'demo-model' };
            }
          }
          return { content: '{"binary_score":"yes"}', usage: { promptTokens: 100, completionTokens: 20, totalTokens: 120 }, model: 'demo-model' };
        }
      }
      
      // Streaming response - return full content
      return { content: matched, usage: { promptTokens: 200, completionTokens: 100, totalTokens: 300 }, model: 'demo-model' };
    },
    
    async stream(req: LLMRequest, onToken: (token: string) => void, signal?: AbortSignal): Promise<LLMResponse> {
      const response = await this.invoke(req);
      const content = response.content || '';
      
      // Simulate streaming by sending chunks
      const words = content.split(' ');
      for (let i = 0; i < words.length; i++) {
        if (signal?.aborted) break;
        const token = words[i] + (i < words.length - 1 ? ' ' : '');
        onToken(token);
        await new Promise(r => setTimeout(r, 10 + Math.random() * 30));
      }
      
      return response;
    },
  };
}