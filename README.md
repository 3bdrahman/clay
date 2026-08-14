# Clay — RAG Assistant

> A browser-based Retrieval-Augmented Generation assistant. Drop your own CSVs, PDFs, or text files and query them with natural language.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/Tests-396_passing-brightgreen.svg)](https://github.com/usef/clay/actions)
[![Deploy](https://img.shields.io/badge/Deploy-Netlify-00C7B7.svg)](https://app.netlify.com/sites/clay-rag/deploys)

Clay combines three retrieval paths behind a single chat surface:

- **Vector search** over uploaded documents (PDFs, markdown, text)
- **Data analysis** over uploaded CSV datasets via Arquero (pandas-like, in-browser)
- **Web search** for current facts and general knowledge

The orchestrator routes each question to the right source, runs an LLM-as-judge self-correction loop, and renders the entire pipeline in real time.

All processing runs in the browser. Two outbound paths are supported for LLM inference: **NVIDIA NIM** (cloud) or a **local OpenAI-compatible server** (Ollama, LM Studio, vLLM, llama.cpp server). Switch in Settings. Web search is optional and also runs client-side via DuckDuckGo or Serper API.

---

## Live Demo

**Try it now:** [https://clay-rag.netlify.app](https://clay-rag.netlify.app)

Add your NVIDIA NIM API key in Settings (free tier: [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)) or configure a local OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp) to use the full AI capabilities.

> **Note:** The Netlify deployment includes a Netlify Function proxy (`web/netlify/functions/nim-proxy.ts`) that automatically handles CORS for NVIDIA NIM. NIM cloud works zero-config on Netlify.

---

## What this project demonstrates

- **Production-tier RAG architecture** — route → retrieve → grade → generate → evaluate, with retries
- **Multi-source synthesis** — documents, structured data, and the open web in one answer
- **Live workflow visualization** — every step (with timing) is shown as it runs
- **Self-correcting quality loop** — LLM-as-judge hallucination check + answer-usefulness grading
- **Dynamic model picker** — fetches NIM's live catalog (~100 models), picks the best per task
- **Bring-your-own-data** — no forced scenario; drop any CSV/PDF/MD/TXT/JSON and start querying
- **Runs entirely in your browser** — no backend server required, deploy anywhere as static files
- **Privacy-first** — your data never leaves your browser; only queries go to your configured LLM provider

---

## Getting Started

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

The app starts **empty**. Click **Data** in the header and either:

1. Drop your own files (CSV → Arquero table; PDF/MD/TXT/JSON → chunked + embedded), or
2. Click **Load sample data** for a tiny 3-table demo dataset

Then paste your NVIDIA NIM API key in **Settings** (free tier: [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)) and ask away.

### Production build

```bash
npm run build        # → web/dist/ (static files, ready to deploy)
```

Deploys to **Netlify** (recommended — NIM cloud works zero-config via Netlify Function), or any static host.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (client-side app)                 │
│                                                             │
│   ┌─────────────┐    ┌──────────────────┐                  │
│   │  React UI   │◄──►│  Zustand Store   │◄──► localStorage │
│   └─────────────┘    └──────────────────┘                  │
│           │                    │                           │
│           ▼                    ▼                           │
│   ┌─────────────────────────────────────────┐               │
│   │     Workflow Orchestrator (FSM)         │               │
│   │                                         │               │
│   │  ┌────────┐  ┌─────────┐  ┌─────────┐   │               │
│   │  │ Route  │→ │Retrieve │→ │ Grade   │   │               │
│   │  └────────┘  └─────────┘  └─────────┘   │               │
│   │       │           │             │       │               │
│   │       ▼           ▼             ▼       │               │
│   │  ┌────────┐  ┌─────────┐  ┌─────────┐   │               │
│   │  │VectorDB│  │Analyzer │  │ Web     │   │               │
│   │  │(cosine)│  │(Arquero)│  │ Search  │   │               │
│   │  └────────┘  └─────────┘  └─────────┘   │               │
│   │       │           │             │       │               │
│   │       └───────────┴─────────────┘       │               │
│   │                   ▼                     │               │
│   │           ┌──────────────┐              │               │
│   │           │  Generate    │              │               │
│   │           │  + Evaluate  │◄─ retry loop │               │
│   │           └──────────────┘              │               │
│   └─────────────────────────────────────────┘               │
│                       │                                     │
│                       ▼  (outbound LLM calls)                │
│   ┌─────────────────────────────────────────┐               │
│   │     NVIDIA NIM or Local LLM Server       │               │
│   │  routing/codeGen/answer/eval + embeddings │               │
│   └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

- **Vite + React 19 + TypeScript** (strict mode)
- **Tailwind CSS** — utility-first styling, dark mode
- **Zustand** — state with `localStorage` persistence
- **Arquero** — pandas-like DataFrame library for in-browser data analysis
- **Recharts** — declarative charts
- **Marked + DOMPurify** — safe markdown rendering
- **pdfjs-dist** — client-side PDF text extraction
- **NVIDIA NIM** — LLM provider (OpenAI-compatible) or local server

---

## Data flow

When you drop a CSV, it's parsed by Arquero into a real `ColumnTable` and registered as a variable the LLM-generated code can query. When you drop a PDF/MD/TXT/JSON, it's chunked (~800 chars / ~200 overlap), embedded via `nv-embedqa-e5-v5` (or your configured embedding model), and added to the in-memory vector store.

The sandbox is your workspace — there's no preloaded scenario. The next question routes against whatever you've loaded. Your data stays in your browser; only the question and relevant context are sent to your LLM provider.

---

## Configuration

Open **Settings** and pick a provider.

### NVIDIA NIM (cloud)

Paste your free `nvapi-...` key. The key is stored in browser localStorage and sent only to `integrate.api.nvidia.com/v1` for LLM inference. Your document data is never sent to NVIDIA — only the question and retrieved context.

### Local server (private)

Pick **Local server** in Settings and point Clay at any OpenAI-compatible endpoint. No API key required (some servers, like LM Studio, accept a key — paste it if yours does).

| Server | Default URL |
|---|---|
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| vLLM | `http://localhost:8000/v1` |
| llama.cpp server | `http://localhost:8080/v1` |

Click **Discover** to fetch the list of models the server exposes, then pick one per task (Routing, Code generation, Answer, Evaluation, Embedding). You can also type model IDs directly — Clay will use them as-is.

### Dynamic model picker (NIM only)

On startup, Clay calls `GET https://integrate.api.nvidia.com/v1/models` and picks the best model per task by size + family using heuristic patterns for known model families (see `web/src/lib/modelPatterns.ts`). The patterns are designed to work with the current NIM catalog but may require updates if the catalog changes significantly.

| Task | Heuristic |
|---|---|
| **Routing** | Smallest instruction-following chat model |
| **Evaluation** | Different small chat model from routing |
| **Code generation** | Code-specialist (Codestral, CodeLlama, etc.) |
| **Answer** | Largest available chat model |
| **Embeddings** | Best QA embedding model |

Cache TTL is 1 hour. Click **Refresh** in Settings to refetch.

---

### Production CORS constraint (NVIDIA NIM only)

NVIDIA NIM's API only returns `Access-Control-Allow-Origin: https://build.nvidia.com`. Browsers enforce this via CORS, so **direct calls from any other origin (GitHub Pages, Netlify, Vercel, Cloudflare Pages, localhost:5173 in preview, etc.) will fail** with a CORS error — even if the API returns HTTP 200.

**This is not a bug in Clay** — it's a browser security feature. The API key would be exposed if the browser allowed reading responses from arbitrary origins.

The development server works because Vite proxies `/nim-api/*` → `https://integrate.api.nvidia.com/v1/*` (see `web/vite.config.ts`). For production, you must replicate this proxy yourself.

#### Option 1: Deploy an edge proxy (recommended)

Deploy a tiny serverless function that forwards requests to NIM and adds CORS headers:

**Cloudflare Worker** (`wrangler.toml` + `src/index.ts`):
```toml
name = "clay-nim-proxy"
main = "src/index.ts"
compatibility_date = "2024-01-01"
```

```typescript
export default {
  async fetch(request: Request, env: any): Promise<Response> {
    const url = new URL(request.url);
    const target = `https://integrate.api.nvidia.com${url.pathname}${url.search}`;
    
    const response = await fetch(target, {
      method: request.method,
      headers: {
        ...Object.fromEntries(request.headers),
        'Origin': 'https://integrate.api.nvidia.com',
      },
      body: request.method !== 'GET' && request.method !== 'HEAD' ? request.body : undefined,
    });

    return new Response(response.body, {
      status: response.status,
      headers: {
        ...Object.fromEntries(response.headers),
        'Access-Control-Allow-Origin': '*', // or restrict to your domain
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
  },
};
```

**Vercel Function** (`api/nim-proxy/[...slug].ts`):
```typescript
import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const { slug } = req.query;
  const path = Array.isArray(slug) ? slug.join('/') : slug;
  const target = `https://integrate.api.nvidia.com/v1/${path}`;

  const response = await fetch(target, {
    method: req.method,
    headers: {
      ...req.headers,
      origin: 'https://integrate.api.nvidia.com',
    } as any,
    body: ['GET', 'HEAD'].includes(req.method) ? undefined : JSON.stringify(req.body),
  });

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') return res.status(200).end();

  const data = await response.text();
  res.status(response.status).send(data);
}
```

After deploying your proxy, rebuild Clay with:
```bash
VITE_NIM_BASE_URL=https://your-proxy.example.com/v1 npm run build
```

The build will automatically inject your proxy's origin into the CSP `connect-src` directive.

#### Option 2: Use a local server

Switch to **Local server** in Settings and run Ollama, LM Studio, vLLM, or llama.cpp on your machine. This avoids CORS entirely since requests go to `localhost`.

For Ollama, you may need to enable CORS:
```bash
OLLAMA_ORIGINS="*" ollama serve
```

---

## Workflow

1. **Routes** the question to one of three sources (vectorstore / data / websearch)
2. **Retrieves** top-K from the chosen source
3. **Grades** retrieved docs with an LLM-as-judge (filters irrelevant)
4. **Generates** a cited answer grounded in the retrieved context
5. **Evaluates** the answer (hallucination check + question-answer match). If not useful, retries with a different source
6. **Returns** the answer with sources, workflow trace, and any analysis code

The entire flow is visible in real time — click "Show workflow" on any response.

---

## Project Structure

```
clay/
├── README.md
├── CHANGELOG.md
├── web/                        ← the app (this is what gets deployed)
│   ├── public/
│   │   └── data/
│   │       └── datasets/       ← optional bundled sample CSVs
│   ├── src/
│   │   ├── components/         ← React components
│   │   ├── hooks/              ← React hooks
│   │   ├── lib/                ← LLM, embeddings, vector store, web search, model picker
│   │   ├── services/           ← orchestrator, data analyzer, file processor, sandbox tables
│   │   ├── store.ts            ← Zustand store
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── vite.config.ts
│   └── package.json
└── .github/
    └── workflows/
        └── deploy.yml          ← GitHub Pages deployment
```

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `/` | Focus input |
| `Esc` | Stop generation |
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `Cmd/Ctrl+K` | New chat |
| `Cmd/Ctrl+Shift+C` | Clear chat |

---

## Development

```bash
# Run tests
npm run test

# Type-check
npm run type-check

# Lint
npm run lint

# Full verification (type-check + lint + test + build)
npm run verify
```

---

## License

MIT