# Clay — RAG Assistant

> A browser-based Retrieval-Augmented Generation assistant. Drop your own CSVs, PDFs, or text files and query them with natural language.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/Tests-130_passing-brightgreen.svg)](https://github.com/usef/clay/actions)
[![Deploy](https://img.shields.io/badge/Deploy-GitHub_Pages-2ea44f.svg)](https://github.com/usef/clay/actions)

Clay combines three retrieval paths behind a single chat surface:

- **Vector search** over uploaded documents (PDFs, markdown, text)
- **Data analysis** over uploaded CSV datasets via Arquero (pandas-like, in-browser)
- **Web search** for current facts and general knowledge

The orchestrator routes each question to the right source, runs an LLM-as-judge self-correction loop, and renders the entire pipeline in real time.

All processing runs in the browser. Two outbound paths are supported for LLM inference: **NVIDIA NIM** (cloud) or a **local OpenAI-compatible server** (Ollama, LM Studio, vLLM, llama.cpp server). Switch in Settings. Web search is optional and also runs client-side via DuckDuckGo or Serper API.

---

## Live Demo

**Try it now:** [https://3bdrahman.github.io/clay](https://3bdrahman.github.io/clay)

The demo runs in **demo mode** — no API key required! Click "Load Sample Data" to try data analysis queries, or add your own files and an API key for full AI capabilities.

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

### Demo Mode (No API Key Required)

Open the app without any configuration — it automatically runs in demo mode with simulated AI responses. Perfect for trying out the UI and data analysis features with the sample dataset.

### Production build

```bash
npm run build        # → web/dist/ (static files, ready to deploy)
```

Deploys to GitHub Pages, Netlify, Vercel, Cloudflare Pages, or any static host.

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

On startup, Clay calls `GET https://integrate.api.nvidia.com/v1/models` and picks the best model per task by size + family. No hardcoded model names — when NIM rotates its catalog, picks update automatically.

| Task | Heuristic |
|---|---|
| **Routing** | Smallest instruction-following chat model |
| **Evaluation** | Different small chat model from routing |
| **Code generation** | Code-specialist (Codestral, CodeLlama, etc.) |
| **Answer** | Largest available chat model |
| **Embeddings** | Best QA embedding model |

Cache TTL is 1 hour. Click **Refresh** in Settings to refetch.

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