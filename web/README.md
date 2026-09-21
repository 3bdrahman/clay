# Clay — RAG Assistant

> A browser-based Retrieval-Augmented Generation assistant. Drop your own CSVs, PDFs, or text files and query them with natural language.

Clay combines three retrieval paths behind a single chat surface:
- **Vector search** over uploaded documents (PDFs, markdown, text)
- **Data analysis** over uploaded CSV datasets via Arquero (pandas-like, in-browser)
- **Web search** for current facts and general knowledge

All processing runs in your browser. Bring your own key for **OpenRouter**, **Groq**, or **Together**, or point Clay at a **local OpenAI-compatible server** (Ollama, LM Studio, vLLM, llama.cpp). One user-chosen chat model drives the whole pipeline; the embedding model is auto-picked from the catalog.

Live demo: **https://3bdrahman.github.io/clay/**

See [the root README](../README.md) for architecture, data flow, deployment, and configuration details.

## Getting Started

```bash
cd web
npm install
npm run dev      # http://localhost:5173
npm run build    # Static deploy to any host
```
