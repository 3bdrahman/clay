# Clay — AI Internal Assistant

> A browser-based Retrieval-Augmented Generation assistant. Drop your own CSVs, PDFs, or text files and query them with natural language.

Clay combines three retrieval paths behind a single chat surface:
- **Vector search** over uploaded documents (PDFs, markdown, text)
- **Data analysis** over uploaded CSV datasets via Arquero (pandas-like, in-browser)
- **Web search** for current facts and general knowledge

All processing runs in your browser. Two provider options: **NVIDIA NIM** (cloud, free tier) or a **local OpenAI-compatible server** (Ollama, LM Studio, vLLM, llama.cpp).

See [the root README](../README.md) for architecture, data flow, and server configuration details.

## Getting Started

```bash
cd web
npm install
npm run dev      # http://localhost:5173
npm run build    # Static deploy to any host
```