# Changelog

All notable changes to **Clay** are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), version follows [SemVer](https://semver.org/).

## [0.2.0] — 2026-08-01

### Breaking changes

- **Drop the forced scenario.** Removed the bundled "Aurora Consulting" dataset and narrative. The app now starts **empty** and asks the user to bring their own data. A tiny generic 3-table sample (`employees.csv`, `projects.csv`, `feedback.csv`) is bundled only as an opt-in convenience.
- **Data-driven metadata.** `DatasetMeta` no longer carries hardcoded `keyFields` / `commonAnalyses`. The analyzer routes datasets by token overlap on dataset names and column names — works with any CSV the user uploads.
- **`SandboxDataset` now persists CSV text** so the Arquero table can be rehydrated on reload.

### Added

- Real Arquero `ColumnTable` instances flow from `addFiles` / `loadSampleData` into the analyzer. Replaces the previous broken `__sandbox: true` stub tables.
- Bundled generic placeholder CSVs in `web/public/data/datasets/`. Sample is opt-in via the **Load sample data** button.
- `web/src/services/sandboxTables.ts` — module-level registry for live Arquero tables (out-of-band from Zustand because `ColumnTable` isn't JSON-serializable).

### Removed

- `Company/` source folder, `scripts/prep_data.py`, `dataset_manager.py`.
- Hardcoded dataset metadata (`DATASET_META` with 12 entries).
- `web/public/data/chunks/` and the broken `index.json` (384-dim BoW hash, dimension-mismatched with the runtime `nv-embedqa-e5-v5` 1024-dim vectors — every retrieval was returning score 0).
- All references to "Aurora Consulting" from prompts, UI copy, and docs.

### Migration notes

- The shipped `index.json` was removed because it contained a 384-dim bag-of-words hash embedding whose dimensions did not match the 1024-dim vectors produced by `nv-embedqa-e5-v5`. Vector search now uses runtime embeddings only — see the next release notes for the localStorage cache layer.
- Users who previously relied on the Aurora sample data can click **Load sample data** in the Data sandbox to load a small equivalent.

## [0.1.0] — 2026-07-31

First publicly-shippable release. Project is fully functional, zero mocking, single-provider (NVIDIA NIM), dynamic model picker, and ships with a Data Sandbox.

### Added

- **Dynamic model picker** — `src/lib/models.ts` fetches the live NVIDIA NIM catalog (`GET /v1/models`) and heuristically picks the best model for routing, code generation, answer, evaluation, and embeddings.
- **Data Sandbox** — drop CSV/PDF/MD/TXT/JSON files. CSVs are loaded as Arquero tables; PDFs/MD/TXT/JSON are chunked, embedded via NIM, and added to the vector store.
- **One provider: NVIDIA NIM** — one API key, one free tier.
- **Live workflow visualization** — animated stepper showing each step of the workflow with timing and status.
- **Self-correcting retry loop** — quality grader scores each answer; failed answers retry with a different source.
- **Vector search** over documents — runtime API embeddings.
- **Data analysis** — LLM generates Arquero code; safely executed via `new Function()` in an isolated scope.
- **Web search** — Serper (with key) or DuckDuckGo HTML (no key).
- **Dark mode** with full theming.
- **Mobile responsive** down to ~360 px.
- **localStorage persistence** — settings, theme, chat history.
- **Charts** via Recharts.
- **Source citations** with inline chips and hover previews.
- **Live markdown rendering** via `marked` + `DOMPurify`.

### Packaging

- **GitHub Pages deployment** via `.github/workflows/deploy.yml`.
- **CI workflow** at `.github/workflows/ci.yml` — type-check, build, lint.
- **ErrorBoundary** wraps the entire app.
- **Vite base path** honors `BASE_PATH` env var.
- **pdfjs-dist** — chunked separately (`pdf-vendor`).
