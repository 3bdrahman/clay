# Changelog

All notable changes to **Clay** are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), version follows [SemVer](https://semver.org/).

## [Unreleased]

### Changed

- **Single-model architecture.** One user-chosen chat model now drives every LLM step — routing, code generation, answering, evaluation, and self-correction. Automatic per-role chat selection is gone: with BYOK you pay per token, so the cost decision stays with you. Only the embedding model is auto-picked (best pattern score in the live catalog, via `pickBestEmbedding`).
- **Typed provider API-key fields.** New `ProviderApiKeyField` union (`openrouterApiKey` / `groqApiKey` / `togetherApiKey`) replaces index-signature access; all `as any` / `as unknown as` escape hatches removed from `providers.ts`, `Header.tsx`, and `SettingsPanel.tsx`.
- **Embedding pattern config trimmed and validated.** `modelPatterns.config.json` now carries only embedding + size-tier rules (chat/code/safety/vision heuristics removed); `parseConfig` validates every consumed field.
- **CSP `script-src` no longer ships `'unsafe-inline'` in production builds** (the built bundle has no inline scripts); the dev server keeps it for the React refresh preamble. `'unsafe-eval'` stays in both modes — Arquero's table-verb code generation cannot run under a CSP without it — and `'wasm-unsafe-eval'` covers the QuickJS/transformers.js WASM compilation.
- **The PDF path is fully lazy.** `pdfjs-dist` (~1.3MB) and the pdf-extraction module load on the first PDF drop instead of at startup: `files.ts` imports `pdf.ts` dynamically and `pdf.ts` pulls `pdfjs-dist` through a memoized dynamic loader whose failure resets, so a later extraction retries instead of every subsequent call rejecting off a memoized failure.
- **`rehydrateSandboxTables` memoizes parsed CSVs** (keyed by dataset name, guarded by the csv's FNV-1a hash): the service bundle is recreated on every settings/catalog/dataset change, and each recreation no longer re-parses every dataset through Arquero. A re-upload under the same name re-parses instead of serving a stale table.

### Removed

- **NVIDIA NIM provider remnants and Netlify deployment.** Deleted the Netlify Function proxy (`web/netlify/`), root `netlify.toml`, and the Netlify `deploy.yml` workflow; dropped the stale `integrate.api.nvidia.com` entry from the CSP `connect-src`.
- **Dead code.** `useErrorHandler` hook, `getProviderDisplayName`, `getProvidersWithFreeTier`, unused `ProviderConfig` fields (`apiKeyEnvVar`, `description`), and the four-role chat picker (`pickBestModels` + per-class scorers).

### Fixed

- DataAnalysisResult.question carried the generated code instead of the user's question.
- An invalid router datasource silently ran no path — now validated with a vectorstore fallback.
- Typed analysis errors (budget exceeded) keep their user-facing messages through the retry layer instead of a generic generation-failure message.
- Analysis with no datasets loaded returns a helpful guide instead of a poor LLM response.
- **Zero skipped tests.** The two `useClay` persistence tests now run: the unavailable-IDB case and the happy path, backed by `fake-indexeddb`'s spec-accurate `IDBFactory`; the happy path now asserts `persistenceAvailable === true` instead of a weak type check.
- Unbalanced `(` in the SettingsPanel API-key hint label.
- Misindented analyzer construction in `eval/runner.ts`.
- `vite.config.ts`: invalid `VITE_DEPLOY_URL` values now log a build-time warning instead of being silently swallowed.
- OpenRouter `Referer` fallback now points at the GitHub Pages origin and is overridable via `VITE_OPENROUTER_REFERER`.
- README/doc drift: GitHub Pages deploy story, current dependency badges, provider list, and model-selection docs now match the code.
- **Broken `404.html` SPA fallback.** It was a stale dev-page copy carrying an unprocessed `%CSP%` placeholder, the dev-only `/src/main.tsx` entry, and root-absolute asset paths — all of which break under a subpath deployment (GitHub Pages serves `public/` files verbatim; `transformIndexHtml` never sees them). It is now a static redirect-to-app-root page rewritten with the resolved build-time `base` by a new `base-404` Vite plugin.
- **Worker leaks across service-bundle recreation.** The embeddings client spawned its worker eagerly on every bundle recreation (every dataset load or settings change) and neither it nor the analysis-sandbox worker was ever terminated. Both now spawn lazily, the embeddings client no longer respawns inside the worker error handler (which could loop endlessly when the worker script keeps failing to load), and each exposes `dispose()` that terminates the worker once pending work drains. The service bundle wires `dispose`, and `useClay` disposes the old bundle when it is replaced or on unmount.
- **CI's build gate now mirrors the deploy workflow's build env** (`DEPLOY_TARGET=github-pages` + `VITE_DEPLOY_URL` instead of a divergent `BASE_PATH` variant), so CI validates the exact configuration that ships to GitHub Pages.
- Doc drift: `index.html` meta/OG/Twitter descriptions still referenced the removed Together/NVIDIA NIM providers; `.env.example` documented `VITE_NIM_BASE_URL` and a removed Netlify deploy workflow (and suggested `VITE_BASE_PATH`, which the build does not read); the README test-count badge was stale; the 0.3.0 `frame-ancestors 'none'` claim was corrected (see that entry).

### Added

- **Agentic analysis tools.** The data-analysis path is now a tool-calling loop: the model inspects real data via deterministic tools (list_datasets, profile_column with full statistics, aggregate, filter_sample, correlate, run_code) before answering, and returns structured insights (finding, evidence, confidence, implication) plus a deliberately-chosen chart. Tool calls appear as live sub-steps in the workflow graph; models without tool support fall back to the previous single-shot path, clearly labeled.
- From-scratch statistics helpers (quantile, Pearson, Spearman) powering the analysis tools — no new dependencies.
- `fake-indexeddb` dev dependency for spec-accurate IndexedDB tests.
- Documented build-time env overrides: `DEPLOY_TARGET`, `BASE_PATH`, `VITE_DEPLOY_URL`, `VITE_CSP_EXTRA_CONNECT_SRC`, `VITE_OPENROUTER_REFERER`.
- **Global keyboard shortcuts** from the README's table: `Cmd/Ctrl+K` (new chat) and `Cmd/Ctrl+Shift+C` (clear chat, through the same confirmation dialog as the header button). `/` focus and `Esc` stop-generation were already wired in `ChatInput`.
- **Session-shared embeddings cache** (`getSharedEmbeddingCache`): the service bundle is recreated on every settings/catalog/dataset change; the shared cache survives recreation so repeated queries skip re-embedding instead of starting cold each time. Tests needing isolation inject their own cache.

## [0.3.0] — 2026-08-06

### Accessibility

- **ChatInput**: Added `aria-describedby` linking to helper text, `useId` for unique IDs, `aria-label` on buttons, fixed Escape key handler (now works when not disabled), added `aria-hidden` on decorative SVGs
- **CitationPanel**: Full ARIA tab pattern implementation with `role="tablist"`, `role="tab"`, `role="tabpanel"`, `aria-selected`, `aria-controls`, `aria-labelledby`, `tabIndex` management for keyboard navigation
- **ExampleQuestions**: Added `role="list"`/`role="listitem"`, `aria-label` on groups, `aria-disabled` on buttons, focus-visible styles with ring offset for dark mode
- **WorkflowGraph**: Changed to semantic `<ol>`/`<li>` with `role="list"`/`role="listitem"`, `aria-label` on container, `aria-hidden` on decorative icons and connectors
- **ChartRenderer**: Added `role="heading" aria-level={3}` on chart titles for screen reader hierarchy
- **index.css**: Enhanced focus-visible styles for all interactive elements, added high-contrast focus utility, animation utilities, reduced motion support, disabled element cursor handling

### Security

- **CSP**: Tightened Content Security Policy — removed `https://3bdrahman.github.io` from `connect-src` (self-referential), added `base-uri 'self'`, `form-action 'self'` directives. Kept `'unsafe-inline'` for `style-src` due to Tailwind/Recharts runtime injection (documented as known exception). (A `frame-ancestors 'none'` claim previously appeared in this entry; it was incorrect — `frame-ancestors` is ignored inside `<meta>` CSP policies and GitHub Pages cannot set response headers, so it was never deliverable.)

### Build & Type Safety

- **vite.config.ts**: Added `sourcemap: true` for production debugging, added `arquero-vendor` and `markdown-vendor` manual chunks for better cache granularity
- **tsconfig.app.json** / **tsconfig.node.json**: Enabled strictest TypeScript flags: `exactOptionalPropertyTypes`, `noImplicitOverride`, `noPropertyAccessFromIndexSignature`, `noUncheckedIndexedAccess`

### Code Quality

- Added explicit `interface ChartRendererProps` for default export
- Fixed WorkflowGraph keyboard focus order and semantic structure
- Improved component prop typing across all polished files

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
