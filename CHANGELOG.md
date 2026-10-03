# Changelog

Notable changes to Clay. Unreleased changes are not a tagged release.

## Unreleased

### Providers and model policy

- Provider choices are now OpenRouter and local servers. Retired Groq and NIM settings migrate without reusing their keys, relay URLs, or model selections for another provider.
- Cloud selection is limited to OpenRouter's approved free-model shortlist. OpenRouter requires exact free variants, live zero pricing, supported tools, and zero-price routing caps; local servers retain their installed-model choices.
- Removed the NIM relay from the current product surface. Local ingestion remains usable while model setup is incomplete, and stale catalog requests cannot overwrite the current provider.

### Fixed

- File uploads and chat now share the same application services and document index, so new documents can be queried without reloading.
- Local file ingestion no longer requires a model API key. Answer generation still requires a configured provider and model.
- Citation excerpts remain accessible in saved answers after a reload.
- Empty web-search responses no longer create fabricated citations. Search errors retain the selected provider's failure instead of silently querying a different provider.
- Setup and sample-data actions expose loading and failure states. Web-search suggestions reflect the configured provider and deployment capabilities.
- Partial sample-data failures preserve the files that loaded successfully and identify the files that failed.
- Suggested questions use numeric measures rather than averaging or summing record IDs, and recognize column names with underscores or mixed casing.
- Closed panels no longer expose modal controls or loading overlays. The conversation sidebar has an explicit close control.

### Changed

- Reworked the README into a demo walkthrough, architecture map, and reproducible development guide. Corrected privacy claims: the model receives relevant document and data context as well as questions.
- Replaced the static passing-test count with the actual CI status. Aligned the private package version with the latest documented version, 0.3.0.
- Removed an inactive live-evaluation test that returned successfully without making any assertions. Evaluation unit tests are documented separately from live model validation.
- CI uses the committed lockfile without an install fallback. GitHub Pages runs type checking, lint, tests, and the production build before deployment.
- Updated social-preview copy and deployment metadata.
- Updated DOMPurify to 3.4.16 to include its upstream sanitization fix.

### Previously implemented

- One user-selected chat model for routing, analysis, generation, and evaluation; fixed local embeddings through Transformers.js.
- Analysis tools for dataset inspection, statistics, aggregation, sampling, correlation, and generated-code execution.
- Fresh QuickJS contexts inside a worker for generated analysis code, with execution time and memory limits.
- IndexedDB persistence, a shared embedding cache, lazy PDF and worker loading, and worker cleanup when services are replaced.
- HNSW search for larger document indexes; exact cosine search for smaller indexes.
- Retry handling that preserves successful source context, tool-call linkage, and actionable provider errors.
- Moved static deployment from Netlify to GitHub Pages.

## 0.3.0 — 2026-08-06

- Added keyboard and screen-reader support for chat controls, source tabs, and workflow steps.
- Improved focus visibility, reduced-motion support, and responsive presentation.
- Split major dependencies into separate build chunks and tightened TypeScript checks.
- Added CSP restrictions for connections, base URLs, and form submissions. GitHub Pages does not provide configurable security response headers.

## 0.2.0 — 2026-08-01

- Made the data workspace empty by default, with three optional sample CSVs.
- Replaced a fixed company scenario with metadata derived from uploaded datasets.
- Added CSV persistence and table rehydration.
- Removed bundled precomputed embeddings whose dimensions did not match the runtime model.

## 0.1.0 — 2026-07-31

- Introduced document retrieval, CSV analysis, web search, citations, charts, and an execution trace.
- Added conversation persistence, themes, CI, and static deployment.
- Initially used NVIDIA NIM for models and embeddings. The current provider and embedding architecture is described above.
