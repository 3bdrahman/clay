# AGENTS.md — Clay Engineering Doctrine

## ZERO STUBS / MOCKS / PROTOTYPES — HARD RULE

**This repository contains NO mocks, stubs, prototypes, hard-coded placeholders, or simulations in production code. Every fix is a complete, real implementation. This is non-negotiable.**

### What this means in practice

| BANNED in `web/src/` (production code) | ALLOWED in `web/src/**/*.test.ts` (test files only) |
|---|---|
| Hardcoded `'unknown'` placeholders | `vi.fn()`, `vi.mock()` for test doubles |
| `// TODO: real impl` | Behavioral assertions (RED → GREEN) |
| Silent fallback lists (e.g., `if (!index) names = ['employees.csv', ...]`) | N/A |
| `Math.random()` for IDs | `crypto.randomUUID()` in production, any in tests |
| `(x as any).field = ...` escape hatches | Type-unsafe test setup is fine |
| `try { ... } catch { /* swallow */ }` | `try { ... } catch (e) { expect(e).toBe(...) }` |
| Placeholder error messages ("Unknown error") | N/A |
| Magic numbers without named constants | N/A |
| `new Function('...')` for user-supplied code | N/A (used for LLM-generated sandbox code only) |
| `console.error/warn` not gated by `import.meta.env.DEV` in production code paths | `console.error` in ErrorBoundary is fine |
| Lint-disabled code (`// eslint-disable-next-line`) | Allowed only with inline justification |

### Definitions

- **Stub**: A function returning a fake/default value where a real implementation is required.
- **Mock**: A fake implementation (class/function/object) substituted for the real thing.
- **Prototype**: Code marked with `TODO`, `FIXME`, `XXX`, `HACK`, "for now", "later", "properly".
- **Hard-coded**: Magic numbers, URLs, model names, or other constants that should be configurable.
- **Simulation**: Code that pretends to perform an action (e.g., silent fallback) instead of executing the real path.

### Test files exception

Test files (`*.test.ts`, `*.test.tsx`, `*.spec.ts`) MAY use mocks, fakes, and stubs **for the purpose of testing real behavior**. The test mocks must drive real production code paths; the production code under test must contain no mocks.

### Review checklist (every PR)

Before opening a PR, the author MUST verify:

1. `grep -rn "TODO\|FIXME\|XXX\|HACK" web/src/ --include='*.ts' --include='*.tsx' | grep -v test` returns no new results.
2. `grep -rn "as any" web/src/ --include='*.ts' --include='*.tsx' | grep -v test` returns no new results.
3. `grep -rn "console\." web/src/ --include='*.ts' --include='*.tsx' | grep -v test | grep -v "import.meta.env.DEV"` returns no new results.
4. `grep -rn "Math.random\|'unknown'" web/src/ --include='*.ts' --include='*.tsx' | grep -v test` returns no new results (unless justified).
5. All catches that don't re-throw have a typed error path with user-visible feedback.
6. No new `try { ... } catch { }` blocks without intentional handling.
7. Linter passes (`npm run lint`).
8. Type-check passes (`npm run type-check`).
9. Full test suite green (`npm run test`).

### Why this matters

The user has been emphatic: every issue is to be pursued with **complete FULL implementations**, never simplified, never mocked, never deferred. The original audit (GH issues #1–#12) found and fixed multiple stubs/hard-codes in the codebase. This AGENTS.md documents the standard that prevents regressions.

### Issue inventory (all closed)

The GitHub issues that tracked the stub/mock/placeholder inventory — all **CLOSED**. Mapping reflects the actual tracker; a few historical commit messages cite swapped issue numbers (#10↔#11 in `14a4359`/`630eabe`, #9↔#12 in `6d529ab`/`567918f`, and a #5→#7→#6→#8 cycle in `8b87ce2`/`c57f9ea`/`319eae9`/`79dbcd9`), so trust this table over old commit labels:

- #1  initialK = maxRetries copy-paste bug — fixed in `c2df56a`
- #2  Silent IDB load failure (persistenceAvailable exposure) — fixed in `ca87c69`
- #3  Hardcoded model-name patterns in scoring heuristics — fixed in `0e058a8` (externalized to `modelPatterns.config.json`)
- #4  Eval suite describe.skip + scenario-bound stub — fixed in `aa4454c` (schema-bound golden set, suite un-skipped)
- #5  Doc/code mismatch: SHA-256 comment vs FNV-1a impl + duplicate hashText variants — fixed in `79dbcd9` (single FNV-1a in `lib/hash.ts`)
- #6  modelId 'unknown' placeholders discard real embedding IDs — fixed in `c57f9ea`
- #7  Hardcoded example questions referencing bundled sample CSVs — fixed in `8b87ce2` (queries derived from loaded data)
- #8  loadSampleDatasets silently falls back to hardcoded CSV list — fixed in `319eae9`
- #9  Math.random() step IDs — fixed in `567918f` (crypto.randomUUID)
- #10 Suspense fallback={null} + duplicated file-extension list drift — fixed in `14a4359` + `8b87e86`
- #11 Production console.error/warn not gated by import.meta.env.DEV (3 sites) — fixed in `630eabe`
- #12 `as any` escape hatches for error step context — fixed in `6d529ab` (typed `RagError.step`)
- #14 Empty catch blocks across 7 files — fixed in `f21b1f1` + `2096f49` (capture + DEV-log)
- #15 Magic numbers without named constants — fixed in `f21b1f1`, `94b73f0`, `d41e029`
- #16 Silent catch in loadSampleDatasets per-file failures — fixed in `8b87e86` (typed `SampleDatasetLoadError`)
- #17 LLM-generated code lacks security documentation — fixed in `49e2320`
- #18 Hardcoded '(unspecified)' embedding placeholder — fixed in `d41e029`
- #19 Skipped tests in embeddings.test.ts — fixed in `94b73f0`
- #20 Duplicate file-extension list (DataSandbox vs fileExtensions) — fixed in `8b87e86`
- #21 Doc claims SHA-256, code uses FNV-1a — fixed in `79dbcd9`

(#13 is a PR — "wire initialK to its own settings field" — not an issue.)
