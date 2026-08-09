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

### Active issues

The following GitHub issues track the stub/mock/placeholder inventory:
- #1  initialK = maxRetries copy-paste bug — **FIXED** in commit c2df56a
- #2  Silent IDB load failure — **FIXED** in commit ca87c69
- #3  Hardcoded model-name patterns in scoring heuristics
- #4  Eval suite is describe.skip + questions.json scenario-bound stub
- #5  Hardcoded example questions referencing bundled sample CSVs
- #6  Silent fallback in loadSampleDatasets — **FIXED** in commit 319eae9
- #7  modelId 'unknown' placeholders — **FIXED** in commit c57f9ea
- #8  Doc/code mismatch: SHA-256 vs FNV-1a
- #9  `as any` escape hatches for error step context
- #10 Production console leak (3 sites)
- #11 Suspense fallback={null} + file extension drift
- #12 Math.random step IDs — **FIXED** in commit 567918f
