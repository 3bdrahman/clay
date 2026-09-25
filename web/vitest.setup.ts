// React 19 requires this flag for act(...) to work outside react-dom/test-utils.
// Set once here so every test file gets it without per-file boilerplate.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// Hermetic-suite guard: an un-mocked network call fails fast with a descriptive
// error instead of attempting a real connection (which is slow, noisy, and
// environment-dependent). Tests that need fetch assign globalThis.fetch = mock
// and bypass this entirely; it returns a rejected promise, matching real fetch
// semantics for unreachable hosts.
const GUARD_FETCH: typeof fetch = (input, init) =>
  Promise.reject(
    new Error(
      `Test attempted a real network call to ${String(input)} — mock globalThis.fetch in this test file.`,
    ),
  );

globalThis.fetch = GUARD_FETCH;
