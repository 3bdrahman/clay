// React 19 requires this flag for act(...) to work outside react-dom/test-utils.
// Set once here so every test file gets it without per-file boilerplate.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
