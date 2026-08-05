import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, info.componentStack);
  }

  handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    if (!this.state.hasError) return this.props.children;

    const error = this.state.error;
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-ink-50 dark:bg-ink-900 p-6">
        <div className="max-w-lg w-full bg-white dark:bg-ink-800 rounded-2xl shadow-2xl p-8 text-center">
          <div className="w-14 h-14 mx-auto rounded-full bg-rose-100 dark:bg-rose-900/40 flex items-center justify-center text-rose-600 dark:text-rose-400">
            <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
           </svg>
         </div>
          <h1 className="mt-5 text-xl font-bold text-ink-900 dark:text-ink-50">
            Something broke
         </h1>
          <p className="mt-2 text-sm text-ink-600 dark:text-ink-300">
            Clay hit an unexpected error. Your API key and sandbox data are safe in localStorage.
         </p>
          {error && (
            <details className="mt-4 text-left">
              <summary className="text-xs text-ink-500 dark:text-ink-400 cursor-pointer hover:text-ink-700 dark:hover:text-ink-200">
                Show error details
             </summary>
              <pre className="mt-2 text-[10px] font-mono text-rose-600 dark:text-rose-400 bg-ink-50 dark:bg-ink-900 rounded p-3 overflow-auto max-h-48">
                {error.name}: {error.message}
                {error.stack && `\n\n${error.stack.split('\n').slice(0, 6).join('\n')}`}
             </pre>
           </details>
          )}
          <div className="mt-6 flex gap-2 justify-center">
            <button
              type="button"
              onClick={this.handleReset}
              className="px-4 py-2 text-sm font-medium text-ink-700 dark:text-ink-200 border border-ink-200 dark:border-ink-700 rounded-lg hover:bg-ink-50 dark:hover:bg-ink-800"
            >
              Try again
           </button>
            <button
              type="button"
              onClick={this.handleReload}
              className="px-4 py-2 text-sm font-medium bg-brand-600 hover:bg-brand-700 text-white rounded-lg"
            >
              Reload page
           </button>
         </div>
       </div>
     </div>
    );
  }
}
