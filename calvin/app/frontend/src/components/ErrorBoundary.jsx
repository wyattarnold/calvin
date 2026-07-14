import { Component } from "react";

/**
 * React error boundary — catches render/lifecycle errors in child components
 * and shows a recovery UI instead of a blank page.
 */
export default class ErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("[ErrorBoundary]", error, info?.componentStack ?? "");
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-center px-6 gap-3">
          <p className="text-red-400 text-sm font-semibold">Something went wrong</p>
          <p className="text-gray-500 text-[11px] font-mono max-w-xs break-all leading-relaxed">
            {this.state.error.message}
          </p>
          <button
            onClick={() => this.setState({ error: null })}
            className="text-xs px-3 py-1 border border-gray-600 rounded text-gray-400 hover:text-gray-200 transition-colors"
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
