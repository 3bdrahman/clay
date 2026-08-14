import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// NVIDIA NIM only serves Access-Control-Allow-Origin: https://build.nvidia.com.
// To make the 100% client-side app work from any other origin (localhost,
// GitHub Pages, Netlify, etc.) we proxy /nim-api to NIM in dev. Production
// deploys must replicate this via a serverless function or edge proxy
// (Cloudflare Worker, Vercel function, Netlify function, etc.).
const NIM_PROXY_PREFIX = '/nim-api';
// DuckDuckGo HTML search returns no CORS headers. Proxying lets the in-browser
// app reach it. Same caveat as NIM: production needs an edge proxy.
const DDG_PROXY_PREFIX = '/ddg';
// In dev, browser-to-localhost requests are allowed by the CSP above and most
// local servers (Ollama, LM Studio) set permissive CORS. If a user runs a
// stricter local server they can point localServerUrl at any origin they want;
// no Vite proxy is configured for arbitrary localhost targets on purpose.

const proxyConfig = {
  [NIM_PROXY_PREFIX]: {
    target: 'https://integrate.api.nvidia.com/v1',
    changeOrigin: true,
    secure: true,
    rewrite: (path: string) => path.replace(new RegExp(`^${NIM_PROXY_PREFIX}`), ''),
    headers: {
      Origin: 'https://integrate.api.nvidia.com',
    },
  },
  [DDG_PROXY_PREFIX]: {
    target: 'https://html.duckduckgo.com',
    changeOrigin: true,
    secure: true,
    rewrite: (path: string) => path.replace(new RegExp(`^${DDG_PROXY_PREFIX}`), ''),
  },
};

function cspPlugin() {
  return {
    name: 'csp-inject',
    transformIndexHtml(html: string) {
      const extraConnectSrc = process.env.VITE_CSP_EXTRA_CONNECT_SRC?.trim();
      const nimBaseUrl = process.env.VITE_NIM_BASE_URL?.trim();

      const connectSrc = [
        "'self'",
        'http://localhost:*',
        'http://127.0.0.1:*',
        'https://integrate.api.nvidia.com',
        'https://*.nvidia.com',
        'https://duckduckgo.com',
        'https://*.duckduckgo.com',
        'https://google.serper.dev',
      ];

      if (nimBaseUrl && !nimBaseUrl.startsWith('/')) {
        try {
          const url = new URL(nimBaseUrl);
          connectSrc.push(url.origin);
        } catch {
        }
      }

      if (extraConnectSrc) {
        connectSrc.push(...extraConnectSrc.split(',').map(s => s.trim()).filter(Boolean));
      }

      const csp = `default-src 'self'; connect-src ${connectSrc.join(' ')}; script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval'; worker-src 'self' blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data: blob: https:; manifest-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'`;

      return html.replace(
        '<meta http-equiv="Content-Security-Policy" content="%CSP%" />',
        `<meta http-equiv="Content-Security-Policy" content="${csp}" />`
      );
    },
  };
}

export default defineConfig({
  plugins: [react(), cspPlugin()],
  server: {
    proxy: proxyConfig,
  },
  preview: {
    proxy: proxyConfig,
  },
  build: {
    chunkSizeWarningLimit: 1000,
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('recharts')) return 'charts-vendor';
            if (id.includes('react') || id.includes('scheduler')) return 'react-vendor';
            if (id.includes('pdfjs')) return 'pdf-vendor';
            if (id.includes('arquero')) return 'arquero-vendor';
            if (id.includes('marked') || id.includes('dompurify')) return 'markdown-vendor';
            return 'vendor';
          }
          return undefined;
        },
      },
    },
  },
  base: process.env.BASE_PATH || './',
});
