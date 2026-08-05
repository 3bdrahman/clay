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

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      [NIM_PROXY_PREFIX]: {
        target: 'https://integrate.api.nvidia.com/v1',
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(new RegExp(`^${NIM_PROXY_PREFIX}`), ''),
        headers: {
          // Strip incoming Origin so NIM doesn't reject the request based on
          // its build.nvidia.com-only CORS policy at the proxy boundary.
          Origin: 'https://integrate.api.nvidia.com',
        },
      },
      [DDG_PROXY_PREFIX]: {
        target: 'https://html.duckduckgo.com',
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(new RegExp(`^${DDG_PROXY_PREFIX}`), ''),
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('recharts')) return 'charts-vendor';
            if (id.includes('react') || id.includes('scheduler')) return 'react-vendor';
            if (id.includes('pdfjs')) return 'pdf-vendor';
            return 'vendor';
          }
          return undefined;
        },
      },
    },
  },
  base: process.env.BASE_PATH || './',
});
