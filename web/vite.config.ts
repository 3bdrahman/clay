import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// GitHub Pages deployment: set BASE_PATH=/<repo-name> at build time
// For user.github.io repo, use BASE_PATH=/
const repoName = process.env.GITHUB_REPOSITORY?.split('/')[1] ?? 'clay';
const isGitHubPages = process.env.DEPLOY_TARGET === 'github-pages';
const base = isGitHubPages ? `/${repoName}/` : (process.env.BASE_PATH || './');

function cspPlugin() {
  return {
    name: 'csp-inject',
    transformIndexHtml(html: string, ctx?: { server?: unknown }) {
      // 'unsafe-inline' in script-src is only needed on the dev server (the
      // @react/refresh preamble is an inline module script and HMR evaluates
      // code). The production bundle has no inline scripts (verified in
      // dist/index.html), so builds ship without it. 'unsafe-eval' stays in
      // both modes: Arquero compiles table verbs through code generation and
      // cannot run under a CSP without it; 'wasm-unsafe-eval' covers the
      // QuickJS/transformers.js WASM compilation.
      const isDevServer = Boolean(ctx?.server);
      const scriptSrc = isDevServer
        ? "'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval'"
        : "'self' 'unsafe-eval' 'wasm-unsafe-eval'";

      const extraConnectSrc = process.env.VITE_CSP_EXTRA_CONNECT_SRC?.trim();
      const deployUrl = process.env.VITE_DEPLOY_URL?.trim();
      let deploymentUrl: URL | undefined;

      const connectSrc = [
        "'self'",
        // Local model servers such as LM Studio and vLLM commonly run on
        // loopback origins and handle CORS themselves.
        'http://localhost:*',
        'http://127.0.0.1:*',
        'https://localhost:*',
        'https://127.0.0.1:*',
        'https://openrouter.ai',
        'https://mwmbl.org',
        'https://google.serper.dev',
        // Local embeddings (transformers.js): model weights + ONNX WASM from CDN
        'https://huggingface.co',
        'https://*.huggingface.co',
        'https://*.hf.co',
        'https://cdn.jsdelivr.net',
      ];

      if (deployUrl) {
        try {
          const url = new URL(deployUrl);
          if (url.protocol !== 'https:' && url.protocol !== 'http:') {
            throw new Error('Deployment URL must use http or https');
          }
          connectSrc.push(url.origin);
          url.pathname = `${url.pathname.replace(/\/+$/, '')}/`;
          url.search = '';
          url.hash = '';
          deploymentUrl = url;
        } catch (e) {
          console.warn(
            `[vite] Ignoring invalid VITE_DEPLOY_URL "${deployUrl}":`,
            e instanceof Error ? e.message : e,
          );
        }
      }

      if (extraConnectSrc) {
        connectSrc.push(...extraConnectSrc.split(',').map(s => s.trim()).filter(Boolean));
      }

      const csp = `default-src 'self'; connect-src ${connectSrc.join(' ')}; script-src ${scriptSrc}; worker-src 'self' blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data: blob: https:; manifest-src 'self'; base-uri 'self'; form-action 'self'`;

      let result = html.replace(
        '<meta http-equiv="Content-Security-Policy" content="%CSP%" />',
        `<meta http-equiv="Content-Security-Policy" content="${csp}" />`
      );
      if (deploymentUrl) {
        const imageUrl = new URL('og-image.png', deploymentUrl).href;
        result = result.replace(
          /(<meta (?:property="og:image"|name="twitter:image") content=")[^"]*("\s*\/>)/g,
          (_, prefix: string, suffix: string) => `${prefix}${imageUrl}${suffix}`,
        );
        result = result.replace(
          '<meta property="og:type"',
          `<meta property="og:url" content="${deploymentUrl.href}" />\n    <meta property="og:type"`,
        );
      }
      return result;
    },
  };
}

// Rewrites the %BASE% placeholders in dist/404.html to the resolved
// build-time base. Files in public/ are copied verbatim (transformIndexHtml
// never sees them), so the SPA fallback needs its own build-time pass. For a
// relative './' base the redirect must be site-root-anchored: 404.html served
// at an arbitrary missing depth cannot resolve a relative path back to the
// app root. Absolute bases pass through unchanged.
function base404Plugin() {
  let root = process.cwd();
  let outDir = 'dist';
  return {
    name: 'base-404-rewrite',
    configResolved(config: { root: string; build: { outDir: string } }) {
      root = config.root;
      outDir = config.build.outDir;
    },
    closeBundle() {
      const redirectBase = base === './' ? '/' : base;
      const p = resolve(root, outDir, '404.html');
      const html = readFileSync(p, 'utf8');
      writeFileSync(p, html.replaceAll('%BASE%', redirectBase));
    },
  } satisfies Plugin;
}

export default defineConfig(() => ({
  plugins: [react(), cspPlugin(), base404Plugin()],
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
  base,
}));
