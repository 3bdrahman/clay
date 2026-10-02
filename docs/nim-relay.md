# NVIDIA NIM relay

NVIDIA's hosted API does not currently provide the CORS headers needed by a browser application. Clay therefore uses a relay for NIM model discovery and chat. The frontend can stay on GitHub Pages; the relay must run separately.

The included [Cloudflare Worker](../workers/nim-proxy.ts) forwards only `GET /v1/models` and `POST /v1/chat/completions` to NVIDIA. It accepts only the configured browser origin, requires the caller's bearer key, limits request bodies to 2 MiB, and rejects chat models outside Clay's [approved NIM shortlist](../web/src/lib/providerModels.config.json). Responses stream through without buffering; upstream status and rate-limit timing are preserved.

## Deploy your relay

You need a Cloudflare account and a NVIDIA API key from [NVIDIA's key settings](https://build.nvidia.com/settings/api-keys). Each user supplies their own NVIDIA key in Clay; do not configure a shared key on the Worker.

1. In [`wrangler.toml`](../wrangler.toml), set `ALLOWED_ORIGIN` to the frontend's **origin**, without a path or trailing slash. For the public demo it is `https://3bdrahman.github.io`, not `https://3bdrahman.github.io/clay/`. Choose a unique Worker name if needed.
2. From the repository root, authenticate and deploy:

   ```bash
   npx wrangler login
   npx wrangler deploy
   ```

3. Copy the Worker URL printed by Wrangler and append `/v1`, for example `https://your-relay.workers.dev/v1`.
4. In Clay Settings, select **NVIDIA NIM**, enter that **NIM relay base URL**, and add your NVIDIA key. The model menu contains only the approved models available in NVIDIA's current catalog.

To check the Worker bundle without deploying it:

```bash
npx wrangler deploy --dry-run
```

The bundled tests run as part of `cd web && npm run verify`. They cover origin restrictions, authentication, approved models, malformed and oversized requests, upstream errors, and streaming.

## Frontend configuration

The public frontend allows Cloudflare `*.workers.dev` relays and localhost connections. For a custom relay domain, include its origin in `VITE_CSP_EXTRA_CONNECT_SRC` when building the frontend, or set `VITE_NIM_BASE_URL` to its full `/v1` base. That build-time setting supplies the default relay and adds its origin to the CSP.

Users can override the default relay in Settings. Only use a relay you control: it receives your NVIDIA key and model request content. The included implementation does not store keys, keep request contents, or log either one. Origin checks are browser access controls, not a substitute for the NVIDIA key required on every forwarded request.

During `npm run dev`, Vite provides `/nim-api/v1` on the same origin and forwards to NVIDIA directly. No Worker deployment is needed for local development. GitHub Pages cannot provide this development proxy.

## Models and access

The NIM shortlist is Nemotron 3 Super 120B and Nemotron 3.5 Lightning 30B. The browser and Worker share the same policy file, so an arbitrary model ID cannot be sent through the relay.

NVIDIA offers [developer access for prototyping and testing](https://docs.api.nvidia.com/nim/docs/product), subject to account credits and quotas. This is not a guarantee of permanent free production use. Clay cannot determine your remaining NVIDIA credits. OpenRouter's separate free-model path enforces zero-price routing; it is not used as an automatic fallback from NIM.

If setup fails, check that the URL ends in `/v1`, the frontend origin exactly matches `ALLOWED_ORIGIN`, the relay domain is permitted by the frontend CSP, and your NVIDIA key is accepted. A `429` response preserves NVIDIA's retry delay; an exhausted quota or invalid key is surfaced rather than hidden by another provider.
