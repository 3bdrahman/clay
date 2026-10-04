# Local Model Setup

Clay can use a local OpenAI-compatible chat server when the browser can reach it and the server allows Clay's page origin. The URL in Settings should be the API base ending in `/v1`, for example `http://127.0.0.1:11434/v1` for Ollama or `http://127.0.0.1:1234/v1` for LM Studio.

Use the page origin shown in Clay's Settings when configuring CORS. For the hosted demo, the origin is `https://3bdrahman.github.io`; the `/clay/` path is not part of the origin.

## Checklist

1. Start the model server and load at least one chat model.
2. Enter the server's `/v1` base URL in Clay Settings.
3. Allow CORS for Clay's exact origin.
4. Click **Discover** and pick a returned model.
5. If Chrome or Edge asks for local-network or loopback access, allow it for the Clay site.

Use `127.0.0.1` if `localhost` resolves to IPv6 on your machine and the server only listens on IPv4. Do not enter `0.0.0.0` in Clay; it is a server bind address, not a browser destination. Do not use browser security flags, `no-cors` fetch mode, or a public proxy for local model traffic.

## Ollama

Ollama's OpenAI-compatible endpoint is usually:

```bash
http://127.0.0.1:11434/v1
```

Set `OLLAMA_ORIGINS` to Clay's exact origin, then restart the Ollama process. Restart matters because a running Ollama service will not inherit a new environment variable from a later shell.

For a one-off shell:

```bash
OLLAMA_ORIGINS='http://localhost:5173' ollama serve
```

For the hosted demo:

```bash
OLLAMA_ORIGINS='https://3bdrahman.github.io' ollama serve
```

On Linux with systemd, create an override for the Ollama service:

```bash
sudo systemctl edit ollama
```

Add:

```ini
[Service]
Environment="OLLAMA_ORIGINS=https://3bdrahman.github.io"
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

On macOS, quit Ollama, run the following command, and reopen the Ollama app:

```bash
launchctl setenv OLLAMA_ORIGINS 'https://3bdrahman.github.io'
```

On Windows, quit Ollama and add the user environment variable `OLLAMA_ORIGINS` with value `https://3bdrahman.github.io`, then reopen Ollama. For a foreground PowerShell session:

```powershell
$env:OLLAMA_ORIGINS='https://3bdrahman.github.io'
ollama serve
```

Ensure the old Ollama process has stopped before launching another on the same port.

## LM Studio

LM Studio's OpenAI-compatible endpoint is usually:

```bash
http://127.0.0.1:1234/v1
```

Enable CORS in LM Studio's server settings before discovering models from Clay. If you use the `lms` CLI, start the server with CORS enabled:

```bash
lms server start --cors
```

LM Studio's CORS switch allows requests from websites. Keep the server bound to loopback (`127.0.0.1`) unless LAN access is intentional.

## Other OpenAI-Compatible Servers

vLLM, llama.cpp server, Jan, GPT4All, and similar servers can work when they expose OpenAI-compatible `/models` and `/chat/completions` endpoints under the same `/v1` base URL. Configure their CORS settings to allow Clay's exact origin and use a browser-reachable host such as `127.0.0.1` or a trusted LAN hostname.

The hosted demo's content security policy allows HTTP and HTTPS connections to `localhost` and `127.0.0.1`. A different server origin requires a deployment built with that exact origin in `VITE_CSP_EXTRA_CONNECT_SRC`. Browser support for HTTP LAN requests from an HTTPS page varies; using a locally served Clay instance or a trusted HTTPS server avoids that mixed-content boundary. Prefer a loopback hostname over an IPv6 literal because CSP support for literal IPv6 addresses is inconsistent.

Clay does not send OpenRouter or other cloud provider keys to a local endpoint. Clay currently sends local model requests without an `Authorization` header; authentication-protected local endpoints are not supported by this UI.

Official references: [Ollama configuration](https://docs.ollama.com/faq), [LM Studio server options](https://lmstudio.ai/docs/cli/serve/server-start), and [browser local-network access](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Local_network_access).
