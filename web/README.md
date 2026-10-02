# Clay web application

The React, TypeScript, and Vite application deployed at [3bdrahman.github.io/clay](https://3bdrahman.github.io/clay/).

Use Node.js 22. From this directory:

```bash
npm ci
npm run dev
npm run verify
```

Use OpenRouter's approved free models, NVIDIA NIM with your key and a configured relay, or models installed on a local OpenAI-compatible server. Document embeddings run locally using a fixed model; API credentials are not needed to load files.

Vite provides the NIM development proxy. For static deployments, see the [NIM relay setup](../docs/nim-relay.md).

See the [project README](../README.md) for the demo walkthrough, architecture, data handling, verification, and deployment configuration.
