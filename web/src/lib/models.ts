import { NIM_BASE_URL } from './providers';
import type { ModelInfo, Settings, LocalModelPicks } from './types';

export type { ModelInfo };

export type ModelClass = 'tiny' | 'small' | 'medium' | 'large' | 'huge';

export type TaskRole = 'routing' | 'codeGen' | 'answer' | 'eval';

export interface PickedModels {
  routing: string | undefined;
  codeGen: string | undefined;
  answer: string | undefined;
  eval: string | undefined;
  embedding: string | undefined;
}

export class ModelsFetchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelsFetchError';
  }
}

/**
 * Fetch the live model catalog from NVIDIA NIM.
 * @param apiKey - NVIDIA NIM API key (nvapi-...)
 * @returns Array of model info objects with id, ownedBy, created
 * @throws ModelsFetchError if API key missing or request fails
 */
export async function listNimModels(apiKey: string): Promise<ModelInfo[]> {
  if (!apiKey) {
    throw new ModelsFetchError('API key required to fetch model catalog.');
  }
  const resp = await fetch(`${NIM_BASE_URL}/models`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (!resp.ok) {
    throw new ModelsFetchError(`Model catalog fetch failed: ${resp.status} ${resp.statusText}`);
  }
  const json = await resp.json();
  const data: Array<{ id: string; object?: string; created?: number; owned_by?: string }> =
    json.data || [];
  return data.map(m => ({
    id: m.id,
    ownedBy: m.owned_by ?? '',
    created: m.created ?? 0,
  }));
}

/**
 * Fetch the model catalog from a local OpenAI-compatible server.
 * @param baseUrl - Base URL of local server (e.g., http://localhost:11434/v1)
 * @param apiKey - Optional API key for servers that require it
 * @returns Array of model info objects
 * @throws ModelsFetchError if URL empty or request fails
 */
export async function listLocalCatalog(
  baseUrl: string,
  apiKey: string,
): Promise<ModelInfo[]> {
  const url = baseUrl.replace(/\/+$/, '');
  if (!url) {
    throw new ModelsFetchError('Local server URL is empty.');
  }
  const headers: Record<string, string> = {};
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  const resp = await fetch(`${url}/models`, { headers });
  if (!resp.ok) {
    throw new ModelsFetchError(`Local catalog fetch failed: ${resp.status} ${resp.statusText}`);
  }
  const json = await resp.json();
  const data: Array<{ id: string; object?: string; created?: number; owned_by?: string }> =
    json.data || [];
  return data.map(m => ({
    id: m.id,
    ownedBy: m.owned_by ?? '',
    created: m.created ?? 0,
  }));
}

/**
 * Normalize user-provided local model picks into PickedModels.
 * The user-facing LocalModelPicks has only 2 slots (chat, embeddings) — the
 * single chat model fans out into all 4 chat-style roles (routing, codeGen,
 * answer, eval) that the orchestrator/analyzer/eval consume. Empty strings
 * become undefined.
 */
export function pickLocalModels(picks: LocalModelPicks): PickedModels {
  const def = (s: string) => s.trim() || undefined;
  const chat = def(picks.chat);
  const embedding = def(picks.embeddings);
  return {
    routing: chat,
    codeGen: chat,
    answer: chat,
    eval: chat,
    embedding,
  };
}

export interface ResolvedModels {
  catalog: ModelInfo[];
  picked: PickedModels;
  warnings: string[];
}

/**
 * Resolve the final model set for the current session.
 * For NIM: auto-picks best models from catalog. For local: validates user picks against catalog.
 */
export function resolveModels(
  settings: Settings,
  nimCatalog: ModelInfo[],
): ResolvedModels {
  if (settings.provider === 'local') {
    const picked = pickLocalModels(settings.localModels);
    const warnings: string[] = [];
    if (settings.localCatalog.length === 0) {
      warnings.push(
        'Local catalog is empty. Click Discover in Settings to fetch models from the server.',
      );
    } else {
      const catalogIds = new Set(settings.localCatalog.map(m => m.id));
      const userSlots = [
        { key: 'chat', model: picked.routing },
        { key: 'embeddings', model: picked.embedding },
      ] as const;
      for (const { key, model } of userSlots) {
        if (model && !catalogIds.has(model)) {
          warnings.push(
            `${key} model "${model}" is not in the catalog. Re-pick or refresh the catalog.`,
          );
        }
      }
    }
    return { catalog: settings.localCatalog, picked, warnings };
  }
  return { catalog: nimCatalog, picked: pickBestModels(nimCatalog), warnings: [] };
}

function isEmbedding(id: string): boolean {
  return /embed|embedqa/i.test(id);
}

function isCodeSpecialist(id: string): boolean {
  const lower = id.toLowerCase();
  return /codestral|codellama|codegemma|granite.*code|deepseek-coder|nemotron.*code|starcoder|embedcode/.test(
    lower,
  );
}

function isSafetyOrGuard(id: string): boolean {
  return /guard|safety|content-safety|topic-control|reward|parse|translate|detector|calibration|neva|vila|ai-synthetic|cosmo/.test(
    id.toLowerCase(),
  );
}

function isVision(id: string): boolean {
  return /vision|vl$|clip|video|diffusion|deplot|recurrent|cosmos/.test(id.toLowerCase());
}

function isGeneralChat(id: string): boolean {
  if (isEmbedding(id)) return false;
  if (isCodeSpecialist(id)) return false;
  if (isSafetyOrGuard(id)) return false;
  if (isVision(id)) return false;
  return /instruct|chat|^.*\/gpt-|it$|nemotron|moe|reasoning|creative|magistral|laguna|kimi|step-|glm|inkling|palmyra|sea-lion|yi-|zamba|granite|gemma/.test(
    id.toLowerCase(),
  );
}

function inferClass(id: string): ModelClass {
  const lower = id.toLowerCase();
  if (/ultra|550b|340b|253b|122b/.test(lower)) return 'huge';
  if (/120b|90b|72b|70b|^.*large/.test(lower)) return 'large';
  if (/49b|51b|34b|30b|22b|15b|14b|13b|12b|11b/.test(lower)) return 'medium';
  if (/8b|7b|nano/.test(lower)) return 'small';
  if (/mini|4b|3b|2b|1b/.test(lower)) return 'tiny';
  return 'medium';
}

function scoreGeneralChat(model: ModelInfo): number {
  const lower = model.id.toLowerCase();
  let score = 0;
  if (/^meta\/llama-3\.3-/.test(lower)) score += 30;
  if (/^meta\/llama-3\.1-(70b|8b)/.test(lower)) score += 25;
  if (/^mistralai\/mistral-large-2/.test(lower)) score += 28;
  if (/^mistralai\/mistral-7b/.test(lower)) score += 22;
  if (/^nvidia\/nemotron-3-(super|ultra)/.test(lower)) score += 40;
  if (/^nvidia\/nemotron-4-/.test(lower)) score += 35;
  if (/^nvidia\/llama-3\.1-nemotron-(70b|ultra|super)/.test(lower)) score += 30;
  if (/^nvidia\/llama-3\.1-nemotron-nano/.test(lower)) score += 22;
  if (/^openai\/gpt-oss-/.test(lower)) score += 30;
  if (/^writer\/palmyra/.test(lower)) score += 20;
  if (/^stepfun-ai\/step-/.test(lower)) score += 18;
  if (/^moonshotai\/kimi-/.test(lower)) score += 25;
  if (/^z-ai\/glm-/.test(lower)) score += 22;
  if (/^deepseek-ai\/deepseek-v/.test(lower)) score += 28;
  if (/^google\/gemma-3-(12b|4b)/.test(lower)) score += 18;
  if (/^google\/gemma-4-/.test(lower)) score += 22;
  if (/^ibm\/granite-3\.0-/.test(lower)) score += 15;
  if (/^poolside\/laguna/.test(lower)) score += 18;
  if (/^zyphra\/zamba/.test(lower)) score += 12;
  return score;
}

function scoreCodeSpecialist(model: ModelInfo): number {
  const lower = model.id.toLowerCase();
  let score = 0;
  if (/codestral-22b/.test(lower)) score += 50;
  if (/codestral/.test(lower)) score += 45;
  if (/codellama-70b/.test(lower)) score += 35;
  if (/codellama/.test(lower)) score += 30;
  if (/codegemma/.test(lower)) score += 25;
  if (/deepseek-coder/.test(lower)) score += 28;
  if (/granite.*code/.test(lower)) score += 25;
  if (/starcoder2/.test(lower)) score += 20;
  if (/nemotron.*code/.test(lower)) score += 22;
  if (/embedcode/.test(lower)) score += 0;
  if (/8b/.test(lower)) score += 3;
  if (/15b/.test(lower)) score += 5;
  if (/22b/.test(lower)) score += 8;
  if (/70b/.test(lower)) score += 12;
  if (/34b/.test(lower)) score += 10;
  return score;
}

function scoreEmbedding(model: ModelInfo): number {
  const lower = model.id.toLowerCase();
  let score = 0;
  if (/nv-embedqa-e5/.test(lower)) score += 50;
  if (/nv-embedqa-mistral/.test(lower)) score += 35;
  if (/embedqa/.test(lower)) score += 30;
  if (/nv-embedcode/.test(lower)) score += 25;
  if (/nv-embed-v1/.test(lower)) score += 20;
  if (/llama-nemotron-embed/.test(lower)) score += 25;
  if (/nemotron-3-embed/.test(lower)) score += 22;
  if (/nemoretriever/.test(lower)) score += 18;
  if (/arctic-embed/.test(lower)) score += 15;
  if (/bge-m3/.test(lower)) score += 12;
  if (/embed-qa-4/.test(lower)) score += 5;
  return score;
}

function pickHighest(
  models: ModelInfo[],
  score: (m: ModelInfo) => number,
  classes: ModelClass[],
): ModelInfo | undefined {
  const candidates = models.filter(m => classes.includes(inferClass(m.id)));
  if (candidates.length === 0) return undefined;
  let bestId: string | undefined;
  let bestScore = -Infinity;
  for (const m of candidates) {
    const s = score(m);
    if (s > bestScore) {
      bestScore = s;
      bestId = m.id;
    }
  }
  return candidates.find(m => m.id === bestId);
}

/**
 * Heuristically pick the best model for each task from the NIM catalog.
 * @param models - Full model catalog from NIM
 * @returns PickedModels with routing, codeGen, answer, eval, embedding
 */
export function pickBestModels(models: ModelInfo[]): PickedModels {
  const chats = models.filter(m => isGeneralChat(m.id));
  const codes = models.filter(m => isCodeSpecialist(m.id));
  const embeddings = models.filter(m => isEmbedding(m.id));

  const small = chats.filter(m => inferClass(m.id) === 'small');
  const tiny = chats.filter(m => inferClass(m.id) === 'tiny');
  const large = chats.filter(m => inferClass(m.id) === 'large');
  const huge = chats.filter(m => inferClass(m.id) === 'huge');

  const routing = pickHighest(small, scoreGeneralChat, ['small', 'tiny'])
    ?? pickHighest(tiny, scoreGeneralChat, ['tiny', 'small'])
    ?? small[0]
    ?? tiny[0]
    ?? chats[0];

  const evalCandidates = small.filter(m => m.id !== routing?.id);
  const evalModel = pickHighest(evalCandidates, scoreGeneralChat, ['small', 'tiny'])
    ?? pickHighest(small, scoreGeneralChat, ['small', 'tiny'])
    ?? evalCandidates[0]
    ?? small[0]
    ?? chats[0];

  const codeGen = pickHighest(codes, scoreCodeSpecialist, ['medium', 'large', 'huge', 'small'])
    ?? codes[0];

  const answer = pickHighest(huge, scoreGeneralChat, ['huge', 'large'])
    ?? pickHighest(large, scoreGeneralChat, ['large', 'huge'])
    ?? huge[0]
    ?? large[0]
    ?? chats[0];

  let embedding: ModelInfo | undefined;
  let bestEmbScore = -Infinity;
  for (const m of embeddings) {
    const s = scoreEmbedding(m);
    if (s > bestEmbScore) {
      bestEmbScore = s;
      embedding = m;
    }
  }

  return {
    routing: routing?.id,
    codeGen: codeGen?.id,
    answer: answer?.id,
    eval: evalModel?.id,
    embedding: embedding?.id,
  };
}

/**
 * Get human-readable description of a model class.
 */
export function describeModelClass(cls: ModelClass): string {
  switch (cls) {
    case 'tiny': return 'tiny';
    case 'small': return 'small';
    case 'medium': return 'medium';
    case 'large': return 'large';
    case 'huge': return 'huge';
  }
}

/**
 * Infer the model class (size tier) from a model ID string.
 */
export function modelClass(id: string): ModelClass {
  return inferClass(id);
}
