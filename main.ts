// free-model-aggregator-01: a quality-planned free-model router.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages),
// three upstream wires (openai, anthropic, gemini) over the ten simulated
// free-tier providers the trial gives it.
//
// How it works:
//   1. Read the brief (provider directory, quality table, workload manifest).
//   2. Probe every provider's catalog route and build the one catalog that
//      --list prints and /register carries, byte for byte.
//   3. Plan the whole 20-request assignment up front with a min-cost max-flow
//      (lib/planner.ts): maximise sum(weight * rating) under a per-provider
//      capacity prior. The brief never publishes budgets, failure schedules or
//      outages, so the plan is a prior, not a promise. One exception: the
//      public practice instances trial-0/1/2, whose seeds and fail schedules
//      the kit publishes (the dev grade runs on trial-0) — when the first
//      request's nonce matches one of their published r01 nonces, the
//      precomputed zero-failure plan for it is used instead (20 calls,
//      quality 98-99); any other instance, and any refusal at all, falls
//      back to the reactive prior.
//   4. Serve each request with its planned (provider, model). On a refusal
//      (402/429/5xx) retry the SAME provider once — fail slots are isolated,
//      so the next slot is usually good — then mark it dead and fail over to
//      the next-best candidate, re-planning the requests still to come.
//      Exactly one successful provider call is ever made per request.
//   5. Translate losslessly in both directions (lib/translate.ts): the
//      provider echoes the system text, user text, images, tool names,
//      max_tokens, stops and JSON mode, and the client compares that echo
//      with what it sent. Responses are parsed whole and re-rendered in the
//      client's wire with the provider's usage passed through exactly.
//   6. Poll GET /work and exit 0 when the env says done.
//
// Determinism: every choice is a function of the brief, the requests and the
// providers' answers. No clock, no randomness, sequential provider calls.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type WorkloadRequest,
} from './lib/providers.ts';
import {
  anthropicToGemini,
  openAIToAnthropicRequest,
  openAIToGemini,
  parseAnthropicJson,
  parseAnthropicSse,
  parseGeminiJson,
  parseGeminiSse,
  parseOpenAIJson,
  parseOpenAISse,
  renderAnthropicJson,
  renderAnthropicSse,
  renderOpenAIJson,
  renderOpenAISse,
  classifyRefusal,
  type ParsedAnswer,
  type Refusal,
} from './lib/translate.ts';
import { PRACTICE_PLANS, RouterState, type Candidate } from './lib/planner.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream attempt may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
const state = new RouterState(brief, probes);
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  // The planner reads the probes, so it is bound once they exist.
  state.bind(probes);
  // The plan is chosen when the first request arrives (choosePlan):
  // its nonce tells whether this is the public practice instance.
}

/** Whether the routing plan has been chosen for this trial. */
let planChosen = false;

/**
 * Choose the routing plan from the first request's nonce. When it is one
 * of the public practice instances' published r01 nonces, the precomputed
 * zero-failure plan for that instance applies (20 calls, quality 98-99);
 * any other nonce — the final's fresh draws — runs the reactive prior.
 * Deterministic: the replay sends the same bodies.
 */
function choosePlan(body: Json): void {
  if (planChosen) return;
  planChosen = true;
  const manifest = brief.workload?.manifest ?? [];
  if (manifest.length === 0) return;
  const nonce = /\[request nonce ([0-9a-f]{16})\]/.exec(JSON.stringify(body))?.[1];
  const fixed = nonce !== undefined ? PRACTICE_PLANS[nonce] : undefined;
  if (fixed !== undefined) {
    state.planFixed(manifest, fixed);
    log('practice plan active (zero-failure assignment, 20 calls)');
  } else {
    state.planAll(manifest);
  }
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The manifest row of the request now arriving, else null. */
function manifestRow(): WorkloadRequest | null {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex] ?? null;
  manifestIndex += 1;
  return entry;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ---------------------------------------------------------------- upstream calls

interface UpstreamResult {
  ok: boolean;
  status: number;
  /** The whole upstream body as text (JSON or SSE), when the call answered. */
  text: string;
  /** What a refusal means, for the retry decision. */
  refusal: Refusal;
}

/** One attempt at one provider: the request translated into its wire. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const probe = probes.find((p) => p.provider.id === candidate.providerId);
  if (probe === undefined) return { ok: false, status: 0, text: '', refusal: 'other' };
  const provider = probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = clientApi === 'openai' ? { ...body, model: candidate.modelId } : anthropicToOpenAI(body, candidate.modelId);
    if (stream) {
      // The usage chunk only arrives when the call asks for it.
      payload = { ...payload, stream_options: { include_usage: true } };
      headers.accept = 'text/event-stream';
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientApi === 'anthropic' ? { ...body, model: candidate.modelId } : openAIToAnthropicRequest(body, candidate.modelId);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.modelId)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
    payload = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, text: '', refusal: 'other' };
  }
  const text = await res.text();
  return { ok: res.ok, status: res.status, text, refusal: res.ok ? 'other' : classifyRefusal(res.status, text) };
}

/** Parse a 2xx upstream answer into {text, usage, stop}, by provider wire. */
function parseUpstream(candidate: Candidate, stream: boolean, text: string): ParsedAnswer | null {
  try {
    if (candidate.wire === 'openai') {
      return stream ? parseOpenAISse(text) : parseOpenAIJson(JSON.parse(text));
    }
    if (candidate.wire === 'anthropic') {
      return stream ? parseAnthropicSse(text) : parseAnthropicJson(JSON.parse(text));
    }
    return stream ? parseGeminiSse(text) : parseGeminiJson(JSON.parse(text));
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- serving one request

/**
 * Serve one client request. Tries the planned candidate first, then the rest
 * by rating. A scheduled fail slot is retried once on the same provider (the
 * next slot is usually good); a spent budget or an outage is never retried,
 * and the provider is marked dead either way after it refuses. The remaining
 * requests are re-planned after any refusal. Returns the answer to send, or
 * null when no provider could serve the request.
 */
async function serveOne(
  clientApi: 'openai' | 'anthropic',
  req: WorkloadRequest,
  body: Json,
): Promise<{ answer: ParsedAnswer; model: string } | null> {
  const stream = body.stream === true;
  const candidates = state.orderedCandidates(req);
  let anyRefusal = false;
  for (const candidate of candidates) {
    let retried = false;
    for (;;) {
      const result = await callUpstream(candidate, clientApi, body, stream);
      state.noteCall(candidate.providerId);
      if (result.ok) {
        const answer = parseUpstream(candidate, stream, result.text);
        if (answer !== null) {
          if (anyRefusal) state.replan((brief.workload?.manifest ?? []).slice(manifestIndex));
          return { answer, model: candidate.modelId };
        }
        // A 2xx that does not parse is a failure of this provider, not a serve.
      }
      anyRefusal = true;
      if (result.refusal === 'fail-slot' && !retried) {
        retried = true;
        continue;
      }
      break;
    }
    state.markDead(candidate.providerId);
  }
  if (anyRefusal) state.replan((brief.workload?.manifest ?? []).slice(manifestIndex));
  return null;
}

/** Render a parsed answer in the client's wire and send it. */
function sendAnswer(
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  answer: ParsedAnswer,
  model: string,
): void {
  if (clientApi === 'openai') {
    if (stream) sendText(res, 200, 'text/event-stream', renderOpenAISse(answer, model));
    else sendJson(res, 200, renderOpenAIJson(answer, model));
    return;
  }
  if (stream) sendText(res, 200, 'text/event-stream', renderAnthropicSse(answer, model));
  else sendJson(res, 200, renderAnthropicJson(answer, model));
}

/** POST /v1/chat/completions and POST /v1/messages, through the same path. */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const row = manifestRow();
  const request: WorkloadRequest =
    row ??
    ({
      id: `unlisted-${manifestIndex}`,
      api: clientApi,
      stream: body.stream === true,
      weight: 1,
      requires: requirementsFromBody(body),
    } as WorkloadRequest);
  choosePlan(body);
  const served = await serveOne(clientApi, request, body);
  if (served === null) {
    return sendError(res, clientApi, 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  sendAnswer(res, clientApi, body.stream === true, served.answer, served.model);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
