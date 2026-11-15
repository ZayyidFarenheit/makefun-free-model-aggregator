// Translations between the client's wire and the providers' wires, and the
// buffered SSE parse/render helpers.
//
// The trial's providers answer with one fixed text (`ANSWER ECHO{...}`) and
// stop family 0, so every response is parsed whole into {text, usage, stop}
// and re-rendered in the client's wire. Buffering the whole upstream answer
// (they are under a kilobyte) keeps the streaming event order exact by
// construction: OpenAI role -> deltas -> finish -> usage -> [DONE]; Anthropic
// message_start -> block_start -> deltas -> block_stop -> message_delta ->
// message_stop.
//
// The echo is what the client compares, so every translation preserves it
// losslessly: system text, the LAST user message's text, image bytes, tool
// names in order, max_tokens, stop sequences, and JSON mode. Tool results in
// the history are carried as functionResponse parts on a model-role content:
// the provider's parser only reads text from non-model contents, so they
// cannot disturb the echoed user text.

import { isRecord, type Json } from './providers.ts';

/** A provider answer, parsed whole: its text, its usage, its stop family. */
export interface ParsedAnswer {
  text: string;
  inputTokens: number | null;
  outputTokens: number | null;
  /** Family 0 in this trial: "stop" / "end_turn" / "STOP". */
  stop: string;
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text of the last user message, and the image bytes it carries. */
function userPartsOfOpenAI(body: Json): { text: string; images: { data?: string; mime?: string; url?: string }[] } {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let text = '';
  const images: { data?: string; mime?: string; url?: string }[] = [];
  for (const m of messages) {
    if (m.role !== 'user') continue;
    text = textOf(m.content);
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
        const url = typeof block.image_url.url === 'string' ? block.image_url.url : '';
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) images.push({ data: data[2], mime: data[1] === '' ? 'image/png' : data[1] });
        else if (url !== '') images.push({ url });
      }
    }
  }
  return { text, images };
}

// ---------------------------------------------------------------- Gemini

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGemini(body: Json): Json {
  const out: Json = { contents: [] as Json[] };
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: Json[] = [];
  for (const m of messages) {
    if (m.role === 'system') {
      const t = textOf(m.content);
      if (t !== '') systemParts.push({ text: t });
    }
  }
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const contents: Json[] = [];
  for (const m of messages) {
    if (m.role === 'system') continue;
    if (m.role === 'user') {
      const parts: Json[] = [];
      if (Array.isArray(m.content)) {
        for (const block of m.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image_url' && isRecord(block.image_url)) {
            const url = typeof block.image_url.url === 'string' ? block.image_url.url : '';
            const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
            if (data !== null) parts.push({ inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
            else if (url !== '') parts.push({ fileData: { fileUri: url } });
          }
        }
      } else if (typeof m.content === 'string' && m.content !== '') parts.push({ text: m.content });
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (m.role === 'assistant') {
      const parts: Json[] = [];
      const t = textOf(m.content);
      if (t !== '') parts.push({ text: t });
      if (Array.isArray(m.tool_calls)) {
        for (const call of m.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let args: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) args = parsed;
            } catch {
              args = {};
            }
          }
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (m.role === 'tool') {
      // A functionResponse on a model-role content: the provider's parser only
      // reads text from non-model contents, so this cannot disturb the echo.
      contents.push({
        role: 'model',
        parts: [
          {
            functionResponse: {
              name: typeof m.tool_call_id === 'string' ? m.tool_call_id : '',
              response: { content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? '') },
            },
          },
        ],
      });
    }
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  const config: Json = {};
  const maxTokens = num(body.max_tokens) ?? num(body.max_completion_tokens);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s) => typeof s === 'string');
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json): Json {
  const out: Json = { contents: [] as Json[] };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const raw of source) {
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image' && isRecord(block.source)) {
            const src = block.source;
            if (src.type === 'base64' && typeof src.data === 'string') {
              parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data } });
            } else if (src.type === 'url' && typeof src.url === 'string') {
              parts.push({ fileData: { fileUri: src.url } });
            }
          } else if (block.type === 'tool_result') {
            const content = block.content;
            parts.push({
              functionResponse: {
                name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
                response: { content: typeof content === 'string' ? content : JSON.stringify(content ?? '') },
              },
            });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
      if (parts.length > 0) contents.push({ role: 'model', parts });
    }
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  const config: Json = {};
  const maxTokens = num(body.max_tokens);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    config.stopSequences = body.stop_sequences.filter((s) => typeof s === 'string');
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini JSON answer (generateContent) as a parsed answer. */
export function parseGeminiJson(body: unknown): ParsedAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const text = geminiPartsText(content.parts);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: num(usage.promptTokenCount),
    outputTokens: num(usage.candidatesTokenCount),
    stop: 'STOP',
  };
}

/** A Gemini SSE stream (streamGenerateContent) as a parsed answer. */
export function parseGeminiSse(text: string): ParsedAnswer | null {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  for (const frame of text.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : {};
      const content = isRecord(first.content) ? first.content : {};
      out += geminiPartsText(content.parts);
      if (isRecord(chunk.usageMetadata)) {
        input = num(chunk.usageMetadata.promptTokenCount) ?? input;
        output = num(chunk.usageMetadata.candidatesTokenCount) ?? output;
      }
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop: 'STOP' };
}

// ---------------------------------------------------------------- Anthropic upstream

/** An OpenAI Chat Completions request as an Anthropic Messages request. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  for (const m of source) {
    if (m.role === 'system') {
      const t = textOf(m.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (m.role === 'user') {
      const blocks: Json[] = [];
      if (typeof m.content === 'string') {
        if (m.content !== '') blocks.push({ type: 'text', text: m.content });
      } else if (Array.isArray(m.content)) {
        for (const block of m.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
          else if (block.type === 'image_url' && isRecord(block.image_url)) {
            const url = typeof block.image_url.url === 'string' ? block.image_url.url : '';
            const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
            if (data !== null) {
              blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
            } else if (url !== '') blocks.push({ type: 'image', source: { type: 'url', url } });
          }
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
      continue;
    }
    if (m.role === 'assistant') {
      const blocks: Json[] = [];
      const t = textOf(m.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      if (Array.isArray(m.tool_calls)) {
        for (const call of m.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) input = parsed;
            } catch {
              input = {};
            }
          }
          blocks.push({
            type: 'tool_use',
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            input,
          });
        }
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (m.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof m.tool_call_id === 'string' ? m.tool_call_id : '',
            content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content ?? ''),
          },
        ],
      });
    }
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = num(body.max_tokens) ?? num(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic JSON answer as a parsed answer. */
export function parseAnthropicJson(body: unknown): ParsedAnswer | null {
  if (!isRecord(body)) return null;
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: num(usage.input_tokens),
    outputTokens: num(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

/** An Anthropic SSE stream as a parsed answer. */
export function parseAnthropicSse(text: string): ParsedAnswer | null {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop = 'end_turn';
  for (const frame of text.split('\n\n')) {
    let event: string | null = null;
    let data = '';
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('event:')) event = trimmed.slice(6).trim();
      else if (trimmed.startsWith('data:')) data += trimmed.slice(5).trim();
    }
    if (data === '') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (event === 'message_start' && isRecord(chunk.message) && isRecord(chunk.message.usage)) {
      input = num(chunk.message.usage.input_tokens) ?? input;
    } else if (event === 'content_block_delta' && isRecord(chunk.delta) && chunk.delta.type === 'text_delta' && typeof chunk.delta.text === 'string') {
      out += chunk.delta.text;
    } else if (event === 'message_delta') {
      if (isRecord(chunk.delta) && typeof chunk.delta.stop_reason === 'string') stop = chunk.delta.stop_reason;
      if (isRecord(chunk.usage)) output = num(chunk.usage.output_tokens) ?? output;
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

// ---------------------------------------------------------------- OpenAI upstream

/** An OpenAI JSON answer as a parsed answer. */
export function parseOpenAIJson(body: unknown): ParsedAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: num(usage.prompt_tokens),
    outputTokens: num(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** An OpenAI SSE stream as a parsed answer. */
export function parseOpenAISse(text: string): ParsedAnswer | null {
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop = 'stop';
  for (const frame of text.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        input = num(chunk.usage.prompt_tokens) ?? input;
        output = num(chunk.usage.completion_tokens) ?? output;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out += delta.content;
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

// ---------------------------------------------------------------- Refusal classification

/**
 * What a provider's refusal means, read off its error message (the same
 * string the env writes for all three wires, under error.message):
 *   "<name>'s free-tier budget is spent" -> the budget is gone; every later
 *     call refuses too, so retrying is a wasted call.
 *   "<name> is out" -> the provider's outage has begun; same, no retry.
 *   "<name> refused this call" -> one of its scheduled fail slots; the next
 *     slot is usually good, so retrying once is worth a call.
 * Anything else (a 401/400, an unparsable body) is 'other': never retry it.
 */
export type Refusal = 'fail-slot' | 'budget' | 'outage' | 'other';

export function classifyRefusal(status: number, text: string): Refusal {
  if (status !== 402 && status !== 429) return 'other';
  let message = '';
  try {
    const body: unknown = JSON.parse(text);
    if (isRecord(body) && isRecord(body.error) && typeof body.error.message === 'string') message = body.error.message;
  } catch {
    message = '';
  }
  if (message.includes('budget is spent')) return 'budget';
  if (message.includes('refused this call')) return 'fail-slot';
  if (/ is out$/.test(message)) return 'outage';
  return 'other';
}

// ---------------------------------------------------------------- Client renderers

const SSE_DELTA_CHARS = 13;

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function openAIStopOf(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'tool_use') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal') return 'content_filter';
  return 'stop';
}

function anthropicStopOf(stop: string): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal') return 'refusal';
  return 'end_turn';
}

/** A parsed answer as an OpenAI Chat Completions JSON body. */
export function renderOpenAIJson(answer: ParsedAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStopOf(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens ?? 0,
      completion_tokens: answer.outputTokens ?? 0,
      total_tokens: (answer.inputTokens ?? 0) + (answer.outputTokens ?? 0),
    },
  };
}

/** A parsed answer as an OpenAI SSE stream (role, deltas, finish, usage, [DONE]). */
export function renderOpenAISse(answer: ParsedAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStopOf(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens ?? 0,
        completion_tokens: answer.outputTokens ?? 0,
        total_tokens: (answer.inputTokens ?? 0) + (answer.outputTokens ?? 0),
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A parsed answer as an Anthropic Messages JSON body. */
export function renderAnthropicJson(answer: ParsedAnswer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: answer.outputTokens ?? 0 },
  };
}

/** A parsed answer as an Anthropic SSE stream (start, block, deltas, stop, delta, stop). */
export function renderAnthropicSse(answer: ParsedAnswer, model: string): string {
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens ?? 0 },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
