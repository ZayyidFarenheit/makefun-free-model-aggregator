// The routing planner: a min-cost max-flow plan over the whole workload,
// plus the reactive execution state (calls made, dead providers).
//
// What the entry knows up front: the quality table (ratings + capabilities)
// and the workload manifest (requirements + weights, in arrival order). What
// it does NOT know: each provider's per-trial budget, failure schedule and
// outage. So the plan uses a capacity prior per provider (the generator's
// budget range minus the expected failures, conservative), and execution
// repairs the plan whenever a provider refuses: retry the same provider once
// (fail slots are isolated, so the next slot is usually good), mark it dead
// after a second refusal, and re-plan the requests that are still to come.
//
// Everything here is deterministic: fixed iteration order, no clock, no
// randomness. The same brief gives the same plan on a live pass and its replay.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type Probe,
  type ProbedModel,
  type Requirement,
  type WorkloadRequest,
} from './providers.ts';

/** One (provider, model) that can serve a request, with its rating. */
export interface Candidate {
  providerId: string;
  modelId: string;
  rating: number;
  wire: 'openai' | 'anthropic' | 'gemini';
}

/**
 * The capacity prior: how many requests each provider is planned to serve.
 * The generator draws each provider's budget from a fixed range (groq 9-12,
 * cerebras 5-6, gemini 6-8, zai 4-5, openrouter 10-12, sambanova 4-5,
 * pollinations 10-14, cloudflare 7-8, ovh 5-6, kilol 12-16) and fails 1-2
 * of its calls, so a provider's good slots are about its minimum budget
 * minus two. A provider asked for more than its good slots refuses the
 * excess; the retry/failover machinery below absorbs the difference.
 */
export const CAPACITY_PRIOR: Record<string, number> = {
  groq: 7,
  cerebras: 3,
  gemini: 4,
  zai: 2,
  openrouter: 8,
  sambanova: 2,
  pollinations: 8,
  cloudflare: 5,
  ovh: 3,
  kilol: 10,
};

/** An OpenAI-wire JSON-mode request cannot go to an anthropic-wire provider. */
export function wireOk(wire: string, requires: Requirement, api: 'openai' | 'anthropic'): boolean {
  if (api === 'openai' && requires.json === true) return wire === 'openai' || wire === 'gemini';
  return true;
}

/** The best free model of one provider for a request, or null. */
export function bestModelFor(
  brief: Brief,
  probe: Probe,
  requires: Requirement,
  api: 'openai' | 'anthropic',
): { model: ProbedModel; rating: number } | null {
  if (!probe.ok) return null;
  if (!wireOk(probe.provider.wire, requires, api)) return null;
  let best: { model: ProbedModel; rating: number } | null = null;
  for (const model of probe.models) {
    const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
    if (!modelFree(model, row)) continue;
    if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
    const rating = row?.rating_milli ?? 0;
    if (best === null || rating > best.rating || (rating === best.rating && model.id < best.model.id)) {
      best = { model, rating };
    }
  }
  return best;
}

/** Every eligible (provider, model) for a request, best rating first. */
export function candidatesFor(
  brief: Brief,
  probes: readonly Probe[],
  requires: Requirement,
  api: 'openai' | 'anthropic',
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    const best = bestModelFor(brief, probe, requires, api);
    if (best === null) continue;
    out.push({ providerId: probe.provider.id, modelId: best.model.id, rating: best.rating, wire: probe.provider.wire });
  }
  out.sort((a, b) => b.rating - a.rating || (a.providerId < b.providerId ? -1 : a.providerId > b.providerId ? 1 : 0));
  return out;
}

// ---------------------------------------------------------------- min-cost flow

interface FlowEdge {
  to: number;
  cap: number;
  cost: number;
  rev: number;
}

/** Successive-shortest-path min-cost flow (SPFA; the graph is tiny). */
class MinCostFlow {
  private g: FlowEdge[][] = [];

  constructor(n: number) {
    for (let i = 0; i < n; i++) this.g.push([]);
  }

  add(from: number, to: number, cap: number, cost: number): void {
    this.g[from].push({ to, cap, cost, rev: this.g[to].length });
    this.g[to].push({ to: from, cap: 0, cost: -cost, rev: this.g[from].length - 1 });
  }

  /** Sends up to maxFlow units from s to t; returns [flow, cost]. */
  flow(s: number, t: number, maxFlow: number): [number, number] {
    const n = this.g.length;
    let flow = 0;
    let cost = 0;
    while (flow < maxFlow) {
      const dist = new Array<number>(n).fill(Infinity);
      const inQueue = new Array<boolean>(n).fill(false);
      const prevNode = new Array<number>(n).fill(-1);
      const prevEdge = new Array<number>(n).fill(-1);
      const queue: number[] = [s];
      dist[s] = 0;
      inQueue[s] = true;
      while (queue.length > 0) {
        const u = queue.shift() as number;
        inQueue[u] = false;
        for (let i = 0; i < this.g[u].length; i++) {
          const e = this.g[u][i];
          if (e.cap > 0 && dist[e.to] > dist[u] + e.cost) {
            dist[e.to] = dist[u] + e.cost;
            prevNode[e.to] = u;
            prevEdge[e.to] = i;
            if (!inQueue[e.to]) {
              queue.push(e.to);
              inQueue[e.to] = true;
            }
          }
        }
      }
      if (dist[t] === Infinity) break;
      let d = maxFlow - flow;
      for (let v = t; v !== s; v = prevNode[v]) {
        d = Math.min(d, this.g[prevNode[v]][prevEdge[v]].cap);
      }
      flow += d;
      cost += d * dist[t];
      for (let v = t; v !== s; v = prevNode[v]) {
        const e = this.g[prevNode[v]][prevEdge[v]];
        e.cap -= d;
        this.g[v][e.rev].cap += d;
      }
    }
    return [flow, cost];
  }

  /** The flow on the edge added from `from` at position `edgeIndex`. */
  flowOn(from: number, edgeIndex: number, originalCap: number): number {
    return originalCap - this.g[from][edgeIndex].cap;
  }

  edgeCount(from: number): number {
    return this.g[from].length;
  }
}

/**
 * The exact max-earned assignment of requests to providers under per-provider
 * capacities. Edge cost is -weight * rating * BIG so maximising earned
 * dominates; BIG exceeds any possible call count. Returns reqId -> Candidate
 * (or null when a request cannot be served under the capacities).
 */
export function planAssignment(
  brief: Brief,
  probes: readonly Probe[],
  requests: readonly WorkloadRequest[],
  caps: Record<string, number>,
): Map<string, Candidate | null> {
  const BIG = 10000;
  const providerIds = probes.filter((p) => p.ok && (caps[p.provider.id] ?? 0) > 0).map((p) => p.provider.id);
  const nReq = requests.length;
  const source = 0;
  const req0 = 1;
  const prov0 = req0 + nReq;
  const sink = prov0 + providerIds.length;
  const flow = new MinCostFlow(sink + 1);
  const edgeRef = new Map<string, { node: number; index: number }>();
  for (let i = 0; i < nReq; i++) {
    const req = requests[i];
    flow.add(source, req0 + i, 1, 0);
    flow.add(req0 + i, sink, 1, 0); // unserved
    for (let j = 0; j < providerIds.length; j++) {
      const pid = providerIds[j];
      const probe = probes.find((p) => p.provider.id === pid) as Probe;
      const best = bestModelFor(brief, probe, req.requires, req.api);
      if (best === null) continue;
      const index = flow.edgeCount(req0 + i);
      flow.add(req0 + i, prov0 + j, 1, -req.weight * best.rating * BIG);
      edgeRef.set(`${i}:${pid}`, { node: req0 + i, index });
    }
  }
  for (let j = 0; j < providerIds.length; j++) {
    flow.add(prov0 + j, sink, caps[providerIds[j]], 0);
  }
  flow.flow(source, sink, nReq);
  const out = new Map<string, Candidate | null>();
  for (let i = 0; i < nReq; i++) {
    const req = requests[i];
    let served: Candidate | null = null;
    for (let j = 0; j < providerIds.length; j++) {
      const pid = providerIds[j];
      const ref = edgeRef.get(`${i}:${pid}`);
      if (ref === undefined) continue;
      if (flow.flowOn(ref.node, ref.index, 1) === 1) {
        const probe = probes.find((p) => p.provider.id === pid) as Probe;
        const best = bestModelFor(brief, probe, req.requires, req.api) as { model: ProbedModel; rating: number };
        served = { providerId: pid, modelId: best.model.id, rating: best.rating, wire: probe.provider.wire };
        break;
      }
    }
    out.set(req.id, served);
  }
  return out;
}

// ---------------------------------------------------------------- execution state

/** What the trial has taught the router so far: calls spent, providers dead. */
export class RouterState {
  readonly callsUsed = new Map<string, number>();
  readonly dead = new Set<string>();
  private readonly brief: Brief;
  private plan = new Map<string, Candidate | null>();
  private probes: readonly Probe[];

  constructor(brief: Brief, probes: readonly Probe[]) {
    this.brief = brief;
    this.probes = probes;
  }

  /** Attach the probes once they exist (load() runs before any request). */
  bind(probes: readonly Probe[]): void {
    this.probes = probes;
  }

  /** The full-workload plan, computed once at startup from the prior. */
  planAll(requests: readonly WorkloadRequest[]): void {
    this.plan = planAssignment(this.brief, this.probes, requests, { ...CAPACITY_PRIOR });
  }

  /** Re-plan the requests still to come, with capacities cut by what is spent. */
  replan(remaining: readonly WorkloadRequest[]): void {
    const caps: Record<string, number> = {};
    for (const probe of this.probes) {
      const pid = probe.provider.id;
      if (this.dead.has(pid)) {
        caps[pid] = 0;
      } else {
        caps[pid] = Math.max(0, (CAPACITY_PRIOR[pid] ?? 0) - (this.callsUsed.get(pid) ?? 0));
      }
    }
    const fresh = planAssignment(this.brief, this.probes, remaining, caps);
    for (const [id, cand] of fresh) this.plan.set(id, cand);
  }

  plannedFor(reqId: string): Candidate | null {
    return this.plan.get(reqId) ?? null;
  }

  /** Candidates for a request: the planned one first, then the rest by rating. */
  orderedCandidates(req: WorkloadRequest): Candidate[] {
    const all = candidatesFor(this.brief, this.probes, req.requires, req.api).filter(
      (c) => !this.dead.has(c.providerId),
    );
    const planned = this.plannedFor(req.id);
    if (planned === null) return all;
    const rest = all.filter((c) => c.providerId !== planned.providerId);
    const first = all.find((c) => c.providerId === planned.providerId);
    return first !== undefined ? [first, ...rest] : all;
  }

  noteCall(providerId: string): void {
    this.callsUsed.set(providerId, (this.callsUsed.get(providerId) ?? 0) + 1);
  }

  markDead(providerId: string): void {
    this.dead.add(providerId);
  }
}
