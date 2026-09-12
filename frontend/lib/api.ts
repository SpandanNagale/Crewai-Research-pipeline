import { GroundednessResponse, ProviderHealthEntry, TraceResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export async function fetchProviderHealth(): Promise<ProviderHealthEntry[]> {
  const res = await fetch(apiUrl("/health/providers"), { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /health/providers failed: ${res.status}`);
  const body = await res.json();
  return body.providers as ProviderHealthEntry[];
}

export interface BudgetHealth {
  daily_budget_usd: number | null;
  spent_today_usd: number;
  remaining_usd: number | null;
  exceeded: boolean;
}

export async function fetchBudgetHealth(): Promise<BudgetHealth> {
  const res = await fetch(apiUrl("/health/budget"), { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /health/budget failed: ${res.status}`);
  return res.json();
}

export async function fetchTrace(questionId: number): Promise<TraceResponse> {
  const res = await fetch(apiUrl(`/questions/${questionId}/trace`), { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /questions/${questionId}/trace failed: ${res.status}`);
  return res.json();
}

export async function fetchGroundedness(questionId: number): Promise<GroundednessResponse> {
  const res = await fetch(apiUrl(`/questions/${questionId}/groundedness`), { cache: "no-store" });
  if (!res.ok) throw new Error(`GET /questions/${questionId}/groundedness failed: ${res.status}`);
  return res.json();
}

export interface ByokCredentials {
  provider: string;
  apiKey: string;
}

export function startResearchStream(
  topic: string,
  opts: {
    forceRerun?: boolean;
    evaluate?: boolean;
    byok?: ByokCredentials;
    signal?: AbortSignal;
  } = {}
): Promise<Response> {
  return fetch(apiUrl("/research/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      force_rerun: opts.forceRerun ?? false,
      evaluate: opts.evaluate ?? false,
      byok_provider: opts.byok?.provider || null,
      byok_api_key: opts.byok?.apiKey || null,
    }),
    signal: opts.signal,
  });
}

/** Pings the backend with a bounded retry, so a cold-starting free-tier
 * instance (Render et al. spin down when idle) shows a "waking up" state
 * instead of the UI just looking broken for up to ~50s. */
export async function waitForBackend(
  opts: { timeoutMs?: number; intervalMs?: number } = {}
): Promise<boolean> {
  const timeoutMs = opts.timeoutMs ?? 60_000;
  const intervalMs = opts.intervalMs ?? 2_000;
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    try {
      const res = await fetch(apiUrl("/health"), { cache: "no-store" });
      if (res.ok) return true;
    } catch {
      // backend not reachable yet -- keep retrying until the deadline
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  return false;
}
