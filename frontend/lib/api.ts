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

export function startResearchStream(
  topic: string,
  opts: { forceRerun?: boolean; evaluate?: boolean; signal?: AbortSignal } = {}
): Promise<Response> {
  return fetch(apiUrl("/research/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      force_rerun: opts.forceRerun ?? false,
      evaluate: opts.evaluate ?? false,
    }),
    signal: opts.signal,
  });
}
