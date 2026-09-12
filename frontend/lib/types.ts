// Mirrors observability/events.py::RunEvent and observability/node_state.py::NodeState.
export type NodeState =
  | "pending"
  | "running"
  | "succeeded"
  | "degraded"
  | "failed"
  | "not_reached"
  | "skipped_cached";

export const STAGES = [
  "planner",
  "researcher",
  "evidence_comparator",
  "classifier",
  "writer",
  "critic",
] as const;
export type Stage = (typeof STAGES)[number];

export interface RunEvent {
  run_id: string;
  seq: number;
  timestamp: string;
  stage: string;
  event_type:
    | "run_started"
    | "run_finished"
    | "stage_started"
    | "stage_finished"
    | "stage_failed"
    | "llm_attempt"
    | "cache_hit"
    | "guardrail_fired";
  node_state?: NodeState | null;
  model_alias?: string | null;
  resolved_deployment?: string | null;
  attempt?: number | null;
  latency_ms?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  cost_usd?: number | null;
  cache_hit?: boolean | null;
  tokens_saved?: number | null;
  cost_avoided_usd?: number | null;
  seconds_saved?: number | null;
  score_name?: string | null;
  score_value?: number | null;
  message: string;
  detail?: string | null;
}

export interface DonePayload {
  run_id: string;
  question_id: number | null;
  report: string;
  saved_to: string | null;
  reused: boolean;
  contradictions_count: number;
  findings_count: number;
  groundedness_score: number | null;
  sub_questions_count: number;
  unresolved_sub_questions_count: number;
  trace_url: string | null;
  eval?: {
    citation_score: number | null;
    coverage_score: number | null;
    clarity_score: number | null;
    total: number | null;
    feedback: string;
    parse_failed: boolean;
  };
}

export interface ErrorPayload {
  stage?: string;
  detail: string;
  error_code?: string;
}

export interface ProviderHealthEntry {
  agent_name: string;
  provider: string;
  model: string;
  status: "ok" | "error" | string;
  detail?: string | null;
}

export interface Finding {
  id: number;
  question_id: number;
  source_id: number | null;
  text: string;
  category: string | null;
  sub_question: string | null;
  created_at: string;
  source_url: string | null;
  source_title: string | null;
}

export interface Conclusion {
  id: number;
  text: string;
  finding_ids: number[];
  created_at: string;
}

export interface TraceResponse {
  question: { id: number; topic: string; sub_questions: string[] };
  conclusions: { conclusion: Conclusion; findings: Finding[] }[];
  contradictions: Contradiction[];
}

export interface Contradiction {
  id: number;
  finding_a_id: number;
  finding_b_id: number;
  explanation: string;
  finding_a_text: string | null;
  finding_b_text: string | null;
  sub_question: string | null;
  created_at: string;
}

export interface ClaimGrounding {
  id: number;
  claim_text: string;
  grounded: boolean;
  supporting_finding_id: number | null;
  objection: string | null;
  created_at: string;
}

export interface GroundednessResponse {
  question_id: number;
  claims: ClaimGrounding[];
  threshold: number;
}
