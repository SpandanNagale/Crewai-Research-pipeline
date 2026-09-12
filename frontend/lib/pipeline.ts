import { DonePayload, ErrorPayload, NodeState, RunEvent, Stage, STAGES } from "./types";

export interface StageInfo {
  stage: Stage;
  nodeState: NodeState;
  startedAtMs?: number;
  finishedAtMs?: number;
  failDetail?: string;
  attempts: RunEvent[];
  cacheHits: RunEvent[];
  guardrails: RunEvent[];
}

export interface RunState {
  status: "idle" | "running" | "done" | "error";
  runId?: string;
  stages: Record<Stage, StageInfo>;
  runLevelEvents: RunEvent[]; // events with no stage yet (e.g. input guardrail)
  totalCostUsd: number;
  totalTokensUsed: number;
  totalTokensSaved: number;
  totalCostAvoidedUsd: number;
  done?: DonePayload;
  error?: ErrorPayload;
}

function emptyStage(stage: Stage): StageInfo {
  return { stage, nodeState: "pending", attempts: [], cacheHits: [], guardrails: [] };
}

export function initialRunState(): RunState {
  return {
    status: "idle",
    stages: Object.fromEntries(STAGES.map((s) => [s, emptyStage(s)])) as Record<Stage, StageInfo>,
    runLevelEvents: [],
    totalCostUsd: 0,
    totalTokensUsed: 0,
    totalTokensSaved: 0,
    totalCostAvoidedUsd: 0,
  };
}

export type RunAction =
  | { type: "start"; runId: string }
  | { type: "event"; event: RunEvent }
  | { type: "done"; payload: DonePayload }
  | { type: "error"; payload: ErrorPayload };

export function runReducer(state: RunState, action: RunAction): RunState {
  switch (action.type) {
    case "start":
      return { ...initialRunState(), status: "running", runId: action.runId };
    case "event":
      return applyEvent(state, action.event);
    case "done":
      return { ...state, status: "done", done: action.payload };
    case "error":
      return { ...state, status: "error", error: action.payload };
  }
}

function isStage(value: string): value is Stage {
  return (STAGES as readonly string[]).includes(value);
}

function applyEvent(state: RunState, event: RunEvent): RunState {
  if (!isStage(event.stage)) {
    if (event.event_type === "run_started") return state;
    return { ...state, runLevelEvents: [...state.runLevelEvents, event] };
  }

  const prev = state.stages[event.stage];
  const next: StageInfo = { ...prev };
  const eventMs = Date.parse(event.timestamp);

  switch (event.event_type) {
    case "stage_started":
      next.nodeState = "running";
      next.startedAtMs = eventMs;
      break;
    case "stage_finished":
      next.nodeState = event.node_state ?? "succeeded";
      next.finishedAtMs = eventMs;
      break;
    case "stage_failed":
      next.nodeState = "failed";
      next.finishedAtMs = eventMs;
      next.failDetail = event.detail ?? undefined;
      break;
    case "llm_attempt":
      next.attempts = [...next.attempts, event];
      break;
    case "cache_hit":
      next.cacheHits = [...next.cacheHits, event];
      break;
    case "guardrail_fired":
      next.guardrails = [...next.guardrails, event];
      break;
  }

  return {
    ...state,
    stages: { ...state.stages, [event.stage]: next },
    totalCostUsd: state.totalCostUsd + (event.cost_usd ?? 0),
    totalTokensUsed: state.totalTokensUsed + (event.prompt_tokens ?? 0) + (event.completion_tokens ?? 0),
    totalTokensSaved: state.totalTokensSaved + (event.tokens_saved ?? 0),
    totalCostAvoidedUsd: state.totalCostAvoidedUsd + (event.cost_avoided_usd ?? 0),
  };
}

/** Stages that never started once the run has ended unsuccessfully render as not_reached,
 * mirroring what the backend itself writes to pipeline_stage_states for the same case
 * (crew/research_crew.py's run_with_meta, on a PipelineStageError). */
export function effectiveNodeState(state: RunState, stage: Stage): NodeState {
  const info = state.stages[stage];
  if (info.nodeState === "pending" && state.status === "error") return "not_reached";
  return info.nodeState;
}
