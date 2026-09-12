"use client";

import { useState } from "react";
import { StageNode } from "./StageNode";
import { effectiveNodeState, RunState } from "@/lib/pipeline";
import { STAGES } from "@/lib/types";

export function PipelinePane({ run }: { run: RunState }) {
  const [copied, setCopied] = useState(false);

  function copyRunId() {
    if (!run.runId) return;
    navigator.clipboard.writeText(run.runId).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <section className="flex-1 min-w-0 p-5 flex flex-col gap-4 overflow-y-auto">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 font-telemetry text-xs">
          {run.runId && (
            <button
              type="button"
              onClick={copyRunId}
              className="rounded border border-border-0 px-2 py-1 text-ink-1 hover:border-border-1"
              title="Copy run_id"
            >
              run_id: {run.runId.slice(0, 8)}… {copied ? "✓" : "⧉"}
            </button>
          )}
          {run.done?.trace_url && (
            <a
              href={run.done.trace_url}
              target="_blank"
              rel="noreferrer"
              className="rounded border border-teal text-teal px-2 py-1 hover:bg-teal-dim/40"
            >
              view trace →
            </a>
          )}
        </div>
        <div className="font-telemetry text-xs text-ink-1">
          ${run.totalCostUsd.toFixed(4)} · {run.totalTokensUsed} tok
          {run.totalCostAvoidedUsd > 0 && (
            <span className="text-teal"> · ${run.totalCostAvoidedUsd.toFixed(4)} saved</span>
          )}
        </div>
      </div>

      {run.status === "idle" && (
        <p className="text-sm text-ink-2">Start a run from the left to see the pipeline light up.</p>
      )}

      {run.runLevelEvents.length > 0 && (
        <div className="rounded-md border border-amber bg-amber-dim/30 px-3 py-2 text-xs font-telemetry text-amber">
          {run.runLevelEvents.map((e, i) => (
            <div key={i}>{e.message}</div>
          ))}
        </div>
      )}

      {run.error && (
        <div className="rounded-md border border-coral bg-coral-dim/30 px-3 py-2 text-sm text-coral">
          <span className="font-medium">
            {run.error.error_code ? `${run.error.error_code}: ` : ""}
          </span>
          {run.error.detail}
        </div>
      )}

      {run.done?.reused && (
        <div className="rounded-md border border-teal bg-teal-dim/30 px-3 py-2 text-sm text-teal">
          Served from a prior matching run — knowledge reuse, no pipeline stages executed this
          time.
        </div>
      )}

      {!run.done?.reused && (
        <div className="flex flex-col gap-2">
          {STAGES.map((stage) => (
            <StageNode
              key={stage}
              info={run.stages[stage]}
              effectiveState={effectiveNodeState(run, stage)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
