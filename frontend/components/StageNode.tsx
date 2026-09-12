"use client";

import { useState } from "react";
import { STAGE_LABELS } from "@/lib/constants";
import { NODE_STATE_STYLE } from "@/lib/nodeStateStyle";
import { StageInfo } from "@/lib/pipeline";
import { NodeState, Stage } from "@/lib/types";

function durationLabel(info: StageInfo): string | null {
  if (info.startedAtMs == null) return null;
  const end = info.finishedAtMs ?? Date.now();
  const seconds = (end - info.startedAtMs) / 1000;
  return `${seconds.toFixed(1)}s`;
}

export function StageNode({
  info,
  effectiveState,
}: {
  info: StageInfo;
  effectiveState: NodeState;
}) {
  const [expanded, setExpanded] = useState(false);
  const style = NODE_STATE_STYLE[effectiveState];
  const lastAttempt = info.attempts[info.attempts.length - 1];
  const canExpand = info.attempts.length > 0;
  const duration = durationLabel(info);

  return (
    <div className={`rounded-lg border bg-surface-1 ${style.ring} transition-colors`}>
      <button
        type="button"
        onClick={() => canExpand && setExpanded((v) => !v)}
        className={`w-full flex items-center gap-3 px-4 py-3 text-left ${
          canExpand ? "cursor-pointer" : "cursor-default"
        }`}
      >
        <span className={`inline-block size-2.5 rounded-full shrink-0 ${style.dot}`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-medium text-ink-0">
              {STAGE_LABELS[info.stage as Stage] ?? info.stage}
            </span>
            <span className={`text-xs font-telemetry ${style.text}`}>{style.label}</span>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1 text-xs text-ink-2 font-telemetry">
            {duration && <span>{duration}</span>}
            {lastAttempt?.resolved_deployment && <span>{lastAttempt.resolved_deployment}</span>}
            {info.attempts.some((a) => a.cost_usd != null) && (
              <span>
                ${info.attempts.reduce((sum, a) => sum + (a.cost_usd ?? 0), 0).toFixed(4)} ·{" "}
                {info.attempts.reduce(
                  (sum, a) => sum + (a.prompt_tokens ?? 0) + (a.completion_tokens ?? 0),
                  0
                )}{" "}
                tok
              </span>
            )}
            {info.cacheHits.length > 0 && (
              <span className="rounded bg-teal-dim text-teal px-1.5 py-0.5">
                cache × {info.cacheHits.length}
              </span>
            )}
            {info.guardrails.length > 0 && (
              <span className="rounded bg-amber-dim text-amber px-1.5 py-0.5">
                guardrail × {info.guardrails.length}
              </span>
            )}
          </div>
        </div>
        {canExpand && <span className="text-ink-2 text-xs">{expanded ? "▲" : "▼"}</span>}
      </button>

      {expanded && (
        <div className="border-t border-border-0 px-4 py-3 text-xs font-telemetry text-ink-1 flex flex-col gap-2">
          {info.failDetail && <p className="text-coral">{info.failDetail}</p>}
          {info.attempts.map((a, i) => (
            <div key={i} className="flex flex-wrap gap-x-3 gap-y-0.5">
              <span className="text-ink-2">attempt {a.attempt}</span>
              <span>{a.resolved_deployment ?? a.model_alias}</span>
              <span className={a.message === "ok" || a.message === "cache_hit" ? "text-teal" : "text-coral"}>
                {a.message}
              </span>
              {a.latency_ms != null && <span>{a.latency_ms.toFixed(0)}ms</span>}
              {a.detail && <span className="text-ink-2">{a.detail}</span>}
            </div>
          ))}
          {info.guardrails.map((g, i) => (
            <div key={`g-${i}`} className="text-amber">
              guardrail: {g.message}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
