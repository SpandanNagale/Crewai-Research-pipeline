"use client";

import { useEffect, useState } from "react";
import { fetchProviderHealth } from "@/lib/api";
import { FAILURE_DEMO_TOPIC } from "@/lib/constants";
import { ProviderHealthEntry } from "@/lib/types";

interface Props {
  running: boolean;
  onStart: (topic: string, opts: { forceRerun: boolean; evaluate: boolean }) => void;
  onCancel: () => void;
}

export function RunConfigPanel({ running, onStart, onCancel }: Props) {
  const [topic, setTopic] = useState("");
  const [forceRerun, setForceRerun] = useState(false);
  const [evaluate, setEvaluate] = useState(false);
  const [providers, setProviders] = useState<ProviderHealthEntry[] | null>(null);
  const [providersError, setProvidersError] = useState<string | null>(null);

  useEffect(() => {
    fetchProviderHealth()
      .then(setProviders)
      .catch((e) => setProvidersError(e instanceof Error ? e.message : String(e)));
  }, []);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!topic.trim() || running) return;
    onStart(topic.trim(), { forceRerun, evaluate });
  }

  return (
    <aside className="w-full lg:w-80 shrink-0 border-b lg:border-b-0 lg:border-r border-border-0 bg-surface-1 p-5 flex flex-col gap-6">
      <div>
        <h1 className="text-base font-semibold text-ink-0">Research Pipeline</h1>
        <p className="text-xs text-ink-2 mt-1">
          Plan, research, cross-check, write, and critique a topic through a live,
          traceable six-agent pipeline.
        </p>
      </div>

      <form onSubmit={submit} className="flex flex-col gap-3">
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="text-ink-1">Topic</span>
          <textarea
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            rows={3}
            placeholder="e.g. The impact of remote work on urban commercial real estate since 2020"
            className="resize-none rounded-md border border-border-0 bg-surface-0 px-3 py-2 text-sm text-ink-0 placeholder:text-ink-2 focus:outline-none focus:border-teal"
            disabled={running}
          />
        </label>

        <label className="flex items-center gap-2 text-sm text-ink-1">
          <input
            type="checkbox"
            checked={forceRerun}
            onChange={(e) => setForceRerun(e.target.checked)}
            disabled={running}
            className="accent-teal"
          />
          Force re-run (skip knowledge reuse)
        </label>

        <label className="flex items-center gap-2 text-sm text-ink-1">
          <input
            type="checkbox"
            checked={evaluate}
            onChange={(e) => setEvaluate(e.target.checked)}
            disabled={running}
            className="accent-teal"
          />
          Judge-score the report when done
        </label>

        {running ? (
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md border border-coral text-coral text-sm font-medium py-2 hover:bg-coral-dim/40 transition-colors"
          >
            Cancel run
          </button>
        ) : (
          <button
            type="submit"
            disabled={!topic.trim()}
            className="rounded-md bg-teal text-surface-0 text-sm font-semibold py-2 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-teal/90 transition-colors"
          >
            Run pipeline
          </button>
        )}
      </form>

      <button
        type="button"
        disabled={running}
        onClick={() => onStart(FAILURE_DEMO_TOPIC, { forceRerun: true, evaluate: false })}
        className="rounded-md border border-amber text-amber text-sm font-medium py-2 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-amber-dim/40 transition-colors"
        title="Runs a topic with an injected instruction so the input guardrail fires live."
      >
        Watch the guardrail fire (failure demo)
      </button>

      <div className="mt-auto">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2 mb-2">
          Provider status
        </h2>
        {providersError && <p className="text-xs text-coral">{providersError}</p>}
        {!providers && !providersError && <p className="text-xs text-ink-2">Checking…</p>}
        <ul className="flex flex-col gap-1.5">
          {providers?.map((p) => (
            <li
              key={`${p.agent_name}-${p.provider}-${p.model}`}
              className="flex items-center gap-2 text-xs font-telemetry"
              title={p.detail ?? undefined}
            >
              <span
                className={`inline-block size-2 rounded-full shrink-0 ${
                  p.status === "ok" ? "bg-teal" : "bg-coral"
                }`}
              />
              <span className="text-ink-1 truncate">
                {p.agent_name} · {p.provider}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
