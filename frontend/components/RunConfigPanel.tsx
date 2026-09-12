"use client";

import { useEffect, useState } from "react";
import { BudgetHealth, ByokCredentials, fetchBudgetHealth, fetchProviderHealth } from "@/lib/api";
import { FAILURE_DEMO_TOPIC } from "@/lib/constants";
import { ProviderHealthEntry } from "@/lib/types";

const BYOK_PROVIDERS = ["groq", "ollama_cloud", "openrouter", "gemini"];

interface Props {
  running: boolean;
  onStart: (
    topic: string,
    opts: { forceRerun: boolean; evaluate: boolean; byok?: ByokCredentials }
  ) => void;
  onCancel: () => void;
}

export function RunConfigPanel({ running, onStart, onCancel }: Props) {
  const [topic, setTopic] = useState("");
  const [forceRerun, setForceRerun] = useState(false);
  const [evaluate, setEvaluate] = useState(false);
  const [providers, setProviders] = useState<ProviderHealthEntry[] | null>(null);
  const [providersError, setProvidersError] = useState<string | null>(null);
  const [budgetHealth, setBudgetHealth] = useState<BudgetHealth | null>(null);
  const [byokProvider, setByokProvider] = useState(BYOK_PROVIDERS[0]);
  const [byokApiKey, setByokApiKey] = useState("");

  useEffect(() => {
    fetchProviderHealth()
      .then(setProviders)
      .catch((e) => setProvidersError(e instanceof Error ? e.message : String(e)));
    fetchBudgetHealth().then(setBudgetHealth).catch(() => setBudgetHealth(null));
  }, []);

  const byok: ByokCredentials | undefined =
    budgetHealth?.exceeded && byokApiKey.trim()
      ? { provider: byokProvider, apiKey: byokApiKey.trim() }
      : undefined;

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!topic.trim() || running) return;
    onStart(topic.trim(), { forceRerun, evaluate, byok });
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

      {budgetHealth?.daily_budget_usd != null && (
        <div
          className={`rounded-md border px-3 py-2 text-xs font-telemetry ${
            budgetHealth.exceeded ? "border-coral text-coral" : "border-border-0 text-ink-1"
          }`}
        >
          Daily budget: ${budgetHealth.spent_today_usd.toFixed(2)} / $
          {budgetHealth.daily_budget_usd.toFixed(2)} spent
          {budgetHealth.exceeded && " — exhausted"}
        </div>
      )}

      <details className="text-sm" open={budgetHealth?.exceeded}>
        <summary className="cursor-pointer text-ink-1">Bring your own key</summary>
        <div className="flex flex-col gap-2 mt-2">
          <p className="text-xs text-ink-2">
            {budgetHealth?.exceeded
              ? "The deployment's daily budget is exhausted — supply your own key to keep going."
              : "Optional: route this run through your own key instead of the deployment's."}
          </p>
          <select
            value={byokProvider}
            onChange={(e) => setByokProvider(e.target.value)}
            className="rounded-md border border-border-0 bg-surface-0 px-2 py-1.5 text-sm text-ink-0"
          >
            {BYOK_PROVIDERS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
          <input
            type="password"
            value={byokApiKey}
            onChange={(e) => setByokApiKey(e.target.value)}
            placeholder="API key"
            autoComplete="off"
            className="rounded-md border border-border-0 bg-surface-0 px-2 py-1.5 text-sm text-ink-0 placeholder:text-ink-2"
          />
        </div>
      </details>

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
