"use client";

import { useEffect, useState } from "react";
import { PipelinePane } from "./PipelinePane";
import { ReportPane } from "./ReportPane";
import { RunConfigPanel } from "./RunConfigPanel";
import { waitForBackend } from "@/lib/api";
import { useResearchRun } from "@/lib/useResearchRun";

export function Dashboard() {
  const { state, start, cancel } = useResearchRun();
  const [backendAwake, setBackendAwake] = useState(false);
  const [backendUnreachable, setBackendUnreachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    waitForBackend().then((ok) => {
      if (cancelled) return;
      if (ok) setBackendAwake(true);
      else setBackendUnreachable(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!backendAwake) {
    return (
      <div className="flex flex-1 items-center justify-center p-6">
        <div className="text-center font-telemetry text-sm text-ink-1">
          {backendUnreachable ? (
            <p className="text-coral">
              Can&apos;t reach the backend. Confirm it&apos;s running and NEXT_PUBLIC_API_BASE_URL
              is correct, then reload.
            </p>
          ) : (
            <p>
              Waking up the backend
              <span className="animate-pulse">…</span>
              <br />
              <span className="text-ink-2">
                Free-tier deployments spin down when idle — this can take up to a minute.
              </span>
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col lg:flex-row flex-1 min-h-0">
      <RunConfigPanel
        running={state.status === "running"}
        onStart={(topic, opts) => start(topic, opts)}
        onCancel={cancel}
      />
      <PipelinePane run={state} />
      <ReportPane run={state} />
    </div>
  );
}
