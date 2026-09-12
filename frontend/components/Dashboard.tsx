"use client";

import { PipelinePane } from "./PipelinePane";
import { ReportPane } from "./ReportPane";
import { RunConfigPanel } from "./RunConfigPanel";
import { useResearchRun } from "@/lib/useResearchRun";

export function Dashboard() {
  const { state, start, cancel } = useResearchRun();

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
