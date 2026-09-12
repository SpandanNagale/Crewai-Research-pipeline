import { NodeState } from "./types";

export interface NodeStateStyle {
  label: string;
  dot: string; // background color class for the state dot
  ring: string; // border color class for the node card
  text: string; // text color class for the state label
}

export const NODE_STATE_STYLE: Record<NodeState, NodeStateStyle> = {
  pending: { label: "Pending", dot: "bg-ink-2", ring: "border-border-0", text: "text-ink-2" },
  running: { label: "Running", dot: "bg-amber animate-pulse", ring: "border-amber", text: "text-amber" },
  succeeded: { label: "Succeeded", dot: "bg-teal", ring: "border-teal", text: "text-teal" },
  degraded: { label: "Degraded", dot: "bg-amber", ring: "border-amber", text: "text-amber" },
  failed: { label: "Failed", dot: "bg-coral", ring: "border-coral", text: "text-coral" },
  not_reached: { label: "Not reached", dot: "bg-ink-2", ring: "border-border-0", text: "text-ink-2" },
  skipped_cached: { label: "Cached", dot: "bg-teal", ring: "border-teal-dim", text: "text-teal" },
};
