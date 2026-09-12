"use client";

import { useCallback, useReducer, useRef } from "react";
import { ByokCredentials, startResearchStream } from "./api";
import { initialRunState, runReducer } from "./pipeline";
import { parseSseStream } from "./sse";
import { DonePayload, ErrorPayload, RunEvent } from "./types";

export function useResearchRun() {
  const [state, dispatch] = useReducer(runReducer, undefined, initialRunState);
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(
    async (
      topic: string,
      opts: { forceRerun?: boolean; evaluate?: boolean; byok?: ByokCredentials } = {}
    ) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await startResearchStream(topic, { ...opts, signal: controller.signal });
        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
          dispatch({ type: "error", payload: { detail: body.detail ?? `HTTP ${res.status}` } });
          return;
        }

        for await (const frame of parseSseStream(res, controller.signal)) {
          if (frame.eventType === "stage_update") {
            const event = frame.data as RunEvent;
            if (event.event_type === "run_started" && event.stage === "") {
              dispatch({ type: "start", runId: event.run_id });
              continue;
            }
            dispatch({ type: "event", event });
          } else if (frame.eventType === "done") {
            dispatch({ type: "done", payload: frame.data as DonePayload });
          } else if (frame.eventType === "error") {
            dispatch({ type: "error", payload: frame.data as ErrorPayload });
          }
        }
      } catch (e) {
        if (controller.signal.aborted) return;
        dispatch({ type: "error", payload: { detail: e instanceof Error ? e.message : String(e) } });
      }
    },
    []
  );

  const cancel = useCallback(() => abortRef.current?.abort(), []);

  return { state, start, cancel };
}
