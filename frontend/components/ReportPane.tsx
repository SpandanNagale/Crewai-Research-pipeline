"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchGroundedness, fetchTrace } from "@/lib/api";
import { RunState } from "@/lib/pipeline";
import { Finding, GroundednessResponse, TraceResponse } from "@/lib/types";

interface ReportData {
  questionId: number;
  trace: TraceResponse;
  groundedness: GroundednessResponse;
}

export function ReportPane({ run }: { run: RunState }) {
  const questionId = run.done?.question_id ?? null;
  const [data, setData] = useState<ReportData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (questionId == null) return;
    let cancelled = false;
    Promise.all([fetchTrace(questionId), fetchGroundedness(questionId)])
      .then(([trace, groundedness]) => {
        if (!cancelled) setData({ questionId, trace, groundedness });
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [questionId]);

  const trace = data?.questionId === questionId ? data.trace : null;
  const groundedness = data?.questionId === questionId ? data.groundedness : null;

  const findingsById = useMemo(() => {
    const map = new Map<number, Finding>();
    trace?.conclusions.forEach((c) => c.findings.forEach((f) => map.set(f.id, f)));
    return map;
  }, [trace]);

  if (!run.done) {
    return (
      <section className="w-full lg:w-[26rem] shrink-0 border-t lg:border-t-0 lg:border-l border-border-0 bg-surface-1 p-5">
        <p className="text-sm text-ink-2">The finished report appears here once a run completes.</p>
      </section>
    );
  }

  const score = run.done.groundedness_score;
  const threshold = groundedness?.threshold;

  return (
    <section className="w-full lg:w-[26rem] shrink-0 border-t lg:border-t-0 lg:border-l border-border-0 bg-surface-1 p-5 overflow-y-auto flex flex-col gap-5">
      {score != null && threshold != null && (
        <div
          className={`rounded-md border px-3 py-2 text-sm ${
            score >= threshold ? "border-teal text-teal" : "border-amber text-amber"
          }`}
        >
          Groundedness {score.toFixed(2)} <span className="text-ink-2">/ threshold {threshold}</span>
        </div>
      )}

      <article className="prose-report text-sm text-ink-0 leading-relaxed">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{run.done.report}</ReactMarkdown>
      </article>

      {error && <p className="text-xs text-coral">{error}</p>}

      {trace && trace.contradictions.length > 0 && (
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2 mb-2">
            Contradictions ({trace.contradictions.length})
          </h2>
          <ul className="flex flex-col gap-3">
            {trace.contradictions.map((c) => (
              <li key={c.id} className="rounded-md border border-amber bg-amber-dim/20 p-3 text-xs">
                <p className="text-ink-1">{c.explanation}</p>
                <p className="mt-1 text-ink-2">A: {c.finding_a_text}</p>
                <p className="text-ink-2">B: {c.finding_b_text}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {groundedness && groundedness.claims.length > 0 && (
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-2 mb-2">
            Claims &amp; sources ({groundedness.claims.length})
          </h2>
          <ul className="flex flex-col gap-2">
            {groundedness.claims.map((claim) => {
              const finding = claim.supporting_finding_id
                ? findingsById.get(claim.supporting_finding_id)
                : undefined;
              return (
                <li key={claim.id} className="text-xs border-b border-border-0 pb-2">
                  <div className="flex items-start gap-2">
                    <span
                      className={`mt-0.5 inline-block size-1.5 rounded-full shrink-0 ${
                        claim.grounded ? "bg-teal" : "bg-coral"
                      }`}
                    />
                    <p className="text-ink-1">{claim.claim_text}</p>
                  </div>
                  {finding?.source_url && finding.source_url !== "none" ? (
                    <a
                      href={finding.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="ml-3.5 text-teal underline underline-offset-2"
                    >
                      {finding.source_title || finding.source_url}
                    </a>
                  ) : (
                    !claim.grounded &&
                    claim.objection && <p className="ml-3.5 text-ink-2">{claim.objection}</p>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}
