"""
Golden-set eval harness. Runs eval/golden/golden_set.json against a running
instance of the API:

- "adversarial" entries: POST /research, assert the input guardrail returns
  HTTP 400 (see app.py::_pipeline_status_code). Deterministic pass/fail.
- "contradiction_fixture" entries: no HTTP call -- seeds two conflicting
  findings directly via db.repository and runs ResearchCrew's
  evidence_comparator stage in-process, then asserts a contradiction was
  recorded. Deterministic pass/fail. Needs to run in the same process/DB as
  the harness (not just against a remote base_url).
- "easy" / "ambiguous" / "low_resource" entries: POST /research/stream,
  consume the live SSE event stream (same protocol frontend/lib/sse.ts parses),
  and compute this run's metrics via eval.metrics. Statistical, not gated --
  reported in the aggregate, not pass/fail, since there is no verified
  historical baseline yet to compare against (see README's Evaluation
  section).

CI fails the build only on the deterministic checks (an adversarial case
that wasn't blocked, or a contradiction that wasn't detected) -- see
run_golden_set()'s return value and main()'s exit code.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from eval.metrics import (
    cache_hit_rate, citation_precision, cost_per_run, fallback_rate,
    stage_latency_percentiles, subquestion_coverage,
)
from eval.stats import bootstrap_ci
from observability import langfuse_sink

GOLDEN_SET_PATH = Path(__file__).parent / "golden" / "golden_set.json"
RESULTS_JSON_PATH = Path(__file__).parent / "results.json"
RESULTS_MD_PATH = Path(__file__).parent / "results.md"

_RESEARCH_CATEGORIES = {"easy", "ambiguous", "low_resource"}


@dataclass
class EntryResult:
    id: str
    category: str
    passed: Optional[bool]  # None for statistical (ungated) entries
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class HarnessReport:
    entries: List[EntryResult]
    generated_at: float

    @property
    def deterministic_failures(self) -> List[EntryResult]:
        return [e for e in self.entries if e.passed is False]

    @property
    def all_deterministic_checks_passed(self) -> bool:
        return len(self.deterministic_failures) == 0


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def _parse_sse_stream(response: httpx.Response):
    """Yields (event_type, data_dict) for each SSE frame -- mirrors frontend/lib/sse.ts."""
    event_type = "message"
    data_lines: List[str] = []
    for raw_line in response.iter_lines():
        line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
        if line == "":
            if data_lines:
                try:
                    yield event_type, json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    pass
            event_type, data_lines = "message", []
            continue
        if line.startswith("event: "):
            event_type = line[len("event: "):].strip()
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
        # lines starting with ":" are heartbeat comments -- ignore


def _run_research_case(client: httpx.Client, entry: dict) -> EntryResult:
    events: List[dict] = []
    done_payload: Optional[dict] = None
    try:
        with client.stream("POST", "/research/stream", json={"topic": entry["topic"]}, timeout=600) as response:
            for event_type, data in _parse_sse_stream(response):
                if event_type == "stage_update":
                    events.append(data)
                elif event_type == "done":
                    done_payload = data
                elif event_type == "error":
                    return EntryResult(entry["id"], entry["category"], passed=None, error=data.get("detail"))
    except httpx.HTTPError as e:
        return EntryResult(entry["id"], entry["category"], passed=None, error=str(e))

    if done_payload is None:
        return EntryResult(entry["id"], entry["category"], passed=None, error="stream ended with no 'done' event")

    metrics = {
        "groundedness_score": done_payload.get("groundedness_score"),
        "citation_precision": None,  # filled in below if the trace is fetched by the caller
        "subquestion_coverage": subquestion_coverage(
            done_payload.get("sub_questions_count", 0), done_payload.get("unresolved_sub_questions_count", 0)
        ),
        "cache_hit_rate": cache_hit_rate(events),
        "fallback_rate": fallback_rate(events),
        "cost_usd": cost_per_run(events),
        "stage_latency_ms": stage_latency_percentiles(events),
        "findings_count": done_payload.get("findings_count"),
        "contradictions_count": done_payload.get("contradictions_count"),
    }

    question_id = done_payload.get("question_id")
    if question_id is not None:
        try:
            trace = client.get(f"/questions/{question_id}/trace", timeout=30).json()
            findings = [f for c in trace.get("conclusions", []) for f in c.get("findings", [])]
            metrics["citation_precision"] = citation_precision(findings)
        except httpx.HTTPError:
            pass

    run_id = done_payload.get("run_id")
    if run_id and metrics.get("citation_precision") is not None:
        # Groundedness is already attached to the trace by the pipeline itself
        # (crew/research_crew.py's run_finished event) -- citation_precision is
        # only known here, after the fact, from the harness's own trace lookup.
        langfuse_sink.attach_score(run_id, "citation_precision", metrics["citation_precision"])

    return EntryResult(entry["id"], entry["category"], passed=None, metrics=metrics)


def _run_adversarial_case(client: httpx.Client, entry: dict) -> EntryResult:
    try:
        response = client.post("/research", json={"topic": entry["topic"]}, timeout=30)
    except httpx.HTTPError as e:
        return EntryResult(entry["id"], entry["category"], passed=False, error=str(e))

    blocked = response.status_code == 400
    return EntryResult(
        entry["id"], entry["category"], passed=blocked,
        metrics={"status_code": response.status_code},
        error=None if blocked else f"expected HTTP 400 (guardrail-blocked), got {response.status_code}",
    )


def _run_contradiction_fixture(entry: dict) -> EntryResult:
    from crew.research_crew import ResearchCrew
    from db import repository

    crew = ResearchCrew()
    question_id = repository.create_question(f"[golden-fixture] {entry['id']}")

    for finding in (entry["finding_a"], entry["finding_b"]):
        source_id = repository.get_or_create_source(question_id, finding["source_url"], finding["source_title"])
        repository.add_finding(question_id, source_id, finding["text"], entry["sub_question"])

    findings = repository.get_findings_list(question_id)
    try:
        crew._run_comparison_stage(question_id, findings)
    except Exception as e:
        return EntryResult(entry["id"], entry["category"], passed=False, error=str(e))

    contradictions = repository.get_contradictions(question_id)
    detected = len(contradictions) > 0
    return EntryResult(
        entry["id"], entry["category"], passed=detected,
        metrics={"contradictions_detected": len(contradictions)},
        error=None if detected else "Evidence Comparator did not flag the deliberately contradictory findings",
    )


def run_golden_set(base_url: str, golden_path: Path = GOLDEN_SET_PATH, client: Optional[httpx.Client] = None) -> HarnessReport:
    owns_client = client is None
    client = client or httpx.Client(base_url=base_url)
    try:
        results = []
        for entry in load_golden_set(golden_path):
            if entry["category"] == "adversarial":
                results.append(_run_adversarial_case(client, entry))
            elif entry["category"] == "contradiction_fixture":
                results.append(_run_contradiction_fixture(entry))
            elif entry["category"] in _RESEARCH_CATEGORIES:
                results.append(_run_research_case(client, entry))
            else:
                results.append(EntryResult(entry["id"], entry["category"], passed=False, error="unknown category"))
        return HarnessReport(entries=results, generated_at=time.time())
    finally:
        if owns_client:
            client.close()


def aggregate_research_metrics(report: HarnessReport) -> Dict[str, Any]:
    research_entries = [e for e in report.entries if e.category in _RESEARCH_CATEGORIES and not e.error]
    if not research_entries:
        return {}

    def _collect(key):
        return [e.metrics[key] for e in research_entries if e.metrics.get(key) is not None]

    groundedness = _collect("groundedness_score")
    aggregate: Dict[str, Any] = {
        "runs": len(research_entries),
        "mean_groundedness": statistics.mean(groundedness) if groundedness else None,
        "mean_citation_precision": _mean_or_none(_collect("citation_precision")),
        "mean_subquestion_coverage": _mean_or_none(_collect("subquestion_coverage")),
        "mean_cache_hit_rate": _mean_or_none(_collect("cache_hit_rate")),
        "mean_fallback_rate": _mean_or_none(_collect("fallback_rate")),
        "mean_cost_usd_per_run": _mean_or_none(_collect("cost_usd")),
    }
    if len(groundedness) >= 2:
        ci = bootstrap_ci(groundedness, n_resamples=1000)
        aggregate["groundedness_95ci"] = [round(ci.lower, 4), round(ci.upper, 4)]
    return aggregate


def _mean_or_none(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def render_markdown(report: HarnessReport, aggregate: Dict[str, Any]) -> str:
    lines = ["# Eval Harness Results", "", f"Generated at `{time.ctime(report.generated_at)}`.", ""]

    lines.append("## Deterministic checks (gate CI)")
    lines.append("")
    lines.append("| Entry | Category | Passed |")
    lines.append("| :--- | :--- | :---: |")
    for e in report.entries:
        if e.category in _RESEARCH_CATEGORIES:
            continue
        lines.append(f"| {e.id} | {e.category} | {'✅' if e.passed else '❌ ' + (e.error or '')} |")

    lines.append("")
    lines.append("## Research-run metrics (reported, not yet gated -- see README)")
    lines.append("")
    if not aggregate:
        lines.append("_No research-category runs completed successfully._")
    else:
        lines.append(f"- Runs: {aggregate['runs']}")
        lines.append(f"- Mean groundedness: {_fmt(aggregate.get('mean_groundedness'))}"
                      + (f" (95% CI {aggregate['groundedness_95ci']})" if aggregate.get("groundedness_95ci") else ""))
        lines.append(f"- Mean citation precision: {_fmt(aggregate.get('mean_citation_precision'))}")
        lines.append(f"- Mean sub-question coverage: {_fmt(aggregate.get('mean_subquestion_coverage'))}")
        lines.append(f"- Mean cache hit rate: {_fmt(aggregate.get('mean_cache_hit_rate'))}")
        lines.append(f"- Mean fallback rate: {_fmt(aggregate.get('mean_fallback_rate'))}")
        lines.append(f"- Mean cost per run: {_fmt(aggregate.get('mean_cost_usd_per_run'), prefix='$')}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Optional[float], prefix: str = "") -> str:
    return "n/a" if value is None else f"{prefix}{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the research-pipeline golden-set eval harness.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--golden-path", default=str(GOLDEN_SET_PATH))
    args = parser.parse_args()

    report = run_golden_set(args.base_url, Path(args.golden_path))
    aggregate = aggregate_research_metrics(report)

    RESULTS_JSON_PATH.write_text(json.dumps({
        "generated_at": report.generated_at,
        "entries": [e.__dict__ for e in report.entries],
        "aggregate": aggregate,
    }, indent=2), encoding="utf-8")
    RESULTS_MD_PATH.write_text(render_markdown(report, aggregate), encoding="utf-8")

    if not report.all_deterministic_checks_passed:
        print("FAILED deterministic checks:")
        for e in report.deterministic_failures:
            print(f"  - {e.id}: {e.error}")
        return 1

    print(f"All deterministic checks passed. {aggregate.get('runs', 0)} research runs completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
