"""
Pure metric functions over one run's collected RunEvents (as plain dicts --
the harness gets these from /research/stream's SSE payloads, already
RunEvent.model_dump(mode="json") shaped) and the pipeline's own result dict.
No network/DB access here, so these are cheaply unit-testable and reusable
between the live harness and any future offline analysis.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from eval.stats import percentile


def citation_precision(findings: Sequence[dict]) -> Optional[float]:
    """Fraction of findings with a real, resolvable source URL (not empty/"none")."""
    if not findings:
        return None
    cited = sum(1 for f in findings if (f.get("source_url") or "").strip().lower() not in ("", "none"))
    return cited / len(findings)


def subquestion_coverage(sub_questions_count: int, unresolved_sub_questions_count: int) -> Optional[float]:
    if sub_questions_count <= 0:
        return None
    resolved = sub_questions_count - unresolved_sub_questions_count
    return max(0.0, resolved / sub_questions_count)


def cache_hit_rate(events: Sequence[dict]) -> Optional[float]:
    """
    hits / (hits + real completions), across all three cache layers (search,
    sub-question semantic, LLM response). Search/sub-question cache MISSES
    aren't independently instrumented as events (a miss just falls through to
    a normal llm_attempt/search call), so this undercounts the true
    denominator for those two layers -- it's precise for the LLM response
    cache and a documented lower-bound approximation for the other two, never
    a fabricated precise-looking number.
    """
    hits = sum(1 for e in events if e.get("cache_hit") is True)
    real_completions = sum(
        1 for e in events if e.get("event_type") == "llm_attempt" and e.get("message") == "ok"
    )
    denom = hits + real_completions
    if denom == 0:
        return None
    return hits / denom


def fallback_rate(events: Sequence[dict]) -> Optional[float]:
    """
    Fraction of successful LLM calls (llm_attempt, message == "ok") that were
    immediately preceded by a failed llm_attempt for the same stage -- i.e.
    RouterLLM had to retry or advance to another deployment before it
    succeeded. Approximate (no explicit "logical call" boundary is emitted),
    but conservative: it only counts a fallback when a failure is directly
    adjacent to the success in the event stream.
    """
    llm_events = [e for e in events if e.get("event_type") == "llm_attempt" and not e.get("cache_hit")]
    successes = [e for e in llm_events if e.get("message") == "ok"]
    if not successes:
        return None
    fallbacks = 0
    for i, e in enumerate(llm_events):
        if e.get("message") != "ok":
            continue
        prev = llm_events[i - 1] if i > 0 else None
        if prev is not None and prev.get("stage") == e.get("stage") and prev.get("message") != "ok":
            fallbacks += 1
    return fallbacks / len(successes)


def cost_per_run(events: Sequence[dict]) -> Optional[float]:
    costs = [
        e["cost_usd"] for e in events
        if e.get("event_type") == "llm_attempt" and e.get("message") == "ok" and e.get("cost_usd") is not None
    ]
    if not costs:
        return None
    return sum(costs)


def stage_latency_percentiles(events: Sequence[dict]) -> Dict[str, Dict[str, float]]:
    """{stage: {"p50": ms, "p95": ms}} over real (non-cache-hit) llm_attempt latencies."""
    by_stage: Dict[str, List[float]] = {}
    for e in events:
        if e.get("event_type") != "llm_attempt" or e.get("cache_hit"):
            continue
        latency = e.get("latency_ms")
        if latency is None:
            continue
        by_stage.setdefault(e.get("stage", ""), []).append(latency)

    return {
        stage: {"p50": percentile(values, 50), "p95": percentile(values, 95)}
        for stage, values in by_stage.items()
        if values
    }
