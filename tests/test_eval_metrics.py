from eval.metrics import (
    cache_hit_rate, citation_precision, cost_per_run, fallback_rate,
    stage_latency_percentiles, subquestion_coverage,
)


def test_citation_precision_counts_only_real_urls():
    findings = [
        {"source_url": "https://a.com"},
        {"source_url": "none"},
        {"source_url": ""},
        {"source_url": "https://b.com"},
    ]
    assert citation_precision(findings) == 0.5


def test_citation_precision_returns_none_for_no_findings():
    assert citation_precision([]) is None


def test_subquestion_coverage_basic():
    assert subquestion_coverage(sub_questions_count=4, unresolved_sub_questions_count=1) == 0.75


def test_subquestion_coverage_returns_none_when_no_sub_questions():
    assert subquestion_coverage(0, 0) is None


def test_cache_hit_rate_counts_hits_against_real_completions():
    events = [
        {"event_type": "llm_attempt", "message": "ok", "cache_hit": None},
        {"event_type": "llm_attempt", "message": "cache_hit", "cache_hit": True},
        {"event_type": "cache_hit", "cache_hit": True},
        {"event_type": "llm_attempt", "message": "PROVIDER_RATE_LIMIT"},
    ]
    # 2 hits (llm cache hit + search/subq cache hit) vs 1 real completion
    assert cache_hit_rate(events) == 2 / 3


def test_cache_hit_rate_returns_none_with_no_data():
    assert cache_hit_rate([]) is None


def test_fallback_rate_detects_a_retry_immediately_before_success():
    events = [
        {"event_type": "llm_attempt", "stage": "researcher", "message": "PROVIDER_RATE_LIMIT"},
        {"event_type": "llm_attempt", "stage": "researcher", "message": "ok"},
        {"event_type": "llm_attempt", "stage": "classifier", "message": "ok"},
    ]
    assert fallback_rate(events) == 0.5


def test_fallback_rate_returns_none_with_no_successes():
    assert fallback_rate([{"event_type": "llm_attempt", "message": "PROVIDER_RATE_LIMIT"}]) is None


def test_cost_per_run_sums_only_successful_completions():
    events = [
        {"event_type": "llm_attempt", "message": "ok", "cost_usd": 0.01},
        {"event_type": "llm_attempt", "message": "ok", "cost_usd": 0.02},
        {"event_type": "llm_attempt", "message": "ok", "cost_usd": None},
        {"event_type": "cache_hit", "cost_avoided_usd": 5.0},
    ]
    assert cost_per_run(events) == 0.03


def test_cost_per_run_returns_none_when_no_priced_completions():
    assert cost_per_run([]) is None


def test_stage_latency_percentiles_excludes_cache_hits():
    events = [
        {"event_type": "llm_attempt", "stage": "classifier", "latency_ms": 100.0, "cache_hit": None},
        {"event_type": "llm_attempt", "stage": "classifier", "latency_ms": 200.0, "cache_hit": None},
        {"event_type": "llm_attempt", "stage": "classifier", "latency_ms": 0.0, "cache_hit": True},
    ]
    result = stage_latency_percentiles(events)
    assert "classifier" in result
    assert result["classifier"]["p50"] in (100.0, 200.0)
