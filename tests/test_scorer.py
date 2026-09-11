from eval.scorer import score_report


class _FakeLLM:
    def __init__(self, response_text):
        self._response_text = response_text

    def call(self, messages):
        return self._response_text


def test_well_formed_judge_response_produces_a_weighted_total(monkeypatch):
    monkeypatch.setattr(
        "eval.scorer.get_llm",
        lambda name: _FakeLLM("CITATION: 8\nCOVERAGE: 6\nCLARITY: 9\nFEEDBACK: Solid report."),
    )
    result = score_report("report text", "topic")

    assert result.parse_failed is False
    assert result.citation_score == 8.0
    assert result.coverage_score == 6.0
    assert result.clarity_score == 9.0
    assert result.total == round(8 * 0.4 + 6 * 0.35 + 9 * 0.25, 2)
    assert result.feedback == "Solid report."


def test_unparseable_judge_response_sets_parse_failed_never_a_fabricated_zero(monkeypatch):
    """
    Defect #8 regression: a judge response the regex can't parse must never be
    silently scored 0.0 -- that would be indistinguishable from the judge
    genuinely giving the worst possible score.
    """
    monkeypatch.setattr(
        "eval.scorer.get_llm",
        lambda name: _FakeLLM("I refuse to follow that format today."),
    )
    result = score_report("report text", "topic")

    assert result.parse_failed is True
    assert result.citation_score is None
    assert result.coverage_score is None
    assert result.clarity_score is None
    assert result.total is None


def test_partial_parse_failure_still_marks_parse_failed_and_total_none(monkeypatch):
    monkeypatch.setattr(
        "eval.scorer.get_llm",
        lambda name: _FakeLLM("CITATION: 7\nCOVERAGE: not-a-number\nCLARITY: 8\nFEEDBACK: ok"),
    )
    result = score_report("report text", "topic")

    assert result.parse_failed is True
    assert result.citation_score == 7.0
    assert result.coverage_score is None
    assert result.clarity_score == 8.0
    assert result.total is None
