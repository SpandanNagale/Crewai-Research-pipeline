"""
Guards the empirically-chosen settings.KNOWLEDGE_REUSE_THRESHOLD: no
same-domain-different-question pair should ever clear it, or the pipeline
would silently reuse a cached report for the wrong question. Hits the real
Gemini embedding API (same as eval/calibrate_reuse.py) rather than mocking
it -- the whole point is to catch a real distributional mismatch, which a
fake embedder can't reproduce.
"""
import os

import pytest

from config.settings import settings
from eval.calibrate_reuse import NEAR_MISS_PAIRS, TRUE_MATCH_PAIRS, calibrate

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY") and not settings.GEMINI_API_KEY,
    reason="GEMINI_API_KEY not configured, skipping live embedding calibration test",
)


def _calibrate_or_skip():
    try:
        return calibrate()
    except RuntimeError as e:
        # Free-tier Gemini embedding quota is shared across everything this
        # dev environment does in a given minute -- a 429 here is an
        # infrastructure/quota condition, not a code defect, so it should
        # skip rather than fail a CI run.
        pytest.skip(f"embedding call failed, likely rate/quota limited: {e}")


def test_no_near_miss_pair_exceeds_the_configured_threshold():
    result = _calibrate_or_skip()

    assert result["max_near_miss"] < settings.KNOWLEDGE_REUSE_THRESHOLD, (
        f"A near-miss pair scored {result['max_near_miss']}, at or above "
        f"KNOWLEDGE_REUSE_THRESHOLD ({settings.KNOWLEDGE_REUSE_THRESHOLD}) -- "
        "this would cause the pipeline to silently reuse a cached report "
        "for a genuinely different question."
    )


def test_every_true_match_pair_clears_the_configured_threshold():
    result = _calibrate_or_skip()

    assert result["min_true_match"] >= settings.KNOWLEDGE_REUSE_THRESHOLD, (
        f"A paraphrase pair scored {result['min_true_match']}, below "
        f"KNOWLEDGE_REUSE_THRESHOLD ({settings.KNOWLEDGE_REUSE_THRESHOLD}) -- "
        "the threshold is set too high and would miss legitimate reuse."
    )


def test_calibration_pair_lists_are_non_trivial():
    assert len(TRUE_MATCH_PAIRS) >= 5
    assert len(NEAR_MISS_PAIRS) >= 5
