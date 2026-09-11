from crew.research_crew import PipelineStageError
from observability.errors import ErrorCode


def test_pipeline_stage_error_defaults_to_internal_error_code():
    err = PipelineStageError("planner", "boom")
    assert err.error_code == ErrorCode.INTERNAL
    assert err.stage == "planner"
    assert err.detail == "boom"


def test_pipeline_stage_error_accepts_explicit_error_code():
    err = PipelineStageError("researcher", "rate limited", ErrorCode.PROVIDER_RATE_LIMIT)
    assert err.error_code == ErrorCode.PROVIDER_RATE_LIMIT
