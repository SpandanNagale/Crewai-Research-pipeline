import time

import openai
import pytest

from llm import response_cache
from llm.router_llm import GuardrailRefusalError, RouterLLM
from observability.bus import EventBus
from observability.context import clear_run, set_stage, start_run


class FakeDeployment:
    """Stands in for a crewai.LLM instance in tests — no real network calls."""

    def __init__(self, name, script):
        self.name = name
        self._script = list(script)  # list of return values or exceptions to raise, in order
        self.calls = 0
        self.model = name
        self.temperature = 0.3
        self.timeout = None

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        self.calls += 1
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def get_context_window_size(self):
        return 8192

    def supports_stop_words(self):
        return True

    def supports_multimodal(self):
        return False

    def get_token_usage_summary(self):
        return {}

    def supports_function_calling(self):
        return True


def _router(deployments):
    return RouterLLM(model="test-alias", deployments=deployments)


def test_supports_function_calling_delegates_to_primary_deployment():
    """
    Regression test: crewai's executor gates its entire native tool-calling path
    on hasattr(self.llm, "supports_function_calling") — BaseLLM does not declare
    this method, so a RouterLLM without its own definition is silently routed to
    crewai's text-parsed ReAct loop instead of native function calling for every
    tool-using agent.
    """
    d1 = FakeDeployment("a", [])
    router = _router([d1])
    assert hasattr(router, "supports_function_calling")
    assert callable(router.supports_function_calling)
    assert router.supports_function_calling() is True


def test_first_deployment_succeeds_immediately():
    d1 = FakeDeployment("a", ["final answer"])
    router = _router([d1])
    assert router.call("hi") == "final answer"
    assert d1.calls == 1


def test_timeout_retries_same_deployment_once_then_advances():
    timeout_err = openai.APITimeoutError(request=None)
    d1 = FakeDeployment("a", [timeout_err, timeout_err])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    result = router.call("hi")
    assert result == "from b"
    assert d1.calls == 2  # one retry on the same deployment
    assert d2.calls == 1


def test_rate_limit_advances_immediately_without_retry():
    rl_err = openai.RateLimitError(
        "rate limited", response=_fake_response(429), body=None
    )
    d1 = FakeDeployment("a", [rl_err])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    result = router.call("hi")
    assert result == "from b"
    assert d1.calls == 1  # no retry on the same deployment


def test_unrecognized_status_error_advances_immediately_without_retry():
    """
    Regression test for a live failure: Groq returns HTTP 413 ("tokens per minute
    limit exceeded") for an over-quota request. openai's SDK only maps
    400/401/403/404/409/422/429/5xx to specific exception subclasses (see
    openai._client.OpenAI._make_status_error) — 413 falls through to the bare
    APIStatusError base class, which is not RateLimitError or any other specific
    type. Before this was added to _IMMEDIATE_ADVANCE_EXCEPTIONS, such an error
    propagated out of RouterLLM.call() uncaught on the very first attempt,
    skipping every other configured deployment entirely.
    """
    status_err = openai.APIStatusError(
        "request too large", response=_fake_response(413), body=None
    )
    d1 = FakeDeployment("a", [status_err])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    result = router.call("hi")
    assert result == "from b"
    assert d1.calls == 1  # no retry on the same deployment


def test_bad_request_error_is_not_treated_as_advance_worthy():
    """
    A deterministic client-request error (400) already has its own specific
    openai.BadRequestError subclass and is not in any exception tuple — retrying a
    different deployment won't fix a malformed request, so it must propagate
    uncaught rather than cycling through every deployment for an identical error.
    """
    bad_request = openai.BadRequestError(
        "malformed request", response=_fake_response(400), body=None
    )
    d1 = FakeDeployment("a", [bad_request])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    with pytest.raises(openai.BadRequestError):
        router.call("hi")
    assert d2.calls == 0  # never fell over to the next deployment


def _crewai_wrapped(outer_type, inner):
    """Mimic crewai's native client: `raise outer_type(msg) from inner`."""
    try:
        raise outer_type(f"wrapped: {inner}") from inner
    except outer_type as e:
        return e


def test_crewai_wrapped_not_found_advances_to_next_deployment():
    """
    Regression test for a live failure: crewai's native OpenAI-compatible client
    re-raises openai.NotFoundError (e.g. an OpenRouter free model that was pulled)
    as a bare ValueError. RouterLLM used to let that escape on the first
    deployment, so the Researcher never reached its groq/ollama/gemini fallbacks
    and every sub-question ended up unresolved.
    """
    not_found = openai.NotFoundError("no endpoints", response=_fake_response(404), body=None)
    d1 = FakeDeployment("a", [_crewai_wrapped(ValueError, not_found)])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    assert router.call("hi") == "from b"
    assert d1.calls == 1


def test_crewai_wrapped_connection_error_retries_then_advances():
    conn_err = _crewai_wrapped(ConnectionError, openai.APIConnectionError(request=None))
    d1 = FakeDeployment("a", [conn_err, conn_err])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    assert router.call("hi") == "from b"
    assert d1.calls == 2
    assert d2.calls == 1


def test_unrelated_value_error_still_propagates():
    d1 = FakeDeployment("a", [ValueError("genuine bug")])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    with pytest.raises(ValueError, match="genuine bug"):
        router.call("hi")
    assert d2.calls == 0


def test_failed_deployment_cools_down_across_calls():
    rl_err = openai.RateLimitError(
        "rate limited", response=_fake_response(429), body=None
    )
    d1 = FakeDeployment("a", [rl_err, "a is back"])
    d2 = FakeDeployment("b", ["from b", "from b again"])
    router = _router([d1, d2])

    first = router.call("hi")
    assert first == "from b"

    second = router.call("hi again")
    assert second == "from b again"
    assert d1.calls == 1  # d1 stayed in cooldown, never retried on the second call


def test_refusal_is_raised_uncaught_no_retry_no_fallback():
    d1 = FakeDeployment("a", ["I cannot assist with that request."])
    d2 = FakeDeployment("b", ["from b"])
    router = _router([d1, d2])
    with pytest.raises(GuardrailRefusalError):
        router.call("hi")
    assert d1.calls == 1
    assert d2.calls == 0  # never fell over to the next deployment


def test_all_deployments_exhausted_raises_last_error():
    auth_err = openai.AuthenticationError(
        "bad key", response=_fake_response(401), body=None
    )
    d1 = FakeDeployment("a", [auth_err])
    d2 = FakeDeployment("b", [auth_err])
    router = _router([d1, d2])
    with pytest.raises(openai.AuthenticationError):
        router.call("hi")


def _fake_response(status_code):
    import httpx

    request = httpx.Request("POST", "https://example.invalid")
    return httpx.Response(status_code, request=request)


def _drain(run_id):
    q = EventBus.instance().queue_sink.get_queue(run_id)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def test_successful_call_emits_llm_attempt_event_with_resolved_deployment():
    EventBus.reset_for_tests()
    start_run("run-events-1")
    set_stage("researcher")
    d1 = FakeDeployment("gpt-oss:120b-cloud", ["final answer"])
    router = RouterLLM(model="researcher", deployments=[d1], aliases=["ollama_cloud"])
    router.call("hi")
    clear_run()

    events = [e for e in _drain("run-events-1") if e.event_type == "llm_attempt"]
    assert len(events) == 1
    assert events[0].resolved_deployment == "ollama_cloud:gpt-oss:120b-cloud"
    assert events[0].attempt == 0
    assert events[0].message == "ok"


def test_fallback_emits_one_llm_attempt_event_per_attempt():
    EventBus.reset_for_tests()
    start_run("run-events-2")
    set_stage("researcher")
    rl_err = openai.RateLimitError("rate limited", response=_fake_response(429), body=None)
    d1 = FakeDeployment("a", [rl_err])
    d2 = FakeDeployment("b", ["from b"])
    router = RouterLLM(model="researcher", deployments=[d1, d2], aliases=["groq", "openrouter"])
    router.call("hi")
    clear_run()

    events = [e for e in _drain("run-events-2") if e.event_type == "llm_attempt"]
    assert len(events) == 2
    assert events[0].resolved_deployment == "groq:a"
    assert events[0].message == "PROVIDER_RATE_LIMIT"
    assert events[1].resolved_deployment == "openrouter:b"
    assert events[1].message == "ok"


def test_classifier_call_is_cached_and_replayed_without_calling_a_second_deployment():
    EventBus.reset_for_tests()
    response_cache.reset_for_tests()
    start_run("run-cache-hit")
    set_stage("classifier")

    d1 = FakeDeployment("a", ["result-1"])
    router1 = RouterLLM(model="classifier", deployments=[d1], aliases=["groq"])
    assert router1.call("same messages") == "result-1"

    d2 = FakeDeployment("a", [])  # would raise IndexError if ever called
    router2 = RouterLLM(model="classifier", deployments=[d2], aliases=["groq"])
    assert router2.call("same messages") == "result-1"
    assert d2.calls == 0

    events = [e for e in _drain("run-cache-hit") if e.event_type == "llm_attempt"]
    assert events[-1].cache_hit is True
    clear_run()


def test_comparator_call_is_cached_but_a_different_prompt_is_not_a_hit():
    response_cache.reset_for_tests()
    d1 = FakeDeployment("a", ["first"])
    RouterLLM(model="evidence_comparator", deployments=[d1], aliases=["groq"]).call("prompt A")

    d2 = FakeDeployment("a", ["second"])
    result = RouterLLM(model="evidence_comparator", deployments=[d2], aliases=["groq"]).call("prompt B")
    assert result == "second"
    assert d2.calls == 1


def test_writer_alias_is_never_cached():
    response_cache.reset_for_tests()
    d1 = FakeDeployment("a", ["v1"])
    RouterLLM(model="writer", deployments=[d1], aliases=["groq"]).call("same messages")

    d2 = FakeDeployment("a", ["v2"])
    result = RouterLLM(model="writer", deployments=[d2], aliases=["groq"]).call("same messages")
    assert result == "v2"
    assert d2.calls == 1


def test_missing_aliases_falls_back_to_index_without_crashing():
    EventBus.reset_for_tests()
    start_run("run-events-3")
    set_stage("planner")
    d1 = FakeDeployment("a", ["final answer"])
    router = RouterLLM(model="planner", deployments=[d1])  # no aliases passed
    router.call("hi")
    clear_run()

    events = [e for e in _drain("run-events-3") if e.event_type == "llm_attempt"]
    assert events[0].resolved_deployment == "0:a"
