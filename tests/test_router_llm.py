import time

import openai
import pytest

from llm.router_llm import GuardrailRefusalError, RouterLLM


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


def _router(deployments):
    return RouterLLM(model="test-alias", deployments=deployments)


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
