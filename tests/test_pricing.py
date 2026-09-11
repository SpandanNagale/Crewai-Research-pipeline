from observability.pricing import estimate_cost


def test_free_alias_is_always_zero_regardless_of_model():
    assert estimate_cost("ollama_cloud", "gpt-oss:120b-cloud", 1000, 500) == 0.0


def test_free_suffix_model_is_zero_even_on_a_paid_alias():
    assert estimate_cost("openrouter", "nvidia/nemotron-3-super-120b-a12b:free", 1000, 500) == 0.0


def test_unknown_paid_model_returns_none_not_a_guess():
    assert estimate_cost("groq", "openai/gpt-oss-120b", 1000, 500) is None


def test_missing_token_counts_returns_none():
    assert estimate_cost("ollama_cloud", "gpt-oss:120b-cloud", None, None) is None
