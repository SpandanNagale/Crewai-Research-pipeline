import pytest

from eval.stats import (
    bootstrap_ci, paired_bootstrap_ci, percentile, spearman_correlation, weighted_cohens_kappa,
)


def test_bootstrap_ci_point_estimate_matches_the_mean():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = bootstrap_ci(samples, n_resamples=500)
    assert result.point_estimate == 3.0
    assert result.lower <= result.point_estimate <= result.upper


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    samples = [1.0, 5.0, 2.0, 9.0, 3.0]
    a = bootstrap_ci(samples, n_resamples=500, seed=42)
    b = bootstrap_ci(samples, n_resamples=500, seed=42)
    assert a == b


def test_bootstrap_ci_single_sample_returns_a_degenerate_interval():
    result = bootstrap_ci([7.0])
    assert result.point_estimate == result.lower == result.upper == 7.0


def test_bootstrap_ci_requires_at_least_one_sample():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_bootstrap_ci_detects_a_consistent_improvement():
    baseline = [0.5, 0.5, 0.5, 0.5, 0.5]
    candidate = [0.7, 0.7, 0.7, 0.7, 0.7]
    result = paired_bootstrap_ci(baseline, candidate, n_resamples=500)
    assert result.point_estimate == pytest.approx(0.2)
    assert result.lower > 0  # interval excludes 0 -- a real improvement


def test_paired_bootstrap_ci_requires_equal_length_sequences():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0, 2.0], [1.0])


def test_spearman_correlation_perfect_monotonic_relationship():
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    assert spearman_correlation(x, y) == pytest.approx(1.0)


def test_spearman_correlation_perfect_inverse_relationship():
    x = [1, 2, 3, 4, 5]
    y = [50, 40, 30, 20, 10]
    assert spearman_correlation(x, y) == pytest.approx(-1.0)


def test_spearman_correlation_requires_matching_lengths():
    with pytest.raises(ValueError):
        spearman_correlation([1, 2], [1])


def test_weighted_cohens_kappa_perfect_agreement_is_one():
    ratings = [1, 2, 3, 4, 5, 3, 2, 1]
    kappa = weighted_cohens_kappa(ratings, ratings, categories=[1, 2, 3, 4, 5])
    assert kappa == pytest.approx(1.0)


def test_weighted_cohens_kappa_penalizes_large_disagreements_more_than_small():
    categories = [1, 2, 3, 4, 5]
    human = [1, 2, 3, 4, 5]
    near_miss_judge = [1, 2, 3, 4, 4]     # one off-by-one disagreement
    far_miss_judge = [1, 2, 3, 4, 1]      # one off-by-four disagreement

    near_kappa = weighted_cohens_kappa(human, near_miss_judge, categories)
    far_kappa = weighted_cohens_kappa(human, far_miss_judge, categories)
    assert near_kappa > far_kappa


def test_percentile_p50_and_p95():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 50) == pytest.approx(50, abs=2)
    assert percentile(values, 95) == pytest.approx(95, abs=2)


def test_percentile_requires_at_least_one_value():
    with pytest.raises(ValueError):
        percentile([], 50)
