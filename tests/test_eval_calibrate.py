from eval.calibrate import bucket_score, compute_agreement, load_labels, save_labels


def test_bucket_score_maps_0_to_10_onto_1_to_5():
    assert bucket_score(0.0) == 1
    assert bucket_score(10.0) == 5
    assert bucket_score(5.0) == 3


def test_bucket_score_clamps_to_range():
    assert bucket_score(-5.0) == 1
    assert bucket_score(99.0) == 5


def test_compute_agreement_excludes_unlabeled_or_unparseable_reports():
    labels = {"r1": 5, "r2": 1, "r3": 4}
    judge_totals = {"r1": 9.0, "r2": 1.0, "r3": None}  # r3's judge output failed to parse

    result = compute_agreement(labels, judge_totals)

    assert result["n"] == 2  # r3 excluded, never coerced to a score
    assert "weighted_kappa" in result
    assert "spearman_correlation" in result


def test_compute_agreement_returns_none_with_fewer_than_two_comparable_pairs():
    assert compute_agreement({"r1": 5}, {"r1": 9.0}) is None
    assert compute_agreement({"r1": 5, "r2": 3}, {"r1": 9.0, "r2": None}) is None


def test_compute_agreement_perfect_alignment_gives_kappa_and_correlation_of_one():
    labels = {"r1": 1, "r2": 2, "r3": 3, "r4": 4, "r5": 5}
    judge_totals = {"r1": 0.0, "r2": 2.0, "r3": 4.0, "r4": 6.0, "r5": 8.0}

    result = compute_agreement(labels, judge_totals)

    assert result["weighted_kappa"] == 1.0
    assert result["spearman_correlation"] == 1.0


def test_labels_round_trip_through_disk(tmp_path):
    path = tmp_path / "labels.json"
    assert load_labels(path) == {}

    save_labels(path, {"report-a": 4, "report-b": 2})
    assert load_labels(path) == {"report-a": 4, "report-b": 2}
