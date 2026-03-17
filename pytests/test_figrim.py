import numpy as np

from membench.figrim import (
    evaluate_mean_ensemble_on_splits,
    evaluate_rank_mean_ensemble_on_splits,
    evaluate_rank_weighted_ensemble_on_splits,
    evaluate_stacked_ridge_ensemble_on_splits,
    evaluate_weighted_ensemble_on_splits,
    evaluate_pls_on_splits,
    evaluate_ridge_on_splits,
    generate_repeated_splits,
    parse_pca_dims,
    parse_positive_dims,
)


def test_generate_repeated_splits_is_deterministic() -> None:
    categories = ["a"] * 12 + ["b"] * 12 + ["c"] * 12
    first = generate_repeated_splits(categories, num_splits=3, random_state=7)
    second = generate_repeated_splits(categories, num_splits=3, random_state=7)

    assert len(first) == len(second) == 3
    for left, right in zip(first, second):
        assert left.seed == right.seed
        assert np.array_equal(left.train, right.train)
        assert np.array_equal(left.val, right.val)
        assert np.array_equal(left.test, right.test)
        assert left.train.size == 24
        assert left.val.size == 4
        assert left.test.size == 8


def test_evaluate_ridge_on_splits_with_pca_returns_finite_metrics() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(45, 8)).astype(np.float32)
    weights = np.asarray([0.9, -0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scores = (features @ weights + rng.normal(scale=0.05, size=45)).astype(np.float32)
    categories = ["a"] * 15 + ["b"] * 15 + ["c"] * 15
    splits = generate_repeated_splits(categories, num_splits=2, random_state=3)

    results = evaluate_ridge_on_splits(features=features, scores=scores, splits=splits, pca_dim=4)

    assert len(results) == 2
    for item in results:
        assert np.isfinite(item["val_metrics"]["spearman"])
        assert np.isfinite(item["val_metrics"]["mse"])
        assert np.isfinite(item["test_metrics"]["spearman"])
        assert np.isfinite(item["test_metrics"]["mse"])
        assert item["ridge_alpha"] > 0.0


def test_parse_pca_dims_handles_none_and_ints() -> None:
    assert parse_pca_dims(["none", "128", "32"]) == [None, 128, 32]


def test_evaluate_pls_on_splits_returns_finite_metrics() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(45, 10)).astype(np.float32)
    weights = np.asarray([0.8, -0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scores = (features @ weights + rng.normal(scale=0.05, size=45)).astype(np.float32)
    categories = ["a"] * 15 + ["b"] * 15 + ["c"] * 15
    splits = generate_repeated_splits(categories, num_splits=2, random_state=11)

    results = evaluate_pls_on_splits(features=features, scores=scores, splits=splits, n_components=4)

    assert len(results) == 2
    for item in results:
        assert np.isfinite(item["val_metrics"]["spearman"])
        assert np.isfinite(item["test_metrics"]["spearman"])
        assert item["pls_components"] == 4


def test_evaluate_mean_ensemble_on_splits_returns_finite_metrics() -> None:
    categories = ["a"] * 15 + ["b"] * 15 + ["c"] * 15
    splits = generate_repeated_splits(categories, num_splits=2, random_state=5)
    scores = np.linspace(0.1, 0.9, num=45, dtype=np.float32)

    member_a = []
    member_b = []
    for split in splits:
        val_truth = scores[split.val]
        test_truth = scores[split.test]
        member_a.append(
            {
                "_val_predictions": val_truth + 0.01,
                "_test_predictions": test_truth + 0.01,
            }
        )
        member_b.append(
            {
                "_val_predictions": val_truth - 0.01,
                "_test_predictions": test_truth - 0.01,
            }
        )

    results = evaluate_mean_ensemble_on_splits([member_a, member_b], scores=scores, splits=splits)

    assert len(results) == 2
    for item in results:
        assert np.isfinite(item["val_metrics"]["spearman"])
        assert np.isfinite(item["test_metrics"]["spearman"])


def test_parse_positive_dims_handles_ints() -> None:
    assert parse_positive_dims(["64", "16"]) == [64, 16]


def test_weighted_ensemble_prefers_better_member() -> None:
    categories = ["a"] * 15 + ["b"] * 15 + ["c"] * 15
    splits = generate_repeated_splits(categories, num_splits=1, random_state=9)
    split = splits[0]
    scores = np.linspace(0.1, 0.9, num=45, dtype=np.float32)

    good_member = [
        {
            "_val_predictions": scores[split.val].copy(),
            "_test_predictions": scores[split.test].copy(),
        }
    ]
    bad_member = [
        {
            "_val_predictions": scores[split.val][::-1].copy(),
            "_test_predictions": scores[split.test][::-1].copy(),
        }
    ]

    results = evaluate_weighted_ensemble_on_splits([good_member, bad_member], scores=scores, splits=splits, step=0.5)

    assert len(results) == 1
    assert results[0]["ensemble_weights"] == [1.0, 0.0]
    assert results[0]["test_metrics"]["spearman"] == 1.0


def test_evaluate_rank_mean_ensemble_on_splits_returns_finite_metrics() -> None:
    categories = ["a"] * 15 + ["b"] * 15 + ["c"] * 15
    splits = generate_repeated_splits(categories, num_splits=2, random_state=13)
    scores = np.linspace(0.1, 0.9, num=45, dtype=np.float32)

    member_a = []
    member_b = []
    for split in splits:
        val_truth = scores[split.val]
        test_truth = scores[split.test]
        member_a.append(
            {
                "_val_predictions": val_truth + 0.02,
                "_test_predictions": test_truth + 0.02,
            }
        )
        member_b.append(
            {
                "_val_predictions": val_truth * 0.9,
                "_test_predictions": test_truth * 0.9,
            }
        )

    results = evaluate_rank_mean_ensemble_on_splits([member_a, member_b], scores=scores, splits=splits)

    assert len(results) == 2
    for item in results:
        assert np.isfinite(item["val_metrics"]["spearman"])
        assert np.isfinite(item["test_metrics"]["spearman"])


def test_rank_weighted_ensemble_prefers_better_member() -> None:
    categories = ["a"] * 15 + ["b"] * 15 + ["c"] * 15
    splits = generate_repeated_splits(categories, num_splits=1, random_state=15)
    split = splits[0]
    scores = np.linspace(0.1, 0.9, num=45, dtype=np.float32)

    good_member = [
        {
            "_val_predictions": scores[split.val].copy(),
            "_test_predictions": scores[split.test].copy(),
        }
    ]
    bad_member = [
        {
            "_val_predictions": scores[split.val][::-1].copy(),
            "_test_predictions": scores[split.test][::-1].copy(),
        }
    ]

    results = evaluate_rank_weighted_ensemble_on_splits(
        [good_member, bad_member],
        scores=scores,
        splits=splits,
        step=1.0,
    )

    assert len(results) == 1
    assert results[0]["ensemble_weights"] == [1.0, 0.0]
    assert results[0]["test_metrics"]["spearman"] == 1.0


def test_stacked_ridge_ensemble_returns_finite_metrics() -> None:
    categories = ["a"] * 15 + ["b"] * 15 + ["c"] * 15
    splits = generate_repeated_splits(categories, num_splits=2, random_state=17)
    scores = np.linspace(0.1, 0.9, num=45, dtype=np.float32)

    member_a = []
    member_b = []
    for split in splits:
        val_truth = scores[split.val]
        test_truth = scores[split.test]
        member_a.append(
            {
                "_val_predictions": val_truth + 0.01,
                "_test_predictions": test_truth + 0.01,
            }
        )
        member_b.append(
            {
                "_val_predictions": (val_truth * 0.7) + 0.03,
                "_test_predictions": (test_truth * 0.7) + 0.03,
            }
        )

    results = evaluate_stacked_ridge_ensemble_on_splits([member_a, member_b], scores=scores, splits=splits)

    assert len(results) == 2
    for item in results:
        assert np.isfinite(item["val_metrics"]["spearman"])
        assert np.isfinite(item["test_metrics"]["spearman"])
        assert item["ridge_alpha"] > 0.0
