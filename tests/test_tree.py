"""Tests for the depth-limited and incrementally trained decision tree."""

import numpy as np
import pytest

from numcompute.tree import DecisionTreeClassifier


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_tree_learns_simple_numeric_split(criterion):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["low", "low", "high", "high"])
    tree = DecisionTreeClassifier(criterion=criterion).fit(X, y)

    np.testing.assert_array_equal(tree.predict(X), y)
    np.testing.assert_array_equal(tree.classes_, ["high", "low"])


def test_max_depth_zero_predicts_training_majority():
    tree = DecisionTreeClassifier(max_depth=0).fit([[0], [1], [2]], [1, 0, 1])

    np.testing.assert_array_equal(tree.predict([[0], [2]]), [1, 1])


def test_binary_features_choose_the_highest_information_gain_question():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)

    assert tree.tree_.feature == 0
    assert tree.tree_.threshold == 0.5
    np.testing.assert_array_equal(tree.predict(X), y)


def test_no_positive_gain_stops_at_majority_leaf():
    tree = DecisionTreeClassifier().fit([[1], [1], [1], [1]], [0, 1, 1, 0])

    assert tree.tree_.is_leaf
    np.testing.assert_array_equal(tree.predict([[1], [1]]), [0, 0])


def test_partial_fit_keeps_old_tree_until_buffer_is_full():
    tree = DecisionTreeClassifier(buffer_size=4).fit([[0], [1]], [0, 1])

    tree.partial_fit([[0], [0]], [1, 1])
    np.testing.assert_array_equal(tree.predict([[0]]), [0])
    assert tree.n_samples_fit_ == 2
    assert tree._buffered_samples == 2

    tree.partial_fit([[0], [0]], [1, 1])
    np.testing.assert_array_equal(tree.predict([[0]]), [1])
    assert tree.n_samples_fit_ == 4
    assert tree._buffered_samples == 0


def test_partial_fit_accepts_chunk_larger_than_buffer():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0], [1], [2]], [0, 0, 1])

    assert tree.n_samples_fit_ == 3
    assert tree._buffered_samples == 0
    np.testing.assert_array_equal(tree.predict([[0], [2]]), [0, 1])


def test_fit_replaces_tree_and_discards_pending_buffer():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0]], [0])
    tree.fit([[1], [2]], [1, 1])

    assert tree.n_samples_fit_ == 2
    assert tree._buffered_samples == 0
    np.testing.assert_array_equal(tree.predict([[1], [2]]), [1, 1])


def test_predict_requires_fitted_tree():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0], [1]], [0, 1])

    with pytest.raises(ValueError, match="not fitted"):
        tree.predict([[0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": -1},
        {"min_samples_split": 1},
        {"buffer_size": 0},
        {"criterion": "misclassification"},
        {"max_features": 0},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        DecisionTreeClassifier(**kwargs)


def test_invalid_training_data_rejected():
    tree = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="same number of rows"):
        tree.fit([[0], [1]], [0])
    with pytest.raises(ValueError, match="finite"):
        tree.fit([[np.nan]], [0])
    with pytest.raises(ValueError, match="1-D"):
        tree.fit([[0], [1]], [[0], [1]])


def test_feature_count_and_max_features_validation():
    with pytest.raises(ValueError, match="exceeds"):
        DecisionTreeClassifier(max_features=3).fit([[0, 1], [1, 0]], [0, 1])

    tree = DecisionTreeClassifier().fit([[0, 1], [1, 0]], [0, 1])
    with pytest.raises(ValueError, match="expected 2 features"):
        tree.predict([[1]])
