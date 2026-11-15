"""A depth-limited decision tree classifier with buffered incremental updates."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .utils import check_array


@dataclass
class _Node:
    """One leaf or split in a fitted decision tree."""

    prediction: object
    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeClassifier:
    """A CART-style classifier for numeric features.

    Parameters
    ----------
    max_depth : int or None, default None
        Maximum number of splits from the root. ``None`` grows until leaves
        are pure or cannot be split.
    min_samples_split : int, default 2
        Minimum number of samples required to consider splitting a node.
    max_features : int, float, {"sqrt", "log2"}, or None, default None
        Number of features considered at each split. A float in (0, 1] selects
        that fraction, rounded up. ``None`` considers every feature.
    buffer_size : int, default 500
        Number of samples to accumulate before ``partial_fit`` retrains the
        tree. A call may trigger retraining with more than this many samples.
    criterion : {"gini", "entropy"}, default "gini"
        Split impurity measure.
    random_state : int or None, default 0
        Seed used when ``max_features`` selects a subset of features.

    Notes
    -----
    Every buffered retrain fits a new tree using only samples accumulated
    since the previous retrain. The previous tree remains available for
    prediction until the buffer reaches ``buffer_size``.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | float | str | None = None,
        buffer_size: int = 500,
        criterion: str = "gini",
        random_state: int | None = 0,
    ) -> None:
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, Integral) or max_depth < 0
        ):
            raise ValueError("max_depth must be a nonnegative integer or None")
        if (
            isinstance(min_samples_split, bool)
            or not isinstance(min_samples_split, Integral)
            or min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer of at least 2")
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, Integral) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_features is not None:
            if isinstance(max_features, str):
                if max_features not in ("sqrt", "log2"):
                    raise ValueError("max_features must be an integer, fraction, 'sqrt', 'log2', or None")
            elif isinstance(max_features, bool) or not isinstance(max_features, Real):
                raise TypeError("max_features must be an integer, fraction, 'sqrt', 'log2', or None")
            elif isinstance(max_features, Integral):
                if max_features < 1:
                    raise ValueError("integer max_features must be positive")
            elif not 0 < max_features <= 1:
                raise ValueError("fractional max_features must be in (0, 1]")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral)
        ):
            raise TypeError("random_state must be an integer or None")

        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.buffer_size = int(buffer_size)
        self.criterion = criterion
        self.random_state = None if random_state is None else int(random_state)
        self._buffer_X: list[np.ndarray] = []
        self._buffer_y: list[np.ndarray] = []
        self._buffered_samples = 0

    @staticmethod
    def _validate_xy(X, y) -> tuple[np.ndarray, np.ndarray]:
        X = check_array(X, ndim=2)
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite numeric values")

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if y.size == 0:
            raise ValueError("y is empty")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X and y must have the same number of rows; got {X.shape[0]} and {y.shape[0]}")
        if y.dtype.kind not in "biufUS":
            raise TypeError("y must contain real numeric or string labels")
        if y.dtype.kind == "f" and not np.all(np.isfinite(y)):
            raise ValueError("y must not contain NaN or infinity")
        return X, y

    def _feature_count(self, n_features: int) -> int:
        setting = self.max_features
        if setting is None:
            return n_features
        if setting == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if setting == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(setting, Integral):
            if setting > n_features:
                raise ValueError(f"max_features={setting} exceeds the {n_features} input features")
            return int(setting)
        return max(1, int(np.ceil(float(setting) * n_features)))

    def _impurity(self, counts: np.ndarray) -> float:
        probabilities = counts[counts > 0] / counts.sum()
        if self.criterion == "gini":
            return float(1.0 - np.sum(probabilities * probabilities))
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def _build_tree(
        self,
        X: np.ndarray,
        encoded_y: np.ndarray,
        classes: np.ndarray,
        depth: int,
        rng: np.random.Generator,
    ) -> _Node:
        counts = np.bincount(encoded_y, minlength=classes.size)
        majority = int(np.argmax(counts))
        node = _Node(prediction=classes[majority])

        if (
            np.count_nonzero(counts) == 1
            or encoded_y.size < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        n_features = X.shape[1]
        n_considered = self._feature_count(n_features)
        if n_considered == n_features:
            features = np.arange(n_features)
        else:
            features = rng.choice(n_features, size=n_considered, replace=False)

        parent_impurity = self._impurity(counts)
        best_score = parent_impurity
        best_feature = None
        best_threshold = None

        for feature in features:
            order = np.argsort(X[:, feature], kind="stable")
            sorted_values = X[order, feature]
            sorted_y = encoded_y[order]
            boundaries = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
            if boundaries.size == 0:
                continue

            left_counts = np.zeros(classes.size, dtype=np.int64)
            previous_boundary = -1
            for boundary in boundaries:
                left_counts += np.bincount(
                    sorted_y[previous_boundary + 1 : boundary + 1],
                    minlength=classes.size,
                )
                left_size = int(boundary + 1)
                right_size = encoded_y.size - left_size
                right_counts = counts - left_counts
                score = (
                    left_size * self._impurity(left_counts)
                    + right_size * self._impurity(right_counts)
                ) / encoded_y.size
                if score < best_score - 1e-15:
                    best_score = score
                    best_feature = int(feature)
                    lower = sorted_values[boundary]
                    upper = sorted_values[boundary + 1]
                    threshold = lower / 2.0 + upper / 2.0
                    best_threshold = float(threshold if lower <= threshold < upper else lower)
                previous_boundary = int(boundary)

        if best_feature is None:
            return node

        left_mask = X[:, best_feature] <= best_threshold
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(X[left_mask], encoded_y[left_mask], classes, depth + 1, rng)
        node.right = self._build_tree(X[~left_mask], encoded_y[~left_mask], classes, depth + 1, rng)
        return node

    def fit(self, X, y) -> "DecisionTreeClassifier":
        """Train a new tree from scratch, replacing any previous fitted tree."""
        X, y = self._validate_xy(X, y)
        classes, encoded_y = np.unique(y, return_inverse=True)
        if self.max_features is not None and isinstance(self.max_features, Integral):
            self._feature_count(X.shape[1])

        rng = np.random.default_rng(self.random_state)
        tree = self._build_tree(X, encoded_y, classes, 0, rng)

        self.tree_ = tree
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.n_samples_fit_ = X.shape[0]
        self._buffer_X.clear()
        self._buffer_y.clear()
        self._buffered_samples = 0
        return self

    def partial_fit(self, X_chunk, y_chunk) -> "DecisionTreeClassifier":
        """Append a chunk and retrain when the buffered sample count reaches the threshold."""
        X_chunk, y_chunk = self._validate_xy(X_chunk, y_chunk)
        if hasattr(self, "n_features_in_") and X_chunk.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X_chunk.shape[1]}")
        if self._buffer_X and X_chunk.shape[1] != self._buffer_X[0].shape[1]:
            raise ValueError(f"expected {self._buffer_X[0].shape[1]} features, got {X_chunk.shape[1]}")

        self._buffer_X.append(X_chunk.copy())
        self._buffer_y.append(y_chunk.copy())
        self._buffered_samples += X_chunk.shape[0]
        if self._buffered_samples >= self.buffer_size:
            buffered_X = np.concatenate(self._buffer_X, axis=0)
            buffered_y = np.concatenate(self._buffer_y, axis=0)
            self.fit(buffered_X, buffered_y)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict labels using the most recently trained tree."""
        if not hasattr(self, "tree_"):
            raise ValueError("DecisionTreeClassifier is not fitted; call fit() or buffer enough samples with partial_fit()")
        X = check_array(X, ndim=2)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite numeric values")

        predictions = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for row_index, row in enumerate(X):
            node = self.tree_
            while not node.is_leaf:
                node = node.left if row[node.feature] <= node.threshold else node.right
            predictions[row_index] = node.prediction
        return predictions
