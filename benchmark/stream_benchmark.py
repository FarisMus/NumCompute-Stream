"""Compare buffered tree training and chunked accuracy implementations."""

from numbers import Integral
from time import perf_counter

import numpy as np

from benchmark import loops
from numcompute.benchmarking import compare
from numcompute.metrics import accuracy
from numcompute_stream.tree import DecisionTreeClassifier


class _TimedTree(DecisionTreeClassifier):
    """Measure time spent in the tree's actual buffered retrains."""

    def __init__(self, *, buffer_size, max_depth):
        super().__init__(buffer_size=buffer_size, max_depth=max_depth, random_state=0)
        self.retrain_seconds = 0.0
        self.retrain_count = 0
        self.trained_samples = 0

    def fit(self, X, y):
        start = perf_counter()
        result = super().fit(X, y)
        self.retrain_seconds += perf_counter() - start
        self.retrain_count += 1
        self.trained_samples += len(y)
        return result


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def benchmark_buffer_sizes(
    X_train, y_train, X_test, y_test, *, buffer_sizes=(300, 600),
    chunk_size=250, max_depth=6, repeat=3,
):
    """Benchmark fresh buffered trees on identical ordered chunks.

    Return one dict per buffer size with the median total ``partial_fit``
    time and the median total time spent in actual tree ``fit`` calls, both
    in seconds. Accuracy is measured on the same untouched holdout set.
    Samples left below the buffer threshold are reported but not fitted.
    """
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    repeat = _positive_integer(repeat, "repeat")
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    if X_train.ndim != 2 or y_train.ndim != 1 or len(X_train) != len(y_train):
        raise ValueError("training features must be 2-D with one label per row")
    if X_test.ndim != 2 or y_test.ndim != 1 or len(X_test) != len(y_test):
        raise ValueError("holdout features must be 2-D with one label per row")
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("training and holdout sets must not be empty")

    sizes = [_positive_integer(size, "buffer_size") for size in buffer_sizes]
    if not sizes:
        raise ValueError("buffer_sizes must not be empty")
    if any(size > len(X_train) for size in sizes):
        raise ValueError("each buffer_size must be no larger than the training set")

    results = []
    for size in sizes:
        update_times, retrain_times, scores = [], [], []
        for _ in range(repeat):
            tree = _TimedTree(buffer_size=size, max_depth=max_depth)
            start = perf_counter()
            for offset in range(0, len(X_train), chunk_size):
                tree.partial_fit(
                    X_train[offset : offset + chunk_size],
                    y_train[offset : offset + chunk_size],
                )
            update_times.append(perf_counter() - start)
            retrain_times.append(tree.retrain_seconds)
            scores.append(accuracy(y_test, tree.predict(X_test)))
        results.append({
            "buffer_size": size,
            "retrain_count": tree.retrain_count,
            "buffered_samples": len(X_train) - tree.trained_samples,
            "update_seconds": float(np.median(update_times)),
            "retrain_seconds": float(np.median(retrain_times)),
            "accuracy": float(np.median(scores)),
        })
    return results


def benchmark_chunk_accuracy(y_true, y_pred, *, chunk_size=250, repeat=3):
    """Compare loop and NumPy cumulative accuracy on identical chunks.

    Chunk scores are weighted by chunk size, including a shorter final chunk.
    Input conversion and chunk construction are excluded from timing.
    """
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    repeat = _positive_integer(repeat, "repeat")
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("labels and predictions must be 1-D with equal lengths")
    if len(y_true) == 0:
        raise ValueError("labels and predictions must not be empty")
    chunks = [
        (y_true[offset : offset + chunk_size], y_pred[offset : offset + chunk_size])
        for offset in range(0, len(y_true), chunk_size)
    ]
    loop_chunks = [(true.tolist(), pred.tolist()) for true, pred in chunks]

    def loop_accuracy():
        return sum(
            loops.accuracy_loop(true, pred) * len(true) for true, pred in loop_chunks
        ) / len(y_true)

    def numpy_accuracy():
        return sum(
            accuracy(true, pred) * len(true) for true, pred in chunks
        ) / len(y_true)

    return compare("chunked cumulative accuracy", loop_accuracy, numpy_accuracy, repeat=repeat)
