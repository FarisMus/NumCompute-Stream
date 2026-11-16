"""Reusable plots for streaming metrics and predictions.

Matplotlib is imported only when a plotting function is called, so importing
NumCompute does not require the optional visualization dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np

__all__ = [
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
]


def _pyplot():
    """Import pyplot or explain how to install the optional dependency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib; install it with "
            "`python -m pip install 'numcompute[visualization]'`."
        ) from exc
    return plt


def _numeric_series(values, name: str) -> np.ndarray:
    """Validate and convert one nonempty finite numeric series."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values")
    array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finish_plot(fig, ax, save_path: str | PathLike | None, show: bool):
    """Apply optional output settings and return the figure and axes."""
    plt = _pyplot()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_metric_over_time(
    metric_values,
    title: str,
    ylabel: str,
    *,
    save_path: str | PathLike | None = None,
    show: bool = False,
    ax=None,
):
    """Plot a metric value for each chunk.

    Parameters
    ----------
    metric_values : array_like, shape (n_chunks,)
        Nonempty finite numeric metric history.
    title : str
        Plot title.
    ylabel : str
        Label for the metric axis.
    save_path : path-like or None, default None
        If supplied, save the figure to this path.
    show : bool, default False
        If true, display the figure with ``matplotlib.pyplot.show``.
    ax : matplotlib.axes.Axes or None, default None
        Existing axes to draw on; otherwise new figure and axes are created.

    Returns
    -------
    tuple
        The ``(figure, axes)`` containing the plot.
    """
    values = _numeric_series(metric_values, "metric_values")
    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(np.arange(values.size), values, marker="o")
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.grid(True, alpha=0.3)
    return _finish_plot(fig, ax, save_path, show)


def compare_models(
    metric1,
    metric2,
    labels: Sequence[str],
    *,
    save_path: str | PathLike | None = None,
    show: bool = False,
    ax=None,
):
    """Compare two models' metric histories over their respective chunks.

    Parameters
    ----------
    metric1, metric2 : array_like, shape (n_chunks,)
        Nonempty finite numeric metric histories. They may have different
        lengths; each series is plotted against its own chunk indices.
    labels : sequence of str, length 2
        Legend labels for the first and second model.
    save_path : path-like or None, default None
        If supplied, save the figure to this path.
    show : bool, default False
        If true, display the figure with ``matplotlib.pyplot.show``.
    ax : matplotlib.axes.Axes or None, default None
        Existing axes to draw on; otherwise new figure and axes are created.

    Returns
    -------
    tuple
        The ``(figure, axes)`` containing the plot.
    """
    first = _numeric_series(metric1, "metric1")
    second = _numeric_series(metric2, "metric2")
    if isinstance(labels, (str, bytes)):
        raise TypeError("labels must be a sequence of two strings")
    try:
        labels = tuple(labels)
    except TypeError as exc:
        raise TypeError("labels must be a sequence of two strings") from exc
    if len(labels) != 2 or not all(isinstance(label, str) for label in labels):
        raise ValueError("labels must contain exactly two strings")

    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(np.arange(first.size), first, marker="o", label=labels[0])
    ax.plot(np.arange(second.size), second, marker="s", label=labels[1])
    ax.set(xlabel="Chunk", ylabel="Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish_plot(fig, ax, save_path, show)


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    *,
    save_path: str | PathLike | None = None,
    show: bool = False,
    ax=None,
):
    """Plot true and predicted labels across the samples in a chunk.

    Labels are mapped to categorical positions so strings and numeric class
    labels can be plotted consistently. Tick labels retain their string form.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        Nonempty one-dimensional arrays of true and predicted labels, with
        matching lengths.
    save_path : path-like or None, default None
        If supplied, save the figure to this path.
    show : bool, default False
        If true, display the figure with ``matplotlib.pyplot.show``.
    ax : matplotlib.axes.Axes or None, default None
        Existing axes to draw on; otherwise new figure and axes are created.

    Returns
    -------
    tuple
        The ``(figure, axes)`` containing the plot.
    """
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim != 1:
        raise ValueError(f"y_true must be one-dimensional, got shape {actual.shape}")
    if predicted.ndim != 1:
        raise ValueError(f"y_pred must be one-dimensional, got shape {predicted.shape}")
    if actual.size == 0 or predicted.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if actual.size != predicted.size:
        raise ValueError(
            f"y_true and y_pred must have the same number of samples; "
            f"got {actual.size} and {predicted.size}"
        )

    actual_labels = [str(value) for value in actual]
    predicted_labels = [str(value) for value in predicted]
    categories = list(dict.fromkeys(actual_labels + predicted_labels))
    category_indices = {label: index for index, label in enumerate(categories)}
    actual_values = np.fromiter(
        (category_indices[label] for label in actual_labels), dtype=int
    )
    predicted_values = np.fromiter(
        (category_indices[label] for label in predicted_labels), dtype=int
    )

    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    samples = np.arange(actual.size)
    ax.plot(samples, actual_values, marker="o", label="Ground truth")
    ax.plot(samples, predicted_values, marker="x", linestyle="--", label="Predicted")
    ax.set(
        title="Predictions vs. ground truth",
        xlabel="Sample",
        ylabel="Class",
        yticks=np.arange(len(categories)),
        yticklabels=categories,
    )
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    return _finish_plot(fig, ax, save_path, show)
