"""Tests for optional plotting helpers."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from numcompute.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_metric_over_time_plots_chunk_values_and_returns_axes(tmp_path):
    output = tmp_path / "accuracy.png"

    fig, ax = plot_metric_over_time(
        [0.5, 0.75, 1.0], "Accuracy by chunk", "Accuracy", save_path=output
    )

    assert ax.figure is fig
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.5, 0.75, 1.0])
    assert ax.get_title() == "Accuracy by chunk"
    assert ax.get_ylabel() == "Accuracy"
    assert output.is_file()


@pytest.mark.parametrize(
    "values, error, message",
    [
        ([], ValueError, "empty"),
        ([[0.5, 0.75]], ValueError, "one-dimensional"),
        ([0.5, np.nan], ValueError, "finite"),
        (["good", "bad"], TypeError, "numeric"),
    ],
)
def test_plot_metric_over_time_rejects_invalid_values(values, error, message):
    with pytest.raises(error, match=message):
        plot_metric_over_time(values, "Metric", "Value")


def test_compare_models_supports_different_history_lengths_and_show(monkeypatch, tmp_path):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    output = tmp_path / "comparison.png"

    _, ax = compare_models(
        [0.5, 0.7, 0.8],
        [0.6, 0.75],
        ("small buffer", "large buffer"),
        save_path=output,
        show=True,
    )

    assert [line.get_label() for line in ax.lines] == ["small buffer", "large buffer"]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2])
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [0, 1])
    assert shown == [True]
    assert output.is_file()


@pytest.mark.parametrize(
    "labels, error",
    [
        (("one",), ValueError),
        (("one", 2), ValueError),
        ("two labels", TypeError),
    ],
)
def test_compare_models_rejects_invalid_legend_labels(labels, error):
    with pytest.raises(error, match="labels"):
        compare_models([0.5], [0.6], labels)


def test_plot_predictions_vs_ground_truth_maps_string_labels_to_ticks():
    _, ax = plot_predictions_vs_ground_truth(
        ["cat", "dog", "cat"], ["cat", "cat", "dog"]
    )

    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["cat", "dog"]
    assert [line.get_label() for line in ax.lines] == ["Ground truth", "Predicted"]
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0, 1, 0])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [0, 0, 1])


@pytest.mark.parametrize(
    "actual, predicted, message",
    [
        ([], [], "empty"),
        ([0, 1], [0], "same number"),
        ([[0], [1]], [0, 1], "one-dimensional"),
    ],
)
def test_plot_predictions_vs_ground_truth_rejects_invalid_labels(
    actual, predicted, message
):
    with pytest.raises(ValueError, match=message):
        plot_predictions_vs_ground_truth(actual, predicted)
