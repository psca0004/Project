
"""Tests for the apple-count evaluation metrics."""

import pytest

from evaluation_metrics import (
    calculate_mae,
    calculate_relative_mae,
)


def test_mae_perfect_predictions():
    """MAE should be zero when all predictions are correct."""
    actual = [3, 5, 2]
    predicted = [3, 5, 2]

    assert calculate_mae(actual, predicted) == 0


def test_mae_calculation():
    """MAE should equal the mean of the absolute errors."""
    actual = [3, 5, 2]
    predicted = [4, 3, 2]

    assert calculate_mae(actual, predicted) == pytest.approx(1)


def test_relative_mae_calculation():
    """Relative MAE should be expressed as a percentage."""
    actual = [4, 6]
    predicted = [5, 5]

    # MAE = 1; mean actual count = 5; relative MAE = 20%
    assert calculate_relative_mae(actual, predicted) == pytest.approx(20)


def test_mae_rejects_different_list_lengths():
    """MAE should reject unmatched actual and predicted counts."""
    with pytest.raises(ValueError):
        calculate_mae([3, 5], [3])


def test_mae_rejects_empty_lists():
    """MAE should reject an empty evaluation dataset."""
    with pytest.raises(ValueError):
        calculate_mae([], [])


def test_relative_mae_rejects_zero_mean_actual_count():
    """Relative MAE cannot be calculated when mean actual count is zero."""
    with pytest.raises(ValueError):
        calculate_relative_mae([0, 0], [0, 1])