import numpy as np
import pytest

from car_foundation.probability_shift import (
    assign_bins,
    gap_decomposition,
    quantile_edges,
    standardized_mean_difference,
)


def test_quantile_edges_and_assignment_are_frozen():
    calibration = np.arange(10, dtype=np.float64)
    edges = quantile_edges(calibration, bins=5)
    assigned = assign_bins(calibration, edges)
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert np.bincount(assigned).tolist() == [2, 2, 2, 2, 2]
    assert assign_bins(np.array([-100.0, 100.0]), edges).tolist() == [0, 4]


def test_standardized_mean_difference_uses_right_minus_left():
    left = np.array([-1.0, 1.0])
    right = left + 1.0
    assert standardized_mean_difference(left, right) == pytest.approx(1.0)


def test_gap_decomposition_recovers_pure_composition_shift():
    calibration_bins = np.array([0, 0, 1, 1])
    calibration_values = np.array([0.0, 0.0, 2.0, 2.0])
    test_bins = np.array([0, 1, 1, 1])
    test_values = np.array([0.0, 2.0, 2.0, 2.0])
    result = gap_decomposition(
        calibration_values,
        test_values,
        calibration_bins,
        test_bins,
    )
    assert result["overall_gap"] == pytest.approx(0.5)
    assert result["composition"] == pytest.approx(0.5)
    assert result["conditional"] == pytest.approx(0.0)
    assert result["reconstruction_error"] == pytest.approx(0.0)
    assert result["composition_reduction_fraction"] == pytest.approx(1.0)


def test_gap_decomposition_recovers_pure_conditional_shift():
    bins = np.array([0, 0, 1, 1])
    calibration_values = np.zeros(4)
    test_values = np.ones(4)
    result = gap_decomposition(
        calibration_values, test_values, bins, bins
    )
    assert result["composition"] == pytest.approx(0.0)
    assert result["conditional"] == pytest.approx(1.0)
    assert result["composition_reduction_fraction"] == pytest.approx(0.0)


def test_gap_decomposition_reports_missing_support():
    result = gap_decomposition(
        np.array([0.0, 1.0]),
        np.array([1.0, 2.0]),
        np.array([0, 1]),
        np.array([1, 2]),
    )
    assert result["calibration_only_bins"] == [0]
    assert result["test_only_bins"] == [2]
    assert result["calibration_common_support_mass"] == pytest.approx(0.5)
    assert result["test_common_support_mass"] == pytest.approx(0.5)
