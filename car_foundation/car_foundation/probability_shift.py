"""Utilities for calibration/test covariate and residual-shift audits."""

from __future__ import annotations

import numpy as np


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    """Return the mean difference normalized by pooled population deviation."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    pooled_variance = 0.5 * (np.var(left) + np.var(right))
    if pooled_variance <= 0:
        return 0.0 if np.mean(left) == np.mean(right) else float("inf")
    return float((np.mean(right) - np.mean(left)) / np.sqrt(pooled_variance))


def quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Freeze monotonic quantile-bin edges with infinite outer bounds."""
    if bins < 2:
        raise ValueError("bins must be at least two")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty vector")
    interior = np.quantile(values, np.arange(1, bins) / bins)
    interior = np.unique(interior)
    return np.concatenate(([-np.inf], interior, [np.inf]))


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign integer bins using frozen edges."""
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3:
        raise ValueError("edges must contain at least two bins")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")
    return np.digitize(values, edges[1:-1], right=False)


def gap_decomposition(
    calibration_values: np.ndarray,
    test_values: np.ndarray,
    calibration_bins: np.ndarray,
    test_bins: np.ndarray,
) -> dict:
    """Decompose a mean gap into composition and within-bin components.

    The exact decomposition is evaluated on bins represented in both splits.
    Common-support masses and absent bins are returned explicitly.
    """
    calibration_values = np.asarray(calibration_values, dtype=np.float64)
    test_values = np.asarray(test_values, dtype=np.float64)
    calibration_bins = np.asarray(calibration_bins)
    test_bins = np.asarray(test_bins)
    if calibration_values.shape != calibration_bins.shape:
        raise ValueError("calibration values/bins must have identical shapes")
    if test_values.shape != test_bins.shape:
        raise ValueError("test values/bins must have identical shapes")

    all_bins = np.union1d(calibration_bins, test_bins)
    rows = []
    common_bins = []
    for bin_id in all_bins:
        calibration_mask = calibration_bins == bin_id
        test_mask = test_bins == bin_id
        calibration_count = int(np.sum(calibration_mask))
        test_count = int(np.sum(test_mask))
        if calibration_count and test_count:
            common_bins.append(bin_id)
        rows.append(
            {
                "bin": int(bin_id),
                "calibration_count": calibration_count,
                "test_count": test_count,
                "calibration_mean": (
                    float(np.mean(calibration_values[calibration_mask]))
                    if calibration_count
                    else None
                ),
                "test_mean": (
                    float(np.mean(test_values[test_mask])) if test_count else None
                ),
                "calibration_rms": (
                    float(
                        np.sqrt(
                            np.mean(np.square(calibration_values[calibration_mask]))
                        )
                    )
                    if calibration_count
                    else None
                ),
                "test_rms": (
                    float(np.sqrt(np.mean(np.square(test_values[test_mask]))))
                    if test_count
                    else None
                ),
                "calibration_68_coverage": (
                    float(
                        np.mean(
                            np.abs(calibration_values[calibration_mask]) <= 1.0
                        )
                    )
                    if calibration_count
                    else None
                ),
                "test_68_coverage": (
                    float(np.mean(np.abs(test_values[test_mask]) <= 1.0))
                    if test_count
                    else None
                ),
            }
        )

    calibration_common = np.isin(calibration_bins, common_bins)
    test_common = np.isin(test_bins, common_bins)
    if not np.any(calibration_common) or not np.any(test_common):
        raise ValueError("No common-support bins")
    calibration_common_values = calibration_values[calibration_common]
    test_common_values = test_values[test_common]
    common_gap = float(
        np.mean(test_common_values) - np.mean(calibration_common_values)
    )

    calibration_total = int(np.sum(calibration_common))
    test_total = int(np.sum(test_common))
    composition = 0.0
    conditional = 0.0
    for bin_id in common_bins:
        calibration_mask = calibration_bins == bin_id
        test_mask = test_bins == bin_id
        calibration_probability = float(
            np.sum(calibration_mask) / calibration_total
        )
        test_probability = float(np.sum(test_mask) / test_total)
        calibration_mean = float(np.mean(calibration_values[calibration_mask]))
        test_mean = float(np.mean(test_values[test_mask]))
        composition += (
            test_probability - calibration_probability
        ) * calibration_mean
        conditional += test_probability * (test_mean - calibration_mean)

    denominator = abs(common_gap)
    composition_reduction = (
        max(0.0, 1.0 - abs(conditional) / denominator)
        if denominator > 1e-12
        else None
    )
    return {
        "overall_gap": float(np.mean(test_values) - np.mean(calibration_values)),
        "common_support_gap": common_gap,
        "composition": float(composition),
        "conditional": float(conditional),
        "reconstruction_error": float(
            common_gap - composition - conditional
        ),
        "composition_signed_share": (
            float(composition / common_gap) if denominator > 1e-12 else None
        ),
        "conditional_signed_share": (
            float(conditional / common_gap) if denominator > 1e-12 else None
        ),
        "composition_reduction_fraction": composition_reduction,
        "calibration_common_support_mass": float(
            np.mean(calibration_common)
        ),
        "test_common_support_mass": float(np.mean(test_common)),
        "calibration_only_bins": [
            int(item)
            for item in all_bins
            if np.any(calibration_bins == item)
            and not np.any(test_bins == item)
        ],
        "test_only_bins": [
            int(item)
            for item in all_bins
            if np.any(test_bins == item)
            and not np.any(calibration_bins == item)
        ],
        "bins": rows,
    }
