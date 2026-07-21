#!/usr/bin/env python3
"""Compare calibrated Gaussian and Student-t residual distributions."""

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import gammaln
from scipy.stats import norm, t


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from car_foundation.probabilistic_residual import (
    horizon_channel_temperature,
    student_t_nll,
)
from validate_nuplan_probabilistic_query import (
    CHANNELS,
    COVERAGE_LEVELS,
    safe_correlation,
)


SOURCE_SUMMARY_SHA256 = (
    "365b6fa69acb199335b08e028aaa06a762b6363a7a6807f2c8d67761504df2d1"
)
SOURCE_DATA_SHA256 = (
    "c0b04e7e50d3d4536fbdae87e6569d2a077cd415a0b0d3bff905252fe1cacacd"
)
INITIAL_DEGREES_OF_FREEDOM = (3.0, 10.0, 30.0)
DF_MINIMUM = 2.05
DF_MAXIMUM = 200.0
TEMPERATURE_GATE_MINIMUM = 0.25
TEMPERATURE_GATE_MAXIMUM = 4.0
BOOTSTRAP_SAMPLES = 5000
CONFIRM_FILES = 744
WINDOWS_PER_FILE = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Gaussian and Student-t on frozen nuPlan errors."
    )
    parser.add_argument(
        "--source-dir",
        default=os.path.join(
            REPO_ROOT,
            "outputs/nuplan_probabilistic_query_smoke/20260717T183423/"
            "large_holdout_confirmation/20260717T185759",
        ),
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--output-name", default="gaussian_vs_student_t"
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source_dir):
    summary_path = source_dir / "summary.json"
    data_path = source_dir / "confirmation_data.npz"
    hashes = {
        "summary": sha256(summary_path),
        "data": sha256(data_path),
    }
    expected = {
        "summary": SOURCE_SUMMARY_SHA256,
        "data": SOURCE_DATA_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(
            f"Step 14 source hash mismatch: actual={hashes}, expected={expected}"
        )
    summary = json.loads(summary_path.read_text())
    arrays = np.load(data_path)
    expected_shapes = {
        "expanded_calibration_error": (11856, 50, 4),
        "expanded_calibration_sigma": (11856, 50, 4),
        "confirm_test_error": (4464, 50, 4),
        "confirm_test_sigma": (4464, 50, 4),
    }
    result = {}
    for name, shape in expected_shapes.items():
        value = arrays[name]
        if value.shape != shape:
            raise RuntimeError(
                f"{name} shape changed: {value.shape}, expected {shape}"
            )
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"{name} contains non-finite values")
        result[name] = value
    if np.any(result["expanded_calibration_sigma"] <= 0) or np.any(
        result["confirm_test_sigma"] <= 0
    ):
        raise RuntimeError("Source sigma must be strictly positive")
    return summary, result, hashes


def bounded_degrees_of_freedom(raw):
    return DF_MINIMUM + (
        DF_MAXIMUM - DF_MINIMUM
    ) * torch.sigmoid(raw)


def inverse_bounded_degrees_of_freedom(value, dtype):
    fraction = (value - DF_MINIMUM) / (
        DF_MAXIMUM - DF_MINIMUM
    )
    return torch.full(
        (len(CHANNELS),),
        math.log(fraction / (1.0 - fraction)),
        dtype=dtype,
    )


def fit_student_t(
    calibration_error,
    calibration_sigma,
    gaussian_temperature,
    initial_df,
    max_iter,
):
    dtype = calibration_error.dtype
    initial_scale_factor = math.sqrt((initial_df - 2.0) / initial_df)
    log_temperature = torch.nn.Parameter(
        torch.log(gaussian_temperature * initial_scale_factor)
    )
    raw_df = torch.nn.Parameter(
        inverse_bounded_degrees_of_freedom(initial_df, dtype)
    )
    optimizer = torch.optim.LBFGS(
        (log_temperature, raw_df),
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_iter * 5 // 4,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        degrees_of_freedom = bounded_degrees_of_freedom(raw_df)
        loss = student_t_nll(
            calibration_error,
            calibration_sigma * temperature.unsqueeze(0),
            degrees_of_freedom.view(1, 1, -1),
        )
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        temperature = torch.exp(log_temperature)
        degrees_of_freedom = bounded_degrees_of_freedom(raw_df)
        loss = student_t_nll(
            calibration_error,
            calibration_sigma * temperature.unsqueeze(0),
            degrees_of_freedom.view(1, 1, -1),
        )
    state_iterations = [
        state.get("n_iter", 0)
        for state in optimizer.state.values()
        if isinstance(state, dict)
    ]
    return {
        "initial_degrees_of_freedom": initial_df,
        "calibration_nll": float(loss),
        "degrees_of_freedom": degrees_of_freedom.detach().cpu().numpy(),
        "temperature": temperature.detach().cpu().numpy(),
        "optimizer_iterations": max(state_iterations, default=0),
        "closure_evaluations": evaluations,
    }


def gaussian_nll_values(error, scale):
    return 0.5 * (
        np.square(error / scale)
        + 2.0 * np.log(scale)
        + math.log(2.0 * math.pi)
    )


def student_t_nll_values(error, scale, degrees_of_freedom):
    degrees_of_freedom = np.asarray(degrees_of_freedom).reshape(1, 1, -1)
    return (
        np.log(scale)
        + 0.5
        * (
            np.log(degrees_of_freedom)
            + math.log(math.pi)
        )
        + gammaln(0.5 * degrees_of_freedom)
        - gammaln(0.5 * (degrees_of_freedom + 1.0))
        + 0.5
        * (degrees_of_freedom + 1.0)
        * np.log1p(
            np.square(error / scale) / degrees_of_freedom
        )
    )


def scope_metrics(
    error,
    scale,
    nll,
    residual_std,
    family,
    degrees_of_freedom=None,
):
    result = {"channels": {}, "coverage": {}, "mean": {}}
    if family == "gaussian":
        predictive_std = scale
    else:
        df = np.asarray(degrees_of_freedom).reshape(1, 1, -1)
        predictive_std = scale * np.sqrt(df / (df - 2.0))
    for channel_index, channel in enumerate(CHANNELS):
        channel_error = error[..., channel_index].reshape(-1)
        channel_scale = scale[..., channel_index].reshape(-1)
        channel_std = predictive_std[..., channel_index].reshape(-1)
        result["channels"][channel] = {
            "nll": float(np.mean(nll[..., channel_index])),
            "scale_physical_mean": float(
                np.mean(channel_scale) * residual_std[channel_index]
            ),
            "predictive_standard_deviation_physical_mean": float(
                np.mean(channel_std) * residual_std[channel_index]
            ),
            "pearson_scale_abs_error": safe_correlation(
                channel_scale, np.abs(channel_error), "pearson"
            ),
            "spearman_scale_abs_error": safe_correlation(
                channel_scale, np.abs(channel_error), "spearman"
            ),
        }
    result["mean"]["nll"] = float(np.mean(nll))
    correlations = [
        result["channels"][channel]["spearman_scale_abs_error"]
        for channel in CHANNELS
        if result["channels"][channel]["spearman_scale_abs_error"] is not None
    ]
    result["mean"]["spearman_scale_abs_error"] = float(
        np.mean(correlations)
    )

    for label, (gaussian_quantile, nominal) in COVERAGE_LEVELS.items():
        probability = 0.5 * (1.0 + nominal)
        if family == "gaussian":
            quantile = np.full(len(CHANNELS), gaussian_quantile)
        else:
            quantile = t.ppf(
                probability, np.asarray(degrees_of_freedom)
            )
        threshold = scale * quantile.reshape(1, 1, -1)
        covered = np.mean(np.abs(error) <= threshold, axis=(0, 1))
        widths = np.mean(
            2.0
            * threshold
            * residual_std.reshape(1, 1, -1),
            axis=(0, 1),
        )
        absolute_errors = np.abs(covered - nominal)
        result["coverage"][label] = {
            "nominal": nominal,
            "empirical_by_channel": {
                channel: float(covered[index])
                for index, channel in enumerate(CHANNELS)
            },
            "absolute_error_by_channel": {
                channel: float(absolute_errors[index])
                for index, channel in enumerate(CHANNELS)
            },
            "absolute_error_mean": float(np.mean(absolute_errors)),
            "absolute_error_maximum": float(np.max(absolute_errors)),
            "physical_width_by_channel": {
                channel: float(widths[index])
                for index, channel in enumerate(CHANNELS)
            },
        }
    return result


def family_metrics(
    error,
    base_sigma,
    temperature,
    residual_std,
    family,
    degrees_of_freedom=None,
):
    scale = base_sigma * temperature.reshape(1, 50, 4)
    if family == "gaussian":
        nll = gaussian_nll_values(error, scale)
    else:
        nll = student_t_nll_values(
            error, scale, degrees_of_freedom
        )
    result = {
        "all_steps": scope_metrics(
            error,
            scale,
            nll,
            residual_std,
            family,
            degrees_of_freedom,
        ),
        "horizons": {},
    }
    for horizon in range(50):
        result["horizons"][str(horizon + 1)] = scope_metrics(
            error[:, horizon : horizon + 1],
            scale[:, horizon : horizon + 1],
            nll[:, horizon : horizon + 1],
            residual_std,
            family,
            degrees_of_freedom,
        )
    return result, nll, scale


def bootstrap_mean(values, rng):
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    indices = rng.integers(
        0, count, size=(BOOTSTRAP_SAMPLES, count)
    )
    samples = values[indices].mean(axis=1)
    return {
        "files": count,
        "mean_delta": float(np.mean(values)),
        "student_t_improvement": float(-np.mean(values)),
        "cluster_bootstrap_95ci": np.quantile(
            samples, (0.025, 0.975)
        ).tolist(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def nll_cluster_inference(student_nll, gaussian_nll, seed):
    delta = student_nll - gaussian_nll
    expected = CONFIRM_FILES * WINDOWS_PER_FILE
    if delta.shape[0] != expected:
        raise RuntimeError(
            f"Expected {expected} confirm episodes, got {delta.shape[0]}"
        )
    by_file = delta.reshape(
        CONFIRM_FILES, WINDOWS_PER_FILE, 50, len(CHANNELS)
    )
    result = {
        "overall": bootstrap_mean(
            by_file.mean(axis=(1, 2, 3)),
            np.random.default_rng(seed),
        ),
        "channels": {},
    }
    for channel_index, channel in enumerate(CHANNELS):
        result["channels"][channel] = bootstrap_mean(
            by_file[..., channel_index].mean(axis=(1, 2)),
            np.random.default_rng(seed + channel_index + 1),
        )
    return result


def plot_coverage(metrics, output_path):
    figure, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
    horizons = np.arange(1, 51)
    for row, channel in enumerate(CHANNELS):
        for column, label in enumerate(COVERAGE_LEVELS):
            axis = axes[row, column]
            for family, family_metrics_result in metrics.items():
                values = [
                    family_metrics_result["horizons"][str(horizon)][
                        "coverage"
                    ][label]["absolute_error_by_channel"][channel]
                    for horizon in horizons
                ]
                axis.plot(horizons, values, label=family)
            axis.axhline(0.05, color="black", linestyle="--")
            axis.axhline(0.10, color="gray", linestyle=":")
            axis.set_title(f"{channel}: {label}% coverage error")
            axis.grid(alpha=0.25)
            if row == len(CHANNELS) - 1:
                axis.set_xlabel("horizon step")
            if column == 0:
                axis.set_ylabel("absolute error")
            axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_nll(metrics, output_path):
    labels = list(CHANNELS) + ["mean"]
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5))
    for offset, family in ((-width / 2, "gaussian"), (width / 2, "student_t")):
        result = metrics[family]["all_steps"]
        values = [
            result["channels"][channel]["nll"]
            for channel in CHANNELS
        ] + [result["mean"]["nll"]]
        axis.bar(x + offset, values, width, label=family)
    axis.set_xticks(x, labels)
    axis.set_ylabel("normalized transition NLL")
    axis.set_title("Confirm-test NLL by distribution family")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_temperature(
    gaussian_temperature,
    student_temperature,
    degrees_of_freedom,
    output_path,
):
    horizons = np.arange(1, 51)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for channel_index, (channel, axis) in enumerate(
        zip(CHANNELS, axes.reshape(-1))
    ):
        axis.plot(
            horizons,
            gaussian_temperature[:, channel_index],
            label="Gaussian std multiplier",
        )
        axis.plot(
            horizons,
            student_temperature[:, channel_index],
            label="Student-t scale multiplier",
        )
        axis.set_title(
            f"{channel}: Student-t df={degrees_of_freedom[channel_index]:.2f}"
        )
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes[-1]:
        axis.set_xlabel("horizon step")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_file_nll_delta(student_nll, gaussian_nll, output_path):
    delta = (student_nll - gaussian_nll).reshape(
        CONFIRM_FILES, WINDOWS_PER_FILE, 50, len(CHANNELS)
    ).mean(axis=(1, 2, 3))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(delta, bins=60, alpha=0.8)
    axis.axvline(0.0, color="black", linestyle="--")
    axis.axvline(np.mean(delta), color="tab:red", label="file mean")
    axis.set_xlabel("Student-t minus Gaussian file-mean NLL")
    axis.set_ylabel("files")
    axis.set_title("File-cluster NLL difference on confirm-test")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if args.max_iter != 200:
        raise ValueError("The frozen formal protocol requires max_iter=200")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    source_dir = Path(args.source_dir)
    source_summary, arrays, source_hashes = verify_source(source_dir)
    output_dir = (
        source_dir
        / args.output_name
        / datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    calibration_error_float32 = torch.from_numpy(
        arrays["expanded_calibration_error"]
    )
    calibration_sigma_float32 = torch.from_numpy(
        arrays["expanded_calibration_sigma"]
    )
    gaussian_temperature_float32 = horizon_channel_temperature(
        calibration_error_float32,
        calibration_sigma_float32,
    )
    stored_gaussian_temperature = np.asarray(
        source_summary["horizon_temperature"]["values"],
        dtype=np.float32,
    )
    temperature_reproduction_error = float(
        np.max(
            np.abs(
                gaussian_temperature_float32.numpy()
                - stored_gaussian_temperature
            )
        )
    )

    calibration_error = calibration_error_float32.to(torch.float64)
    calibration_sigma = calibration_sigma_float32.to(torch.float64)
    gaussian_temperature = gaussian_temperature_float32.to(torch.float64)
    fit_runs = []
    for initial_df in INITIAL_DEGREES_OF_FREEDOM:
        print(f"Fitting Student-t from df={initial_df:g}", flush=True)
        fit_run = fit_student_t(
            calibration_error,
            calibration_sigma,
            gaussian_temperature,
            initial_df,
            args.max_iter,
        )
        fit_runs.append(fit_run)
        print(
            json.dumps(
                {
                    "initial_df": initial_df,
                    "calibration_nll": fit_run["calibration_nll"],
                    "degrees_of_freedom": fit_run[
                        "degrees_of_freedom"
                    ].tolist(),
                    "temperature_minimum": float(
                        np.min(fit_run["temperature"])
                    ),
                    "temperature_maximum": float(
                        np.max(fit_run["temperature"])
                    ),
                    "iterations": fit_run["optimizer_iterations"],
                    "evaluations": fit_run["closure_evaluations"],
                },
                indent=2,
            ),
            flush=True,
        )
    selected_index = int(
        np.argmin([run["calibration_nll"] for run in fit_runs])
    )
    selected = fit_runs[selected_index]
    student_temperature = selected["temperature"]
    degrees_of_freedom = selected["degrees_of_freedom"]
    mean_checkpoint_path = Path(source_summary["mean_checkpoint"])
    if sha256(mean_checkpoint_path) != source_summary["mean_checkpoint_sha256"]:
        raise RuntimeError("Mean checkpoint hash changed after Step 14")
    mean_checkpoint = torch.load(
        mean_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    residual_std = (
        mean_checkpoint["stats"]["residual"][1].cpu().numpy()
    )
    if not np.all(np.isfinite(residual_std)):
        raise RuntimeError("Could not reconstruct residual physical scales")

    metrics = {"calibration": {}, "confirm_test": {}}
    nll_arrays = {}
    scale_arrays = {}
    for split, error_key, sigma_key in (
        (
            "calibration",
            "expanded_calibration_error",
            "expanded_calibration_sigma",
        ),
        ("confirm_test", "confirm_test_error", "confirm_test_sigma"),
    ):
        error = arrays[error_key].astype(np.float64)
        sigma = arrays[sigma_key].astype(np.float64)
        (
            metrics[split]["gaussian"],
            nll_arrays[(split, "gaussian")],
            scale_arrays[(split, "gaussian")],
        ) = family_metrics(
            error,
            sigma,
            gaussian_temperature.numpy(),
            residual_std,
            "gaussian",
        )
        (
            metrics[split]["student_t"],
            nll_arrays[(split, "student_t")],
            scale_arrays[(split, "student_t")],
        ) = family_metrics(
            error,
            sigma,
            student_temperature,
            residual_std,
            "student_t",
            degrees_of_freedom,
        )

    stored_gaussian_nll = source_summary["metrics"][
        "horizon_temperature"
    ]["all_steps"]["mean"]["nll"]
    reproduced_gaussian_nll = metrics["confirm_test"]["gaussian"][
        "all_steps"
    ]["mean"]["nll"]
    nll_reproduction_error = abs(
        reproduced_gaussian_nll - stored_gaussian_nll
    )
    cluster_inference = nll_cluster_inference(
        nll_arrays[("confirm_test", "student_t")],
        nll_arrays[("confirm_test", "gaussian")],
        args.seed,
    )

    confirm_gaussian = metrics["confirm_test"]["gaussian"]["all_steps"]
    confirm_student = metrics["confirm_test"]["student_t"]["all_steps"]
    mean_nll_improvement = (
        confirm_gaussian["mean"]["nll"]
        - confirm_student["mean"]["nll"]
    )
    channel_nll_deltas = {
        channel: (
            confirm_student["channels"][channel]["nll"]
            - confirm_gaussian["channels"][channel]["nll"]
        )
        for channel in CHANNELS
    }
    coverage_errors = {
        label: confirm_student["coverage"][label][
            "absolute_error_by_channel"
        ]
        for label in COVERAGE_LEVELS
    }
    width_ratios_95 = {
        channel: (
            confirm_student["coverage"]["95"][
                "physical_width_by_channel"
            ][channel]
            / confirm_gaussian["coverage"]["95"][
                "physical_width_by_channel"
            ][channel]
        )
        for channel in CHANNELS
    }
    all_finite = bool(
        np.all(np.isfinite(degrees_of_freedom))
        and np.all(np.isfinite(student_temperature))
        and all(
            np.isfinite(value)
            for value in channel_nll_deltas.values()
        )
        and all(
            np.isfinite(value)
            for label_result in coverage_errors.values()
            for value in label_result.values()
        )
    )
    temperature_in_range = bool(
        np.all(student_temperature >= TEMPERATURE_GATE_MINIMUM)
        and np.all(student_temperature <= TEMPERATURE_GATE_MAXIMUM)
    )
    gates = {
        "gaussian_temperature_reproduced": (
            temperature_reproduction_error <= 1e-6
        ),
        "gaussian_confirm_nll_reproduced": (
            nll_reproduction_error <= 1e-6
        ),
        "confirm_mean_nll_improves_at_least_0.005": (
            mean_nll_improvement >= 0.005
        ),
        "file_cluster_nll_delta_ci_below_zero": (
            cluster_inference["overall"]["cluster_bootstrap_95ci"][1]
            < 0.0
        ),
        "no_channel_nll_regresses_over_0.01": all(
            delta <= 0.01 for delta in channel_nll_deltas.values()
        ),
        "all_pooled_channel_coverage_within_5pp": all(
            value < 0.05
            for label_result in coverage_errors.values()
            for value in label_result.values()
        ),
        "all_95_width_ratios_at_most_1.25": all(
            ratio <= 1.25 for ratio in width_ratios_95.values()
        ),
        "parameters_and_metrics_finite": all_finite,
        "temperature_in_frozen_range": temperature_in_range,
    }
    gates["student_t_preferred"] = all(gates.values())

    plot_coverage(
        metrics["confirm_test"], output_dir / "coverage_by_channel_horizon.png"
    )
    plot_nll(metrics["confirm_test"], output_dir / "nll_by_channel.png")
    plot_temperature(
        gaussian_temperature.numpy(),
        student_temperature,
        degrees_of_freedom,
        output_dir / "temperature_and_df.png",
    )
    plot_file_nll_delta(
        nll_arrays[("confirm_test", "student_t")],
        nll_arrays[("confirm_test", "gaussian")],
        output_dir / "file_nll_delta.png",
    )

    fit_run_summaries = []
    for run in fit_runs:
        fit_run_summaries.append(
            {
                "initial_degrees_of_freedom": run[
                    "initial_degrees_of_freedom"
                ],
                "calibration_nll": run["calibration_nll"],
                "degrees_of_freedom": run[
                    "degrees_of_freedom"
                ].tolist(),
                "temperature_minimum": float(
                    np.min(run["temperature"])
                ),
                "temperature_maximum": float(
                    np.max(run["temperature"])
                ),
                "optimizer_iterations": run["optimizer_iterations"],
                "closure_evaluations": run["closure_evaluations"],
            }
        )
    summary = {
        "protocol": "nuplan_gaussian_vs_student_t_v1",
        "source_dir": str(source_dir.resolve()),
        "source_hashes": source_hashes,
        "args": vars(args),
        "frozen_distribution_specification": {
            "gaussian_parameters": 200,
            "student_t_parameters": 204,
            "student_t_location": 0.0,
            "student_t_degrees_of_freedom_bounds": [
                DF_MINIMUM,
                DF_MAXIMUM,
            ],
            "initial_degrees_of_freedom": list(
                INITIAL_DEGREES_OF_FREEDOM
            ),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "gaussian_reproduction": {
            "temperature_max_abs_error": temperature_reproduction_error,
            "stored_confirm_nll": stored_gaussian_nll,
            "reproduced_confirm_nll": reproduced_gaussian_nll,
            "confirm_nll_abs_error": nll_reproduction_error,
        },
        "student_t_fit_runs": fit_run_summaries,
        "selected_fit_index": selected_index,
        "student_t_degrees_of_freedom": {
            channel: float(degrees_of_freedom[index])
            for index, channel in enumerate(CHANNELS)
        },
        "gaussian_temperature": gaussian_temperature.tolist(),
        "student_t_temperature": student_temperature.tolist(),
        "metrics": metrics,
        "file_cluster_nll_inference": cluster_inference,
        "gate_inputs": {
            "confirm_mean_nll_improvement": mean_nll_improvement,
            "confirm_channel_nll_student_minus_gaussian": (
                channel_nll_deltas
            ),
            "student_t_pooled_coverage_absolute_errors": coverage_errors,
            "student_t_over_gaussian_95_width_ratios": width_ratios_95,
            "student_t_temperature_minimum": float(
                np.min(student_temperature)
            ),
            "student_t_temperature_maximum": float(
                np.max(student_temperature)
            ),
        },
        "gates": gates,
        "limitations": [
            "The confirm-test files were already inspected in Step 14, so this "
            "is a frozen controlled comparison rather than a fresh blind test.",
            "The deterministic mean and heteroscedastic sigma head are frozen; "
            "only distribution calibration parameters are fitted.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "protocol": summary["protocol"],
            "source_hashes": source_hashes,
            "degrees_of_freedom": torch.from_numpy(
                degrees_of_freedom
            ),
            "temperature": torch.from_numpy(student_temperature),
            "gates": gates,
        },
        output_dir / "student_t_calibration.pt",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "gaussian_reproduction": summary[
                    "gaussian_reproduction"
                ],
                "selected_fit_index": selected_index,
                "student_t_degrees_of_freedom": summary[
                    "student_t_degrees_of_freedom"
                ],
                "confirm_nll": {
                    family: metrics["confirm_test"][family][
                        "all_steps"
                    ]["mean"]["nll"]
                    for family in ("gaussian", "student_t")
                },
                "file_cluster_nll_inference": cluster_inference,
                "gate_inputs": summary["gate_inputs"],
                "gates": gates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
