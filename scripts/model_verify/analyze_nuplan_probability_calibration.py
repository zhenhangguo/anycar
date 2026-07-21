#!/usr/bin/env python3
"""Diagnose nuPlan residual uncertainty and test horizon-wise calibration."""

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import (
    kurtosis,
    kstest,
    norm,
    probplot,
    skew,
    wasserstein_distance,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from car_foundation.kinematic_residual import KinematicBicycleParams
from car_foundation.probabilistic_residual import (
    FrozenMeanGaussianResidual,
    channel_temperature,
    horizon_channel_temperature,
)
from train_kinematic_residual_ablation import make_model
from validate_nuplan_probabilistic_query import (
    CHANNELS,
    COVERAGE_LEVELS,
    collect_predictions,
    distribution_metrics,
    make_loader,
)


SELECTED_HORIZONS = (1, 5, 10, 20, 50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose and horizon-calibrate frozen nuPlan uncertainty."
    )
    parser.add_argument(
        "--run-dir",
        default=os.path.join(
            REPO_ROOT,
            "outputs/nuplan_probabilistic_query_smoke/20260717T183423",
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output-name", default="horizon_calibration")
    return parser.parse_args()


def read_paths(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def checkpoint_stats(checkpoint, device):
    return {
        name: tuple(value.to(device) for value in pair)
        for name, pair in checkpoint["stats"].items()
    }


def scalar_distribution_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    ks_result = kstest(values, "norm")
    result = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "std": float(np.std(values)),
        "skewness": float(skew(values, bias=False)),
        "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)),
        "normal_ks_statistic": float(ks_result.statistic),
        "normal_ks_pvalue": float(ks_result.pvalue),
        "quantiles": {
            label: float(np.quantile(values, probability))
            for label, probability in (
                ("0.5%", 0.005),
                ("2.5%", 0.025),
                ("16%", 0.16),
                ("50%", 0.5),
                ("84%", 0.84),
                ("97.5%", 0.975),
                ("99.5%", 0.995),
            )
        },
        "coverage": {},
    }
    for label, (threshold, nominal) in COVERAGE_LEVELS.items():
        empirical = float(np.mean(np.abs(values) <= threshold))
        result["coverage"][label] = {
            "nominal": nominal,
            "empirical": empirical,
            "absolute_error": abs(empirical - nominal),
        }
    return result


def split_diagnostics(z_values):
    result = {"channels": {}, "horizons": {}}
    for channel_index, channel in enumerate(CHANNELS):
        result["channels"][channel] = scalar_distribution_stats(
            z_values[..., channel_index]
        )
    for horizon_index in range(z_values.shape[1]):
        result["horizons"][str(horizon_index + 1)] = {
            channel: scalar_distribution_stats(
                z_values[:, horizon_index, channel_index]
            )
            for channel_index, channel in enumerate(CHANNELS)
        }
    return result


def all_horizon_metrics(errors, sigmas, residual_std):
    return {
        "all_steps": distribution_metrics(errors, sigmas, residual_std),
        "horizons": {
            str(horizon + 1): distribution_metrics(
                errors[:, horizon : horizon + 1],
                sigmas[:, horizon : horizon + 1],
                residual_std,
            )
            for horizon in range(errors.shape[1])
        },
    }


def drift_diagnostics(calibration_z, test_z):
    result = {"channels": {}, "horizons": {}}
    for channel_index, channel in enumerate(CHANNELS):
        left = calibration_z[..., channel_index].reshape(-1)
        right = test_z[..., channel_index].reshape(-1)
        result["channels"][channel] = {
            "wasserstein_distance": float(wasserstein_distance(left, right)),
            "rms_difference": float(
                np.sqrt(np.mean(np.square(right)))
                - np.sqrt(np.mean(np.square(left)))
            ),
            "mean_difference": float(np.mean(right) - np.mean(left)),
        }
    for horizon_index in range(calibration_z.shape[1]):
        result["horizons"][str(horizon_index + 1)] = {}
        for channel_index, channel in enumerate(CHANNELS):
            left = calibration_z[:, horizon_index, channel_index]
            right = test_z[:, horizon_index, channel_index]
            result["horizons"][str(horizon_index + 1)][channel] = {
                "wasserstein_distance": float(
                    wasserstein_distance(left, right)
                ),
                "rms_difference": float(
                    np.sqrt(np.mean(np.square(right)))
                    - np.sqrt(np.mean(np.square(left)))
                ),
                "mean_difference": float(np.mean(right) - np.mean(left)),
            }
    return result


def plot_histograms_and_qq(calibration_z, test_z, output_path):
    figure, axes = plt.subplots(4, 2, figsize=(12, 16))
    x_grid = np.linspace(-4.0, 4.0, 400)
    for channel_index, channel in enumerate(CHANNELS):
        calibration_values = calibration_z[..., channel_index].reshape(-1)
        test_values = test_z[..., channel_index].reshape(-1)
        histogram_axis = axes[channel_index, 0]
        histogram_axis.hist(
            calibration_values,
            bins=120,
            density=True,
            alpha=0.45,
            range=(-4, 4),
            label="calibration",
        )
        histogram_axis.hist(
            test_values,
            bins=120,
            density=True,
            alpha=0.45,
            range=(-4, 4),
            label="test",
        )
        histogram_axis.plot(x_grid, norm.pdf(x_grid), "k--", label="N(0,1)")
        histogram_axis.set_title(f"{channel}: standardized error")
        histogram_axis.set_xlim(-4, 4)
        histogram_axis.grid(alpha=0.2)
        histogram_axis.legend()

        qq_axis = axes[channel_index, 1]
        calibration_qq = probplot(calibration_values, dist="norm", fit=False)
        test_qq = probplot(test_values, dist="norm", fit=False)
        stride_calibration = max(1, len(calibration_qq[0]) // 3000)
        stride_test = max(1, len(test_qq[0]) // 3000)
        qq_axis.scatter(
            calibration_qq[0][::stride_calibration],
            calibration_qq[1][::stride_calibration],
            s=3,
            alpha=0.45,
            label="calibration",
        )
        qq_axis.scatter(
            test_qq[0][::stride_test],
            test_qq[1][::stride_test],
            s=3,
            alpha=0.45,
            label="test",
        )
        limits = (-4, 4)
        qq_axis.plot(limits, limits, "k--", label="Gaussian")
        qq_axis.set_xlim(limits)
        qq_axis.set_ylim(limits)
        qq_axis.set_title(f"{channel}: Q-Q")
        qq_axis.grid(alpha=0.2)
        qq_axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_horizon_diagnostics(
    calibration_diagnostics,
    test_diagnostics,
    drift,
    output_path,
):
    figure, axes = plt.subplots(4, 3, figsize=(16, 15), sharex=True)
    horizons = np.arange(1, 51)
    for channel_index, channel in enumerate(CHANNELS):
        calibration_stats = [
            calibration_diagnostics["horizons"][str(horizon)][channel]
            for horizon in horizons
        ]
        test_stats = [
            test_diagnostics["horizons"][str(horizon)][channel]
            for horizon in horizons
        ]
        drift_stats = [
            drift["horizons"][str(horizon)][channel]
            for horizon in horizons
        ]
        axes[channel_index, 0].plot(
            horizons,
            [entry["rms"] for entry in calibration_stats],
            label="calibration",
        )
        axes[channel_index, 0].plot(
            horizons,
            [entry["rms"] for entry in test_stats],
            label="test",
        )
        axes[channel_index, 0].axhline(1.0, color="black", linestyle="--")
        axes[channel_index, 0].set_ylabel(channel)
        axes[channel_index, 0].set_title("raw z RMS")
        axes[channel_index, 0].legend()

        axes[channel_index, 1].plot(
            horizons,
            [entry["excess_kurtosis"] for entry in calibration_stats],
            label="calibration",
        )
        axes[channel_index, 1].plot(
            horizons,
            [entry["excess_kurtosis"] for entry in test_stats],
            label="test",
        )
        axes[channel_index, 1].axhline(0.0, color="black", linestyle="--")
        axes[channel_index, 1].set_title("excess kurtosis")
        axes[channel_index, 1].legend()

        axes[channel_index, 2].plot(
            horizons,
            [entry["wasserstein_distance"] for entry in drift_stats],
        )
        axes[channel_index, 2].set_title("calibration-test Wasserstein")
        for axis in axes[channel_index]:
            axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("horizon step")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_temperature(temperature, output_path):
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(
        temperature.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=min(0.5, float(np.min(temperature))),
        vmax=max(1.5, float(np.max(temperature))),
    )
    axis.set_yticks(np.arange(len(CHANNELS)), CHANNELS)
    axis.set_xlabel("horizon index (zero-based in image)")
    axis.set_title("horizon × channel temperature")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_coverage_comparison(metrics, output_path):
    figure, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    horizons = np.arange(1, 51)
    for axis, label in zip(axes, COVERAGE_LEVELS):
        for method, method_metrics in metrics.items():
            values = [
                method_metrics["horizons"][str(horizon)]["coverage"][label][
                    "absolute_error_mean"
                ]
                for horizon in horizons
            ]
            axis.plot(horizons, values, label=method)
        axis.axhline(0.05, color="black", linestyle="--", label="5 pp gate")
        axis.axhline(0.10, color="gray", linestyle=":", label="10 pp gate")
        axis.set_ylabel(f"{label}% error")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("horizon step")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    source_summary = json.loads((run_dir / "summary.json").read_text())
    device = torch.device(args.device)
    output_dir = (
        run_dir / args.output_name / datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    mean_checkpoint_path = Path(source_summary["checkpoint"])
    mean_checkpoint = torch.load(
        mean_checkpoint_path, map_location=device, weights_only=False
    )
    model_args = SimpleNamespace(**mean_checkpoint["args"])
    mean_model = make_model(model_args, device, "query")
    mean_model.load_state_dict(mean_checkpoint["model_state_dict"])
    mean_model.eval()
    wrapper = FrozenMeanGaussianResidual(
        mean_model,
        output_dim=4,
        sigma_floor=source_summary["args"]["sigma_floor"],
    ).to(device)
    probability_checkpoint_path = run_dir / "probabilistic_query_best.pt"
    probability_checkpoint = torch.load(
        probability_checkpoint_path, map_location=device, weights_only=False
    )
    wrapper.sigma_head.load_state_dict(
        probability_checkpoint["sigma_head_state_dict"]
    )
    wrapper.eval()

    stats = checkpoint_stats(mean_checkpoint, device)
    params = KinematicBicycleParams(**mean_checkpoint["params"])
    calibration_files = read_paths(run_dir / "calibration_files.txt")
    test_files = read_paths(run_dir / "probability_test_files.txt")
    _, calibration_loader = make_loader(
        calibration_files,
        model_args,
        params,
        args.eval_batch_size,
        0,
        False,
    )
    _, test_loader = make_loader(
        test_files,
        model_args,
        params,
        args.eval_batch_size,
        0,
        False,
    )
    calibration_error, calibration_sigma, _ = collect_predictions(
        wrapper, calibration_loader, stats, device
    )
    test_error, test_sigma, maximum_mean_difference = collect_predictions(
        wrapper, test_loader, stats, device, compare_mean=True
    )

    global_temperature = channel_temperature(
        calibration_error, calibration_sigma
    )
    horizon_temperature = horizon_channel_temperature(
        calibration_error, calibration_sigma
    )
    global_test_sigma = test_sigma * global_temperature.view(1, 1, -1)
    horizon_test_sigma = test_sigma * horizon_temperature.view(
        1, test_sigma.shape[1], test_sigma.shape[2]
    )
    residual_std = stats["residual"][1].cpu()
    metrics = {
        "learned_raw": all_horizon_metrics(
            test_error, test_sigma, residual_std
        ),
        "global_temperature": all_horizon_metrics(
            test_error, global_test_sigma, residual_std
        ),
        "horizon_temperature": all_horizon_metrics(
            test_error, horizon_test_sigma, residual_std
        ),
    }

    calibration_z = (calibration_error / calibration_sigma).numpy()
    test_z = (test_error / test_sigma).numpy()
    calibration_diagnostics = split_diagnostics(calibration_z)
    test_diagnostics = split_diagnostics(test_z)
    drift = drift_diagnostics(calibration_z, test_z)

    plot_histograms_and_qq(
        calibration_z, test_z, output_dir / "standardized_error_hist_qq.png"
    )
    plot_horizon_diagnostics(
        calibration_diagnostics,
        test_diagnostics,
        drift,
        output_dir / "horizon_error_diagnostics.png",
    )
    plot_temperature(
        horizon_temperature.numpy(), output_dir / "horizon_temperature.png"
    )
    plot_coverage_comparison(metrics, output_dir / "coverage_by_horizon.png")

    horizon_metrics = metrics["horizon_temperature"]
    full_coverage_errors = {
        label: horizon_metrics["all_steps"]["coverage"][label][
            "absolute_error_mean"
        ]
        for label in COVERAGE_LEVELS
    }
    selected_coverage_errors = {
        str(horizon): {
            label: horizon_metrics["horizons"][str(horizon)]["coverage"][
                label
            ]["absolute_error_mean"]
            for label in COVERAGE_LEVELS
        }
        for horizon in SELECTED_HORIZONS
    }
    temperatures = horizon_temperature.numpy()
    stored_global_temperature = np.asarray(
        source_summary["calibration"]["learned_temperature"]
    )
    gates = {
        "full_coverage_within_5pp": all(
            value <= 0.05 for value in full_coverage_errors.values()
        ),
        "nll_within_global_plus_0.01": (
            horizon_metrics["all_steps"]["mean"]["nll"]
            <= metrics["global_temperature"]["all_steps"]["mean"]["nll"]
            + 0.01
        ),
        "selected_horizon_coverage_within_10pp": all(
            value <= 0.10
            for horizon_result in selected_coverage_errors.values()
            for value in horizon_result.values()
        ),
        "temperature_finite_positive": bool(
            np.all(np.isfinite(temperatures)) and np.all(temperatures > 0)
        ),
    }
    gates["all_passed"] = all(gates.values())
    temperature_summary = {
        "minimum": float(np.min(temperatures)),
        "maximum": float(np.max(temperatures)),
        "mean": float(np.mean(temperatures)),
        "count_below_0.5": int(np.sum(temperatures < 0.5)),
        "count_above_2.0": int(np.sum(temperatures > 2.0)),
        "values": temperatures.tolist(),
    }
    summary = {
        "protocol": "nuplan_horizon_probability_calibration_v1",
        "source_run": str(run_dir.resolve()),
        "source_summary_sha256": sha256(run_dir / "summary.json"),
        "mean_checkpoint": str(mean_checkpoint_path.resolve()),
        "probability_checkpoint": str(probability_checkpoint_path.resolve()),
        "args": vars(args),
        "calibration_files": len(calibration_files),
        "test_files": len(test_files),
        "calibration_episodes": int(calibration_error.shape[0]),
        "test_episodes": int(test_error.shape[0]),
        "maximum_mean_difference": maximum_mean_difference,
        "global_temperature": global_temperature.tolist(),
        "stored_global_temperature": stored_global_temperature.tolist(),
        "global_temperature_max_abs_reproduction_error": float(
            np.max(
                np.abs(
                    global_temperature.numpy() - stored_global_temperature
                )
            )
        ),
        "horizon_temperature": temperature_summary,
        "metrics": metrics,
        "diagnostics": {
            "calibration": calibration_diagnostics,
            "test": test_diagnostics,
            "calibration_test_drift": drift,
        },
        "gate_inputs": {
            "full_coverage_errors": full_coverage_errors,
            "selected_horizon_coverage_errors": selected_coverage_errors,
            "global_temperature_nll": metrics["global_temperature"][
                "all_steps"
            ]["mean"]["nll"],
            "horizon_temperature_nll": horizon_metrics["all_steps"]["mean"][
                "nll"
            ],
        },
        "gates": gates,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
        {
            "protocol": summary["protocol"],
            "source_run": summary["source_run"],
            "global_temperature": global_temperature,
            "horizon_temperature": horizon_temperature,
            "gates": gates,
        },
        output_dir / "calibration.pt",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "mean_invariance": maximum_mean_difference,
                "global_temperature_reproduction_error": summary[
                    "global_temperature_max_abs_reproduction_error"
                ],
                "temperature_summary": {
                    key: value
                    for key, value in temperature_summary.items()
                    if key != "values"
                },
                "gate_inputs": summary["gate_inputs"],
                "gates": gates,
                "pooled_test_diagnostics": test_diagnostics["channels"],
                "pooled_drift": drift["channels"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
