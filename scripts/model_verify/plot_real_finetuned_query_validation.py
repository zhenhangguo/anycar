#!/usr/bin/env python3
"""Plot the formal real fine-tuned Query model on its validation split."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from car_foundation.kinematic_residual import (
    rollout_kinematic_residual_consistent,
)
from evaluate_real_kinematic_residual import (
    checkpoint_params,
    checkpoint_stats,
)
from finetune_real_kinematic_residual import build_filtered_view
from train_kinematic_residual_ablation import (
    forward_independent_history,
    make_model,
    prepare_batch,
    prepare_nominal_query,
)


CHECKPOINT_SHA256 = (
    "711b3721dc7526b2e71a8b1c13298852a7f17db0630843d2feeab351787bd8fe"
)
VALIDATION_FILES_SHA256 = (
    "a7db05175cf3cb2169194cd82d9b7941aa83262b7e796ecb8af505500961adf0"
)
EXPECTED_EPISODES = 22763
EXPECTED_FILES = 3846
EXPECTED_WINDOWS_PER_FILE = 6
SELECTED_HORIZONS = (1, 5, 10, 20, 50)
METRICS = ("position", "yaw", "vx", "yawrate")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the formal real fine-tuned Query validation split."
    )
    parser.add_argument(
        "--run-dir",
        default=os.path.join(
            REPO_ROOT,
            "outputs/formal_real_finetune_residual_vs_query/20260717T173715",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--output-name",
        default="query_validation_visualization",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_paths(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def verify_inputs(run_dir):
    checkpoint_path = run_dir / "query_best.pt"
    files_path = run_dir / "val_files.txt"
    actual_hashes = {
        "checkpoint": sha256(checkpoint_path),
        "validation_files": sha256(files_path),
    }
    expected_hashes = {
        "checkpoint": CHECKPOINT_SHA256,
        "validation_files": VALIDATION_FILES_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            f"Formal input hash mismatch: {actual_hashes} != {expected_hashes}"
        )
    files = read_paths(files_path)
    if len(files) != EXPECTED_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_FILES} validation files, got {len(files)}"
        )
    return checkpoint_path, files, actual_hashes


def predict_validation(model, loader, stats, params, device):
    predictions = []
    truths = []
    initial_states = []
    actions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="query validation rollout"):
            history, action, context, mask = prepare_batch(
                batch,
                stats["history"][0],
                stats["history"][1],
                stats["context"][0],
                stats["context"][1],
                device,
            )
            nominal_state, nominal_transition = prepare_nominal_query(
                batch, stats, device
            )
            prediction_normalized = forward_independent_history(
                model,
                history,
                action,
                context,
                mask,
                nominal_state,
                nominal_transition,
            )
            residual_mean, residual_std = stats["residual"]
            residual = (
                prediction_normalized * residual_std + residual_mean
            )
            initial = batch["initial_state"].to(device)
            prediction = rollout_kinematic_residual_consistent(
                initial,
                action,
                residual,
                params,
            )
            predictions.append(prediction.cpu())
            truths.append(batch["truth"].cpu())
            initial_states.append(batch["initial_state"].cpu())
            actions.append(batch["action"].cpu())
    return {
        "prediction": torch.cat(predictions).numpy(),
        "truth": torch.cat(truths).numpy(),
        "initial_state": torch.cat(initial_states).numpy(),
        "action": torch.cat(actions).numpy(),
    }


def wrapped_yaw_difference(prediction, truth):
    difference = prediction[..., 2] - truth[..., 2]
    return np.arctan2(np.sin(difference), np.cos(difference))


def squared_errors(prediction, truth):
    difference = prediction - truth
    return {
        "position": np.square(difference[..., 0])
        + np.square(difference[..., 1]),
        "yaw": np.square(wrapped_yaw_difference(prediction, truth)),
        "vx": np.square(difference[..., 3]),
        "yawrate": np.square(difference[..., 4]),
    }


def summarize_predictions(prediction, truth):
    errors = squared_errors(prediction, truth)
    episode = {
        metric: np.sqrt(np.mean(values, axis=1))
        for metric, values in errors.items()
    }
    horizon = {
        metric: np.sqrt(np.mean(values, axis=0))
        for metric, values in errors.items()
    }
    all_steps = {
        f"{metric}_rmse": float(np.sqrt(np.mean(values)))
        for metric, values in errors.items()
    }
    selected = {
        str(horizon_index): {
            f"{metric}_rmse": float(values[horizon_index - 1])
            for metric, values in horizon.items()
        }
        for horizon_index in SELECTED_HORIZONS
    }
    return errors, episode, horizon, all_steps, selected


def reproduction_audit(checkpoint, all_steps, selected_horizons):
    expected = checkpoint["val_metrics"]
    differences = {}
    for metric in METRICS:
        key = f"{metric}_rmse"
        differences[f"all_steps/{key}"] = abs(
            all_steps[key] - expected["all_steps"][key]
        )
    for horizon in SELECTED_HORIZONS:
        for metric in METRICS:
            key = f"{metric}_rmse"
            differences[f"horizon_{horizon}/{key}"] = abs(
                selected_horizons[str(horizon)][key]
                - expected["horizons"][str(horizon)][key]
            )
    maximum = max(differences.values())
    return {
        "expected": {
            "all_steps": expected["all_steps"],
            "horizons": {
                str(horizon): expected["horizons"][str(horizon)]
                for horizon in SELECTED_HORIZONS
            },
        },
        "absolute_differences": differences,
        "maximum_absolute_difference": maximum,
        "within_1e-6": maximum <= 1e-6,
    }


def metric_percentiles(episode_metrics):
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {
        metric: {
            f"{probability:.2f}": float(np.quantile(values, probability))
            for probability in probabilities
        }
        for metric, values in episode_metrics.items()
    }


def nearest_quantile_index(values, probability):
    target = np.quantile(values, probability)
    return int(np.argmin(np.abs(values - target)))


def select_representatives(episode_metrics, arrays):
    position = episode_metrics["position"]
    selections = []
    for probability in (
        0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
    ):
        selections.append(
            (
                f"position_q{int(probability * 100):02d}",
                nearest_quantile_index(position, probability),
            )
        )
    for metric in ("yaw", "vx", "yawrate"):
        for probability in (0.95, 0.99, 1.0):
            label = (
                f"worst_{metric}"
                if probability == 1.0
                else f"{metric}_q{int(probability * 100):02d}"
            )
            selections.append(
                (
                    label,
                    nearest_quantile_index(
                        episode_metrics[metric], probability
                    ),
                )
            )

    peak_steering = np.max(np.abs(arrays["action"][..., 1]), axis=1)
    peak_truth_yawrate = np.max(
        np.abs(arrays["truth"][..., 4]), axis=1
    )
    for label, values in (
        ("steering_peak", peak_steering),
        ("truth_yawrate_peak", peak_truth_yawrate),
    ):
        selections.extend(
            (
                (f"{label}_q99", nearest_quantile_index(values, 0.99)),
                (f"{label}_max", int(np.argmax(values))),
            )
        )

    unique = []
    used = set()
    for label, index in selections:
        if index not in used:
            unique.append((label, index))
            used.add(index)
    return unique[:28]


def write_episode_metrics(
    output_path,
    episode_metrics,
    raw_indices,
    files,
    windows_per_file,
):
    with open(output_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "validation_index",
                "raw_window_index",
                "source_file",
                "chunk_index",
                "position_rmse_m",
                "yaw_rmse_rad",
                "vx_rmse_mps",
                "yawrate_rmse_radps",
            )
        )
        for validation_index, raw_index in enumerate(raw_indices):
            file_index = int(raw_index) // windows_per_file
            chunk_index = int(raw_index) % windows_per_file
            writer.writerow(
                (
                    validation_index,
                    int(raw_index),
                    files[file_index],
                    chunk_index,
                    episode_metrics["position"][validation_index],
                    episode_metrics["yaw"][validation_index],
                    episode_metrics["vx"][validation_index],
                    episode_metrics["yawrate"][validation_index],
                )
            )


def plot_episode_distributions(episode_metrics, output_path):
    units = {
        "position": "m",
        "yaw": "rad",
        "vx": "m/s",
        "yawrate": "rad/s",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for metric, axis in zip(METRICS, axes.reshape(-1)):
        values = episode_metrics[metric]
        axis.hist(values, bins=100, alpha=0.8)
        for probability, linestyle in ((0.5, "--"), (0.9, ":"), (0.99, "-.")):
            value = np.quantile(values, probability)
            axis.axvline(
                value,
                linestyle=linestyle,
                label=f"q{int(probability * 100)}={value:.5g}",
            )
        axis.set_title(f"Episode {metric} RMSE")
        axis.set_xlabel(units[metric])
        axis.set_ylabel("validation windows")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_horizon_metrics(horizon_metrics, output_path):
    units = {
        "position": "m",
        "yaw": "rad",
        "vx": "m/s",
        "yawrate": "rad/s",
    }
    horizons = np.arange(1, 51)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for metric, axis in zip(METRICS, axes.reshape(-1)):
        axis.plot(horizons, horizon_metrics[metric], marker="o", markersize=3)
        for selected in SELECTED_HORIZONS:
            axis.scatter(
                selected,
                horizon_metrics[metric][selected - 1],
                color="tab:red",
                s=24,
            )
        axis.set_title(f"Validation {metric} RMSE by horizon")
        axis.set_xlabel("future step (0.05 s)")
        axis.set_ylabel(units[metric])
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def wrap_numpy(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def unwrap_future_from_history(history_yaw, future_yaw):
    """Unwrap future yaw on the exact branch selected by the history."""
    history_unwrapped = np.unwrap(history_yaw)
    increments = wrap_numpy(
        np.diff(np.concatenate(([history_yaw[-1]], future_yaw)))
    )
    future_unwrapped = history_unwrapped[-1] + np.cumsum(increments)
    return history_unwrapped, np.concatenate(
        ([history_unwrapped[-1]], future_unwrapped)
    )


def continuity_arrays(prediction, truth, initial_state, dt):
    result = {}
    for name, states in (("truth", truth), ("prediction", prediction)):
        yaw = np.concatenate(
            (initial_state[:, None, 2], states[..., 2]), axis=1
        )
        yawrate = np.concatenate(
            (initial_state[:, None, 4], states[..., 4]), axis=1
        )
        result[name] = {
            "yaw_increment": wrap_numpy(np.diff(yaw, axis=1)),
            "yawrate_change": np.diff(yawrate, axis=1),
        }
        result[name]["yaw_integration_residual"] = wrap_numpy(
            result[name]["yaw_increment"] - yawrate[:, :-1] * dt
        )
    return result


def quantile_summary(values, scale=1.0):
    probabilities = (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
    absolute = np.abs(values * scale).reshape(-1)
    return {
        f"q{probability * 100:g}": float(
            np.quantile(absolute, probability)
        )
        for probability in probabilities
    }


def continuity_summary(prediction, truth, initial_state, dt):
    values = continuity_arrays(prediction, truth, initial_state, dt)
    summary = {}
    for name in ("truth", "prediction"):
        yawrate_change = values[name]["yawrate_change"]
        integration_residual = values[name]["yaw_integration_residual"]
        summary[name] = {
            "yawrate_change_all_abs_degps": quantile_summary(
                yawrate_change, 180.0 / np.pi
            ),
            "yawrate_change_boundary_abs_degps": quantile_summary(
                yawrate_change[:, 0], 180.0 / np.pi
            ),
            "yawrate_change_internal_abs_degps": quantile_summary(
                yawrate_change[:, 1:], 180.0 / np.pi
            ),
            "yaw_integration_residual_abs_deg": quantile_summary(
                integration_residual, 180.0 / np.pi
            ),
        }
    truth_residual_deg = np.abs(
        values["truth"]["yaw_integration_residual"] * 180.0 / np.pi
    )
    summary["truth_yaw_integration_thresholds"] = {
        f"above_{threshold:.2f}_deg_frame_fraction": float(
            np.mean(truth_residual_deg > threshold)
        )
        for threshold in (0.01, 0.02, 0.05, 0.1)
    }
    summary["truth_yaw_integration_thresholds"].update(
        {
            f"above_{threshold:.2f}_deg_window_fraction": float(
                np.mean(np.max(truth_residual_deg, axis=1) > threshold)
            )
            for threshold in (0.01, 0.02, 0.05, 0.1)
        }
    )
    return summary


def plot_continuity_diagnostics(
    prediction, truth, initial_state, dt, output_path
):
    values = continuity_arrays(prediction, truth, initial_state, dt)
    scale = 180.0 / np.pi
    truth_change = values["truth"]["yawrate_change"] * scale
    prediction_change = values["prediction"]["yawrate_change"] * scale
    truth_residual = (
        values["truth"]["yaw_integration_residual"] * scale
    )
    prediction_residual = (
        values["prediction"]["yaw_integration_residual"] * scale
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, truth_values, prediction_values, title in (
        (
            axes[0, 0],
            truth_change[:, 0],
            prediction_change[:, 0],
            "Current to first-future yawrate change",
        ),
        (
            axes[0, 1],
            truth_change[:, 1:].reshape(-1),
            prediction_change[:, 1:].reshape(-1),
            "Internal future yawrate change",
        ),
    ):
        limit = max(
            np.quantile(np.abs(truth_values), 0.999),
            np.quantile(np.abs(prediction_values), 0.999),
        )
        axis.hist(
            np.abs(truth_values), bins=100, range=(0.0, limit),
            density=True, alpha=0.55, label="measured truth",
        )
        axis.hist(
            np.abs(prediction_values), bins=100, range=(0.0, limit),
            density=True, alpha=0.55, label="Query prediction",
        )
        axis.set(
            title=title,
            xlabel="absolute change [deg/s per 0.05 s]",
            ylabel="density",
        )
        axis.legend()
        axis.grid(alpha=0.2)

    residual_limit = np.quantile(np.abs(truth_residual), 0.999)
    axes[1, 0].hist(
        np.abs(truth_residual).reshape(-1), bins=100,
        range=(0.0, residual_limit), density=True, alpha=0.6,
        label="measured truth",
    )
    axes[1, 0].hist(
        np.abs(prediction_residual).reshape(-1), bins=100,
        range=(0.0, residual_limit), density=True, alpha=0.6,
        label="Query prediction",
    )
    axes[1, 0].set(
        title="Yaw integration residual",
        xlabel="|dYaw - previous yawrate * dt| [deg]",
        ylabel="density",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    signed_limit = max(
        np.quantile(np.abs(truth_change[:, 0]), 0.999),
        np.quantile(np.abs(prediction_change[:, 0]), 0.999),
    )
    axes[1, 1].hexbin(
        truth_change[:, 0], prediction_change[:, 0],
        gridsize=70, mincnt=1, bins="log", cmap="viridis",
    )
    axes[1, 1].plot(
        (-signed_limit, signed_limit),
        (-signed_limit, signed_limit),
        linestyle="--", color="tab:red", label="prediction = truth",
    )
    axes[1, 1].set(
        title="Signed first-step yawrate change",
        xlabel="measured truth [deg/s]",
        ylabel="Query prediction [deg/s]",
        xlim=(-signed_limit, signed_limit),
        ylim=(-signed_limit, signed_limit),
    )
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def safe_label(label):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", label)


def positions_in_current_body(points, initial_state):
    difference = points - initial_state[:2]
    cosine = math.cos(float(initial_state[2]))
    sine = math.sin(float(initial_state[2]))
    return np.column_stack(
        (
            difference[:, 0] * cosine + difference[:, 1] * sine,
            -difference[:, 0] * sine + difference[:, 1] * cosine,
        )
    )


def plot_representative(
    output_path,
    label,
    validation_index,
    raw_index,
    source_file,
    chunk_index,
    history,
    prediction,
    truth,
    initial_state,
    action,
    episode_metrics,
    dt,
):
    future_time = np.arange(1, prediction.shape[0] + 1) * dt
    state_time = np.concatenate(([0.0], future_time))
    history_time = (
        np.arange(history.shape[0]) - (history.shape[0] - 1)
    ) * dt
    prediction_path = positions_in_current_body(
        np.vstack((initial_state[:2], prediction[:, :2])), initial_state
    )
    truth_path = positions_in_current_body(
        np.vstack((initial_state[:2], truth[:, :2])), initial_state
    )
    history_path = positions_in_current_body(history[-51:, :2], initial_state)

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0, 0].plot(
        history_path[:, 0], history_path[:, 1], color="0.65", label="last 2.5 s"
    )
    axes[0, 0].plot(
        truth_path[:, 0], truth_path[:, 1], marker="o", markersize=3,
        label="future truth"
    )
    axes[0, 0].plot(
        prediction_path[:, 0], prediction_path[:, 1], marker="x", markersize=4,
        label="Query prediction"
    )
    axes[0, 0].scatter(
        0.0, 0.0, color="black", s=35, label="current"
    )
    axes[0, 0].set(
        title="Current-body XY rollout",
        xlabel="longitudinal [m]",
        ylabel="lateral [m]",
    )
    axes[0, 0].axis("equal")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)
    position_error = np.linalg.norm(
        prediction[:, :2] - truth[:, :2], axis=1
    )
    inset = axes[0, 0].inset_axes((0.57, 0.08, 0.38, 0.26))
    inset.plot(future_time, position_error, color="tab:red")
    inset.set(title="position error", xlabel="s", ylabel="m")
    inset.grid(alpha=0.2)

    axes[0, 1].plot(history_time, history[:, 3], color="0.65", label="history vx")
    axes[0, 1].plot(
        state_time,
        np.concatenate(([initial_state[3]], truth[:, 3])),
        marker="o", markersize=2.5, label="measured truth vx",
    )
    axes[0, 1].plot(
        state_time,
        np.concatenate(([initial_state[3]], prediction[:, 3])),
        label="Query vx",
    )
    axes[0, 1].set(title="Longitudinal velocity", xlabel="time [s]", ylabel="m/s")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    history_yaw, truth_yaw = unwrap_future_from_history(
        history[:, 2], truth[:, 2]
    )
    _, prediction_yaw = unwrap_future_from_history(
        history[:, 2], prediction[:, 2]
    )
    truth_yawrate_with_current = np.concatenate(
        ([initial_state[4]], truth[:, 4])
    )
    integrated_truth_yaw = np.concatenate(
        (
            [history_yaw[-1]],
            history_yaw[-1]
            + np.cumsum(truth_yawrate_with_current[:-1] * dt),
        )
    )
    axes[1, 0].plot(
        history_time, np.rad2deg(history_yaw), color="0.65",
        label="history measured yaw",
    )
    axes[1, 0].plot(
        state_time, np.rad2deg(truth_yaw), marker="o", markersize=2.5,
        label="measured truth yaw",
    )
    axes[1, 0].plot(
        state_time, np.rad2deg(prediction_yaw), label="Query yaw"
    )
    axes[1, 0].plot(
        state_time, np.rad2deg(integrated_truth_yaw), linestyle="--",
        label="yaw integrated from truth yawrate",
    )
    axes[1, 0].set(title="Yaw", xlabel="time [s]", ylabel="deg")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(
        history_time,
        np.rad2deg(history[:, 4]),
        color="0.65",
        label="history yawrate",
    )
    axes[1, 1].plot(
        state_time,
        np.rad2deg(np.concatenate(([initial_state[4]], truth[:, 4]))),
        marker="o", markersize=2.5, label="measured truth yawrate",
    )
    axes[1, 1].plot(
        state_time,
        np.rad2deg(np.concatenate(([initial_state[4]], prediction[:, 4]))),
        label="Query yawrate",
    )
    axes[1, 1].set(
        title="Yaw rate and future steering",
        xlabel="time [s]",
        ylabel="yawrate [deg/s]",
    )
    steer_axis = axes[1, 1].twinx()
    steer_axis.plot(
        future_time,
        np.rad2deg(action[:, 1]),
        color="tab:green",
        alpha=0.5,
        label="steering command",
    )
    steer_axis.set_ylabel("steering command [deg]", color="tab:green")
    lines, labels = axes[1, 1].get_legend_handles_labels()
    steer_lines, steer_labels = steer_axis.get_legend_handles_labels()
    axes[1, 1].legend(lines + steer_lines, labels + steer_labels, loc="best")
    axes[1, 1].grid(alpha=0.25)

    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.axvline(0.0, color="black", linestyle=":", linewidth=1.0)

    truth_first_yawrate_jump = np.rad2deg(
        truth[0, 4] - initial_state[4]
    )
    prediction_first_yawrate_jump = np.rad2deg(
        prediction[0, 4] - initial_state[4]
    )
    truth_yaw_integration_residual = wrap_numpy(
        np.diff(np.concatenate(([initial_state[2]], truth[:, 2])))
        - truth_yawrate_with_current[:-1] * dt
    )
    continuity_text = (
        f"first dYawrate: truth={truth_first_yawrate_jump:+.3f}deg/s, "
        f"Query={prediction_first_yawrate_jump:+.3f}deg/s; "
        f"truth yaw-integral max residual="
        f"{np.max(np.abs(np.rad2deg(truth_yaw_integration_residual))):.3f}deg"
    )

    metrics_text = " ".join(
        (
            f"pos={episode_metrics['position'][validation_index]:.4f}m",
            f"yaw={episode_metrics['yaw'][validation_index]:.5f}rad",
            f"vx={episode_metrics['vx'][validation_index]:.4f}m/s",
            f"yawrate={episode_metrics['yawrate'][validation_index]:.5f}rad/s",
        )
    )
    figure.suptitle(
        f"{label} | validation={validation_index} raw={raw_index} chunk={chunk_index}\n"
        f"{Path(source_file).name}\n{metrics_text}\n{continuity_text}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the formal Query model")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir)
    checkpoint_path, files, input_hashes = verify_inputs(run_dir)
    device = torch.device(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("variant") != "query":
        raise RuntimeError("Checkpoint is not a Query model")
    if checkpoint.get("protocol") != "real_finetune_consistent_v2":
        raise RuntimeError("Unexpected fine-tune checkpoint protocol")

    filter_args = SimpleNamespace(**checkpoint["real_finetune_args"])
    filter_args.history_length = int(checkpoint["args"]["history_length"])
    filter_args.prediction_length = int(
        checkpoint["args"]["prediction_length"]
    )
    filter_args.max_episodes_per_split = 0
    params = checkpoint_params(checkpoint, filter_args.dt)
    validation_dataset, audit = build_filtered_view(
        files, filter_args, params, "validation"
    )
    if len(validation_dataset) != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Expected {EXPECTED_EPISODES} episodes, got {len(validation_dataset)}"
        )
    if audit["total_windows"] != EXPECTED_FILES * EXPECTED_WINDOWS_PER_FILE:
        raise RuntimeError("Validation windows-per-file assumption changed")

    loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    model_args = SimpleNamespace(**checkpoint["args"])
    model_args.device = args.device
    model = make_model(model_args, device, variant="query")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = checkpoint_stats(checkpoint, device)

    arrays = predict_validation(model, loader, stats, params, device)
    (
        _,
        episode_metrics,
        horizon_metrics,
        all_steps,
        selected_horizons,
    ) = summarize_predictions(arrays["prediction"], arrays["truth"])
    reproduction = reproduction_audit(
        checkpoint, all_steps, selected_horizons
    )
    if not reproduction["within_1e-6"]:
        raise RuntimeError(
            "Validation metrics do not reproduce checkpoint: "
            f"{reproduction['maximum_absolute_difference']}"
        )

    output_dir = (
        run_dir
        / args.output_name
        / datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_indices = validation_dataset.indices.cpu().numpy()
    write_episode_metrics(
        output_dir / "episode_metrics.csv",
        episode_metrics,
        raw_indices,
        files,
        EXPECTED_WINDOWS_PER_FILE,
    )
    plot_episode_distributions(
        episode_metrics, output_dir / "episode_rmse_distribution.png"
    )
    plot_horizon_metrics(
        horizon_metrics, output_dir / "horizon_rmse.png"
    )
    continuity = continuity_summary(
        arrays["prediction"],
        arrays["truth"],
        arrays["initial_state"],
        filter_args.dt,
    )
    plot_continuity_diagnostics(
        arrays["prediction"],
        arrays["truth"],
        arrays["initial_state"],
        filter_args.dt,
        output_dir / "yaw_continuity_diagnostics.png",
    )

    selections = select_representatives(episode_metrics, arrays)
    base_dataset = validation_dataset.dataset.dataset
    representative_rows = []
    representative_arrays = {}
    for order, (label, validation_index) in enumerate(selections):
        raw_index = int(raw_indices[validation_index])
        file_index = raw_index // EXPECTED_WINDOWS_PER_FILE
        chunk_index = raw_index % EXPECTED_WINDOWS_PER_FILE
        source_file = files[file_index]
        history = base_dataset.data[
            raw_index, : filter_args.history_length + 1, :6
        ][:, [0, 1, 2, 3, 5]].cpu().numpy()
        filename = (
            f"rollout_{order:02d}_{safe_label(label)}_"
            f"validation_{validation_index:05d}.png"
        )
        plot_representative(
            output_dir / filename,
            label,
            validation_index,
            raw_index,
            source_file,
            chunk_index,
            history,
            arrays["prediction"][validation_index],
            arrays["truth"][validation_index],
            arrays["initial_state"][validation_index],
            arrays["action"][validation_index],
            episode_metrics,
            filter_args.dt,
        )
        row = {
            "label": label,
            "validation_index": validation_index,
            "raw_window_index": raw_index,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "figure": filename,
            "metrics": {
                f"{metric}_rmse": float(
                    episode_metrics[metric][validation_index]
                )
                for metric in METRICS
            },
            "continuity": {
                "truth_first_yawrate_change_degps": float(
                    np.rad2deg(
                        arrays["truth"][validation_index, 0, 4]
                        - arrays["initial_state"][validation_index, 4]
                    )
                ),
                "prediction_first_yawrate_change_degps": float(
                    np.rad2deg(
                        arrays["prediction"][validation_index, 0, 4]
                        - arrays["initial_state"][validation_index, 4]
                    )
                ),
            },
        }
        representative_rows.append(row)
        prefix = f"sample_{order:02d}"
        representative_arrays[f"{prefix}_history"] = history
        representative_arrays[f"{prefix}_prediction"] = arrays[
            "prediction"
        ][validation_index]
        representative_arrays[f"{prefix}_truth"] = arrays["truth"][
            validation_index
        ]
        representative_arrays[f"{prefix}_action"] = arrays["action"][
            validation_index
        ]
        representative_arrays[f"{prefix}_initial_state"] = arrays[
            "initial_state"
        ][validation_index]
    np.savez_compressed(
        output_dir / "representative_predictions.npz",
        **representative_arrays,
    )

    all_finite = bool(
        all(np.all(np.isfinite(value)) for value in arrays.values())
        and all(
            np.all(np.isfinite(value))
            for value in episode_metrics.values()
        )
    )
    gates = {
        "checkpoint_hash_matches": input_hashes["checkpoint"]
        == CHECKPOINT_SHA256,
        "validation_files_hash_matches": input_hashes["validation_files"]
        == VALIDATION_FILES_SHA256,
        "expected_validation_episodes": len(validation_dataset)
        == EXPECTED_EPISODES,
        "metrics_reproduced_within_1e-6": reproduction["within_1e-6"],
        "all_outputs_finite": all_finite,
    }
    gates["all_passed"] = all(gates.values())
    summary = {
        "protocol": "real_finetuned_query_validation_visualization_v2",
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "input_hashes": input_hashes,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection_value": float(checkpoint["selection_value"]),
        "args": vars(args),
        "validation": {
            "minute_groups": 34,
            "files": len(files),
            "episodes": len(validation_dataset),
            "audit": audit,
            "diagnostic_only": True,
            "reason": "Validation was used for checkpoint selection.",
        },
        "metrics": {
            "all_steps": all_steps,
            "selected_horizons": selected_horizons,
            "all_horizons": {
                str(index + 1): {
                    f"{metric}_rmse": float(values[index])
                    for metric, values in horizon_metrics.items()
                }
                for index in range(filter_args.prediction_length)
            },
            "episode_percentiles": metric_percentiles(episode_metrics),
        },
        "checkpoint_metric_reproduction": reproduction,
        "yaw_continuity": continuity,
        "representative_samples": representative_rows,
        "gates": gates,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "validation": summary["validation"],
                "all_steps": all_steps,
                "episode_percentiles": summary["metrics"][
                    "episode_percentiles"
                ],
                "reproduction": reproduction,
                "yaw_continuity": continuity,
                "representative_samples": representative_rows,
                "gates": gates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
