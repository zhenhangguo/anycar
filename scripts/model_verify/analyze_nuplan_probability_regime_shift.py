#!/usr/bin/env python3
"""Attribute nuPlan probability calibration/test shift to observable regimes."""

import argparse
import hashlib
import json
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
from scipy.stats import (
    ks_2samp,
    pearsonr,
    spearmanr,
    ttest_ind,
    wasserstein_distance,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from car_foundation.kinematic_residual import KinematicBicycleParams
from car_foundation.probabilistic_residual import FrozenMeanGaussianResidual
from car_foundation.probability_shift import (
    assign_bins,
    gap_decomposition,
    quantile_edges,
    standardized_mean_difference,
)
from train_kinematic_residual_ablation import (
    forward_independent_history,
    make_model,
)
from validate_nuplan_probabilistic_query import (
    CHANNELS,
    forward_batch,
    make_loader,
)


VEHICLE_FEATURES = (
    "initial_vx",
    "initial_yawrate",
    "initial_throttle",
    "initial_steer",
    "future_mean_steer",
    "future_mean_abs_steer",
    "future_max_abs_steer",
    "future_steer_std",
    "future_mean_throttle",
    "future_throttle_std",
)
ALL_FEATURES = VEHICLE_FEATURES + (
    "chunk_index",
    "file_timestamp_seconds",
    "file_log_id",
)
KEY_TARGETS = (
    "mean_dy_body",
    "mean_dyawrate",
    "step1_dy_body",
    "step1_dyawrate",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze observable regime shift for nuPlan uncertainty."
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
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-name", default="regime_shift")
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


def parse_file_metadata(files, windows_per_file):
    log_ids = []
    seconds = []
    pattern = re.compile(
        r"log_(\d+)_\d{4}-\d{2}-\d{2}T(\d{2}):(\d{2}):(\d{2}\.\d+)\.pkl$"
    )
    for path in files:
        match = pattern.search(path)
        if match is None:
            raise ValueError(f"Cannot parse nuPlan filename metadata: {path}")
        log_ids.append(float(match.group(1)))
        seconds.append(
            int(match.group(2)) * 3600
            + int(match.group(3)) * 60
            + float(match.group(4))
        )
    return {
        "file_log_id": np.repeat(log_ids, windows_per_file),
        "file_timestamp_seconds": np.repeat(seconds, windows_per_file),
        "chunk_index": np.tile(
            np.arange(windows_per_file), len(files)
        ).astype(np.float64),
        "file_index": np.repeat(
            np.arange(len(files)), windows_per_file
        ).astype(np.int64),
    }


def collect_split(
    wrapper,
    loader,
    files,
    stats,
    device,
    compare_mean=False,
):
    errors = []
    sigmas = []
    maximum_mean_difference = 0.0
    feature_batches = {name: [] for name in VEHICLE_FEATURES}
    wrapper.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="collect regimes", leave=False):
            mean, sigma, target, prepared = forward_batch(
                wrapper, batch, stats, device
            )
            if compare_mean:
                reference = forward_independent_history(
                    wrapper.mean_model, *prepared
                )
                maximum_mean_difference = max(
                    maximum_mean_difference,
                    torch.max(torch.abs(reference - mean)).item(),
                )
            error = target - mean
            action = batch["action"]
            initial_state = batch["initial_state"]
            context = batch["current_context"]
            errors.append(error.cpu())
            sigmas.append(sigma.cpu())
            feature_batches["initial_vx"].append(initial_state[:, 3])
            feature_batches["initial_yawrate"].append(initial_state[:, 4])
            feature_batches["initial_throttle"].append(context[:, 2])
            feature_batches["initial_steer"].append(context[:, 3])
            future_steer = action[:, :, 1]
            future_throttle = action[:, :, 0]
            feature_batches["future_mean_steer"].append(
                future_steer.mean(dim=1)
            )
            feature_batches["future_mean_abs_steer"].append(
                future_steer.abs().mean(dim=1)
            )
            feature_batches["future_max_abs_steer"].append(
                future_steer.abs().amax(dim=1)
            )
            feature_batches["future_steer_std"].append(
                future_steer.std(dim=1, unbiased=False)
            )
            feature_batches["future_mean_throttle"].append(
                future_throttle.mean(dim=1)
            )
            feature_batches["future_throttle_std"].append(
                future_throttle.std(dim=1, unbiased=False)
            )
    error = torch.cat(errors).numpy()
    sigma = torch.cat(sigmas).numpy()
    episode_count = error.shape[0]
    if episode_count % len(files) != 0:
        raise RuntimeError("Episode count is not divisible by file count")
    windows_per_file = episode_count // len(files)
    if windows_per_file != 6:
        raise RuntimeError(
            f"Expected six windows per file, got {windows_per_file}"
        )
    features = {
        name: torch.cat(values).numpy()
        for name, values in feature_batches.items()
    }
    features.update(parse_file_metadata(files, windows_per_file))
    z = error / sigma
    targets = {}
    for channel_index, channel in enumerate(CHANNELS):
        targets[f"mean_{channel}"] = z[..., channel_index].mean(axis=1)
        targets[f"step1_{channel}"] = z[:, 0, channel_index]
    return {
        "error": error,
        "sigma": sigma,
        "z": z,
        "features": features,
        "targets": targets,
        "windows_per_file": windows_per_file,
        "maximum_mean_difference": maximum_mean_difference,
    }


def feature_shift(calibration, test):
    result = {}
    for name in ALL_FEATURES:
        left = calibration["features"][name]
        right = test["features"][name]
        ks = ks_2samp(left, right)
        result[name] = {
            "calibration_mean": float(np.mean(left)),
            "calibration_std": float(np.std(left)),
            "test_mean": float(np.mean(right)),
            "test_std": float(np.std(right)),
            "standardized_mean_difference": standardized_mean_difference(
                left, right
            ),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "wasserstein_distance": float(
                wasserstein_distance(left, right)
            ),
        }
    return result


def target_correlations(split):
    result = {}
    for target_name, target in split["targets"].items():
        result[target_name] = {}
        for feature_name in ALL_FEATURES:
            feature = split["features"][feature_name]
            result[target_name][feature_name] = {
                "pearson": float(pearsonr(feature, target).statistic),
                "spearman": float(spearmanr(feature, target).statistic),
            }
    return result


def error_sigma_components(calibration, test, residual_std):
    result = {}
    for channel_index, channel in enumerate(CHANNELS):
        result[channel] = {}
        for scope, selector in (
            ("all_steps", (slice(None), slice(None), channel_index)),
            ("step1", (slice(None), 0, channel_index)),
        ):
            calibration_error = calibration["error"][selector]
            test_error = test["error"][selector]
            calibration_sigma = calibration["sigma"][selector]
            test_sigma = test["sigma"][selector]
            calibration_z = calibration["z"][selector]
            test_z = test["z"][selector]
            result[channel][scope] = {
                "calibration": {
                    "error_normalized_mean": float(
                        np.mean(calibration_error)
                    ),
                    "error_normalized_rms": float(
                        np.sqrt(np.mean(np.square(calibration_error)))
                    ),
                    "error_physical_mean": float(
                        np.mean(calibration_error) * residual_std[channel_index]
                    ),
                    "sigma_normalized_mean": float(
                        np.mean(calibration_sigma)
                    ),
                    "z_mean": float(np.mean(calibration_z)),
                    "z_rms": float(
                        np.sqrt(np.mean(np.square(calibration_z)))
                    ),
                },
                "test": {
                    "error_normalized_mean": float(np.mean(test_error)),
                    "error_normalized_rms": float(
                        np.sqrt(np.mean(np.square(test_error)))
                    ),
                    "error_physical_mean": float(
                        np.mean(test_error) * residual_std[channel_index]
                    ),
                    "sigma_normalized_mean": float(np.mean(test_sigma)),
                    "z_mean": float(np.mean(test_z)),
                    "z_rms": float(
                        np.sqrt(np.mean(np.square(test_z)))
                    ),
                },
                "test_minus_calibration": {
                    "error_normalized_mean": float(
                        np.mean(test_error) - np.mean(calibration_error)
                    ),
                    "error_physical_mean": float(
                        (
                            np.mean(test_error)
                            - np.mean(calibration_error)
                        )
                        * residual_std[channel_index]
                    ),
                    "sigma_normalized_mean": float(
                        np.mean(test_sigma) - np.mean(calibration_sigma)
                    ),
                    "z_mean": float(
                        np.mean(test_z) - np.mean(calibration_z)
                    ),
                },
            }
    return result


def file_cluster_inference(calibration, test, seed, bootstrap_samples=5000):
    rng = np.random.default_rng(seed)
    result = {}
    windows_per_file = calibration["windows_per_file"]
    if windows_per_file != test["windows_per_file"]:
        raise ValueError("Split window counts per file must match")
    for target_name in calibration["targets"]:
        calibration_file = calibration["targets"][target_name].reshape(
            -1, windows_per_file
        ).mean(axis=1)
        test_file = test["targets"][target_name].reshape(
            -1, windows_per_file
        ).mean(axis=1)
        calibration_count = len(calibration_file)
        test_count = len(test_file)
        bootstrap_gap = np.empty(bootstrap_samples, dtype=np.float64)
        for index in range(bootstrap_samples):
            calibration_sample = calibration_file[
                rng.integers(0, calibration_count, calibration_count)
            ]
            test_sample = test_file[
                rng.integers(0, test_count, test_count)
            ]
            bootstrap_gap[index] = (
                np.mean(test_sample) - np.mean(calibration_sample)
            )
        gap = float(np.mean(test_file) - np.mean(calibration_file))
        pooled_std = float(
            np.sqrt(0.5 * (np.var(calibration_file) + np.var(test_file)))
        )
        confidence_interval = np.quantile(
            bootstrap_gap, (0.025, 0.975)
        )
        welch = ttest_ind(test_file, calibration_file, equal_var=False)
        result[target_name] = {
            "calibration_files": calibration_count,
            "test_files": test_count,
            "gap": gap,
            "cluster_bootstrap_95ci": confidence_interval.tolist(),
            "file_level_standardized_effect": (
                gap / pooled_std if pooled_std > 0 else None
            ),
            "welch_t_statistic": float(welch.statistic),
            "welch_pvalue": float(welch.pvalue),
            "bootstrap_samples": bootstrap_samples,
            "confidence_interval_excludes_zero": bool(
                confidence_interval[0] > 0
                or confidence_interval[1] < 0
            ),
        }
    return result


def make_groupings(calibration, test):
    groupings = {}
    for name in VEHICLE_FEATURES:
        edges = quantile_edges(calibration["features"][name], bins=5)
        groupings[name] = {
            "kind": "quantile",
            "edges": edges.tolist(),
            "calibration": assign_bins(
                calibration["features"][name], edges
            ),
            "test": assign_bins(test["features"][name], edges),
        }

    for first, second, name in (
        (
            "initial_steer",
            "initial_yawrate",
            "initial_steer_x_initial_yawrate",
        ),
        (
            "future_mean_steer",
            "initial_vx",
            "future_mean_steer_x_initial_vx",
        ),
    ):
        first_edges = quantile_edges(
            calibration["features"][first], bins=4
        )
        second_edges = quantile_edges(
            calibration["features"][second], bins=4
        )
        first_calibration = assign_bins(
            calibration["features"][first], first_edges
        )
        first_test = assign_bins(test["features"][first], first_edges)
        second_calibration = assign_bins(
            calibration["features"][second], second_edges
        )
        second_test = assign_bins(test["features"][second], second_edges)
        second_bin_count = len(second_edges) - 1
        groupings[name] = {
            "kind": "two_dimensional_quantile",
            "first_feature": first,
            "second_feature": second,
            "first_edges": first_edges.tolist(),
            "second_edges": second_edges.tolist(),
            "calibration": (
                first_calibration * second_bin_count + second_calibration
            ),
            "test": first_test * second_bin_count + second_test,
        }

    groupings["chunk_index"] = {
        "kind": "categorical",
        "calibration": calibration["features"]["chunk_index"].astype(int),
        "test": test["features"]["chunk_index"].astype(int),
    }
    groupings["file_time_15min"] = {
        "kind": "fixed_time",
        "calibration": (
            calibration["features"]["file_timestamp_seconds"] // 900
        ).astype(int),
        "test": (
            test["features"]["file_timestamp_seconds"] // 900
        ).astype(int),
    }
    log_edges = quantile_edges(
        calibration["features"]["file_log_id"], bins=5
    )
    groupings["file_log_id"] = {
        "kind": "quantile",
        "edges": log_edges.tolist(),
        "calibration": assign_bins(
            calibration["features"]["file_log_id"], log_edges
        ),
        "test": assign_bins(test["features"]["file_log_id"], log_edges),
    }
    return groupings


def decomposition_results(calibration, test, groupings):
    result = {}
    for grouping_name, grouping in groupings.items():
        result[grouping_name] = {
            "kind": grouping["kind"],
            "targets": {},
        }
        for key in (
            "edges",
            "first_feature",
            "second_feature",
            "first_edges",
            "second_edges",
        ):
            if key in grouping:
                result[grouping_name][key] = grouping[key]
        for target_name in calibration["targets"]:
            result[grouping_name]["targets"][target_name] = gap_decomposition(
                calibration["targets"][target_name],
                test["targets"][target_name],
                grouping["calibration"],
                grouping["test"],
            )
    return result


def classifier_matrix(split, names):
    return np.column_stack([split["features"][name] for name in names])


def split_classifiers(calibration, test, seed):
    outputs = {}
    labels = np.concatenate(
        (
            np.zeros(len(calibration["z"]), dtype=np.int64),
            np.ones(len(test["z"]), dtype=np.int64),
        )
    )
    groups = np.concatenate(
        (
            calibration["features"]["file_index"],
            test["features"]["file_index"] + len(
                np.unique(calibration["features"]["file_index"])
            ),
        )
    )
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed
    )
    for feature_set_name, feature_names in (
        ("vehicle_only", VEHICLE_FEATURES),
        ("vehicle_plus_file", ALL_FEATURES),
    ):
        matrix = np.concatenate(
            (
                classifier_matrix(calibration, feature_names),
                classifier_matrix(test, feature_names),
            )
        )
        estimators = {
            "logistic": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0, max_iter=2000, random_state=seed
                ),
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            ),
        }
        outputs[feature_set_name] = {}
        for model_name, estimator in estimators.items():
            probabilities = cross_val_predict(
                estimator,
                matrix,
                labels,
                groups=groups,
                cv=splitter,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            outputs[feature_set_name][model_name] = {
                "auc": float(roc_auc_score(labels, probabilities)),
                "probabilities": probabilities,
            }
    return outputs


def propensity_diagnostics(calibration, test, classifier_outputs):
    result = {}
    calibration_count = len(calibration["z"])
    for feature_set_name, models in classifier_outputs.items():
        result[feature_set_name] = {}
        for model_name, model_result in models.items():
            probabilities = np.clip(
                np.asarray(model_result["probabilities"][:calibration_count]),
                0.05,
                0.95,
            )
            weights = np.clip(
                probabilities / (1.0 - probabilities), 0.1, 10.0
            )
            weights /= np.mean(weights)
            effective_sample_size = float(
                np.square(np.sum(weights)) / np.sum(np.square(weights))
            )
            targets = {}
            for target_name in calibration["targets"]:
                calibration_values = calibration["targets"][target_name]
                test_values = test["targets"][target_name]
                calibration_mean = float(np.mean(calibration_values))
                test_mean = float(np.mean(test_values))
                weighted_mean = float(
                    np.average(calibration_values, weights=weights)
                )
                original_gap = test_mean - calibration_mean
                remaining_gap = test_mean - weighted_mean
                targets[target_name] = {
                    "calibration_mean": calibration_mean,
                    "weighted_calibration_mean": weighted_mean,
                    "test_mean": test_mean,
                    "original_gap": original_gap,
                    "remaining_gap": remaining_gap,
                    "absolute_gap_reduction_fraction": (
                        float(
                            1.0
                            - abs(remaining_gap) / abs(original_gap)
                        )
                        if abs(original_gap) > 1e-12
                        else None
                    ),
                }
            result[feature_set_name][model_name] = {
                "auc": model_result["auc"],
                "weight_minimum": float(np.min(weights)),
                "weight_maximum": float(np.max(weights)),
                "effective_sample_size": effective_sample_size,
                "targets": targets,
            }
    return result


def summarize_decompositions(decompositions):
    summary = {}
    for target_name in KEY_TARGETS:
        candidates = []
        for grouping_name, grouping in decompositions.items():
            result = grouping["targets"][target_name]
            if (
                result["composition_reduction_fraction"] is not None
                and result["calibration_common_support_mass"] >= 0.95
                and result["test_common_support_mass"] >= 0.95
            ):
                candidates.append(
                    (
                        result["composition_reduction_fraction"],
                        grouping_name,
                        result,
                    )
                )
        candidates.sort(reverse=True, key=lambda item: item[0])
        summary[target_name] = [
            {
                "grouping": grouping_name,
                "composition_reduction_fraction": reduction,
                "composition": result["composition"],
                "conditional": result["conditional"],
                "common_support_gap": result["common_support_gap"],
            }
            for reduction, grouping_name, result in candidates[:5]
        ]
    return summary


def plot_feature_shift(feature_results, classifier_results, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    names = list(VEHICLE_FEATURES)
    smd = [
        feature_results[name]["standardized_mean_difference"]
        for name in names
    ]
    axes[0].barh(names, smd)
    axes[0].axvline(0.1, color="black", linestyle="--")
    axes[0].axvline(-0.1, color="black", linestyle="--")
    axes[0].set_title("Calibration→test standardized mean difference")
    axes[0].grid(axis="x", alpha=0.25)

    labels = []
    values = []
    for feature_set, models in classifier_results.items():
        for model_name, result in models.items():
            labels.append(f"{feature_set}\n{model_name}")
            values.append(result["auc"])
    axes[1].bar(labels, values)
    axes[1].axhline(0.5, color="black", linestyle="--")
    axes[1].axhline(0.6, color="gray", linestyle=":")
    axes[1].set_ylim(0.45, max(0.65, max(values) + 0.03))
    axes[1].set_ylabel("file-grouped OOF AUC")
    axes[1].set_title("Can observable conditions identify the split?")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_key_binned_means(decompositions, output_path):
    group_names = (
        "initial_vx",
        "initial_yawrate",
        "initial_steer",
        "future_mean_steer",
    )
    targets = ("mean_dy_body", "mean_dyawrate")
    figure, axes = plt.subplots(2, 4, figsize=(18, 8))
    for row, target_name in enumerate(targets):
        for column, group_name in enumerate(group_names):
            rows = decompositions[group_name]["targets"][target_name]["bins"]
            common = [
                item
                for item in rows
                if item["calibration_mean"] is not None
                and item["test_mean"] is not None
            ]
            x = np.arange(len(common))
            axes[row, column].plot(
                x,
                [item["calibration_mean"] for item in common],
                marker="o",
                label="calibration",
            )
            axes[row, column].plot(
                x,
                [item["test_mean"] for item in common],
                marker="o",
                label="test",
            )
            axes[row, column].axhline(0.0, color="black", linestyle="--")
            axes[row, column].set_title(f"{target_name}\nby {group_name}")
            axes[row, column].set_xlabel("calibration-quantile bin")
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_gap_reweighting(propensity, output_path):
    methods = (
        ("vehicle_only", "logistic"),
        ("vehicle_only", "hist_gradient_boosting"),
        ("vehicle_plus_file", "hist_gradient_boosting"),
    )
    targets = list(KEY_TARGETS)
    x = np.arange(len(targets))
    width = 0.24
    figure, axis = plt.subplots(figsize=(13, 6))
    for index, (feature_set, model) in enumerate(methods):
        values = [
            propensity[feature_set][model]["targets"][target][
                "absolute_gap_reduction_fraction"
            ]
            for target in targets
        ]
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=f"{feature_set}/{model}",
        )
    axis.axhline(0.5, color="black", linestyle="--", label="50% gate")
    axis.axhline(0.0, color="gray")
    axis.set_xticks(x, targets)
    axis.set_ylabel("absolute gap reduction after OOF reweighting")
    axis.set_title("Does observable covariate reweighting explain z-mean shift?")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
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
    calibration = collect_split(
        wrapper, calibration_loader, calibration_files, stats, device
    )
    test = collect_split(wrapper, test_loader, test_files, stats, device)

    feature_results = feature_shift(calibration, test)
    residual_std = stats["residual"][1].cpu().numpy()
    component_results = error_sigma_components(
        calibration, test, residual_std
    )
    cluster_inference = file_cluster_inference(
        calibration, test, args.seed
    )
    correlations = {
        "calibration": target_correlations(calibration),
        "test": target_correlations(test),
    }
    groupings = make_groupings(calibration, test)
    decompositions = decomposition_results(
        calibration, test, groupings
    )
    decomposition_summary = summarize_decompositions(decompositions)
    classifier_raw = split_classifiers(calibration, test, args.seed)
    propensity = propensity_diagnostics(
        calibration, test, classifier_raw
    )
    classifier_results = {
        feature_set: {
            model: {"auc": result["auc"]}
            for model, result in models.items()
        }
        for feature_set, models in classifier_raw.items()
    }

    max_vehicle_smd = max(
        abs(feature_results[name]["standardized_mean_difference"])
        for name in VEHICLE_FEATURES
    )
    vehicle_auc = classifier_results["vehicle_only"][
        "hist_gradient_boosting"
    ]["auc"]
    key_reweighting = propensity["vehicle_only"][
        "hist_gradient_boosting"
    ]["targets"]
    key_gap_reductions = {
        target: key_reweighting[target][
            "absolute_gap_reduction_fraction"
        ]
        for target in KEY_TARGETS
    }
    gates = {
        "vehicle_nonlinear_auc_below_0.60": vehicle_auc < 0.60,
        "all_vehicle_abs_smd_below_0.10": max_vehicle_smd < 0.10,
        "mean_dy_body_reweighting_explains_at_least_half": (
            key_gap_reductions["mean_dy_body"] is not None
            and key_gap_reductions["mean_dy_body"] >= 0.50
        ),
        "mean_dyawrate_reweighting_explains_at_least_half": (
            key_gap_reductions["mean_dyawrate"] is not None
            and key_gap_reductions["mean_dyawrate"] >= 0.50
        ),
    }
    gates["observable_covariate_mix_is_minor"] = (
        gates["vehicle_nonlinear_auc_below_0.60"]
        and gates["all_vehicle_abs_smd_below_0.10"]
        and not gates[
            "mean_dy_body_reweighting_explains_at_least_half"
        ]
        and not gates[
            "mean_dyawrate_reweighting_explains_at_least_half"
        ]
    )

    plot_feature_shift(
        feature_results,
        classifier_results,
        output_dir / "feature_shift_and_classifier.png",
    )
    plot_key_binned_means(
        decompositions, output_dir / "conditional_bin_means.png"
    )
    plot_gap_reweighting(
        propensity, output_dir / "propensity_gap_reduction.png"
    )
    np.savez_compressed(
        output_dir / "regime_data.npz",
        calibration_z=calibration["z"],
        test_z=test["z"],
        calibration_error=calibration["error"],
        test_error=test["error"],
        calibration_sigma=calibration["sigma"],
        test_sigma=test["sigma"],
        **{
            f"calibration_feature_{name}": calibration["features"][name]
            for name in ALL_FEATURES
        },
        **{
            f"test_feature_{name}": test["features"][name]
            for name in ALL_FEATURES
        },
    )
    summary = {
        "protocol": "nuplan_probability_regime_shift_v1",
        "source_run": str(run_dir.resolve()),
        "source_summary_sha256": sha256(run_dir / "summary.json"),
        "args": vars(args),
        "files": {
            "calibration": len(calibration_files),
            "test": len(test_files),
            "intersection": len(
                set(calibration_files) & set(test_files)
            ),
            "windows_per_file": calibration["windows_per_file"],
        },
        "feature_shift": feature_results,
        "error_sigma_components": component_results,
        "file_cluster_inference": cluster_inference,
        "target_correlations": correlations,
        "classifier_results": classifier_results,
        "propensity_diagnostics": propensity,
        "decompositions": decompositions,
        "decomposition_summary": decomposition_summary,
        "gate_inputs": {
            "maximum_absolute_vehicle_smd": max_vehicle_smd,
            "vehicle_nonlinear_auc": vehicle_auc,
            "key_vehicle_reweighting_gap_reductions": key_gap_reductions,
        },
        "gates": gates,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files": summary["files"],
                "gate_inputs": summary["gate_inputs"],
                "gates": gates,
                "classifier_results": classifier_results,
                "largest_feature_shifts": sorted(
                    (
                        (
                            name,
                            result["standardized_mean_difference"],
                            result["ks_statistic"],
                        )
                        for name, result in feature_results.items()
                    ),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:8],
                "key_propensity": {
                    feature_set: {
                        model: {
                            "auc": result["auc"],
                            "effective_sample_size": result[
                                "effective_sample_size"
                            ],
                            "targets": {
                                target: result["targets"][target]
                                for target in KEY_TARGETS
                            },
                        }
                        for model, result in models.items()
                    }
                    for feature_set, models in propensity.items()
                },
                "error_sigma_components": {
                    channel: component_results[channel]
                    for channel in CHANNELS
                },
                "key_file_cluster_inference": {
                    target: cluster_inference[target]
                    for target in KEY_TARGETS
                },
                "decomposition_summary": decomposition_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
