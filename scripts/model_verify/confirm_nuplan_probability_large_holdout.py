#!/usr/bin/env python3
"""Confirm nuPlan probability diagnostics on larger untouched slices."""

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch


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
from analyze_nuplan_probability_calibration import (
    all_horizon_metrics,
    drift_diagnostics,
    plot_coverage_comparison,
    plot_histograms_and_qq,
    plot_horizon_diagnostics,
    plot_temperature,
    split_diagnostics,
)
from analyze_nuplan_probability_regime_shift import (
    KEY_TARGETS,
    VEHICLE_FEATURES,
    checkpoint_stats,
    collect_split,
    error_sigma_components,
    feature_shift,
    file_cluster_inference,
    plot_feature_shift,
    read_paths,
    split_classifiers,
)
from train_kinematic_residual_ablation import make_model
from validate_nuplan_probabilistic_query import (
    CHANNELS,
    COVERAGE_LEVELS,
    make_loader,
)


CALIBRATION_START = 1024
TEST_START = 256
BOOTSTRAP_SAMPLES = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Confirm frozen nuPlan probability results on large holdouts."
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
    parser.add_argument(
        "--output-name", default="large_holdout_confirmation"
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_paths(path, paths):
    path.write_text("".join(f"{item}\n" for item in paths))


def audit_splits(source_run, run_dir):
    validation = read_paths(source_run / "val_files.txt")
    test = read_paths(source_run / "test_files.txt")
    if len(validation) != 3000 or len(test) != 1000:
        raise RuntimeError(
            f"Expected 3000 validation and 1000 test files, got "
            f"{len(validation)} and {len(test)}"
        )
    historical = {
        "sigma_train": read_paths(run_dir / "sigma_train_files.txt"),
        "sigma_val": read_paths(run_dir / "sigma_val_files.txt"),
        "calibration": read_paths(run_dir / "calibration_files.txt"),
        "probability_test": read_paths(
            run_dir / "probability_test_files.txt"
        ),
    }
    expected = {
        "sigma_train": validation[:512],
        "sigma_val": validation[512:768],
        "calibration": validation[768:1024],
        "probability_test": test[:256],
    }
    for name in expected:
        if historical[name] != expected[name]:
            raise RuntimeError(
                f"Historical {name} list does not match the frozen source slice"
            )

    expanded_calibration = validation[CALIBRATION_START:]
    confirm_test = test[TEST_START:]
    candidates = {
        "expanded_calibration": expanded_calibration,
        "confirm_test": confirm_test,
    }
    intersections = {}
    all_splits = {**historical, **candidates}
    names = list(all_splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            count = len(set(all_splits[left]) & set(all_splits[right]))
            intersections[f"{left}/{right}"] = count
            if (left in candidates or right in candidates) and count:
                raise RuntimeError(f"{left}/{right} overlap: {count} files")
    if len(expanded_calibration) != 1976 or len(confirm_test) != 744:
        raise RuntimeError("Frozen expanded slice counts changed")
    return expanded_calibration, confirm_test, intersections


def load_wrapper(source_summary, run_dir, device):
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
    return (
        wrapper,
        mean_checkpoint,
        model_args,
        mean_checkpoint_path,
        probability_checkpoint_path,
    )


def collect_files(
    files,
    wrapper,
    model_args,
    params,
    stats,
    device,
    batch_size,
    compare_mean,
):
    dataset, loader = make_loader(
        files, model_args, params, batch_size, 0, False
    )
    result = collect_split(
        wrapper,
        loader,
        files,
        stats,
        device,
        compare_mean=compare_mean,
    )
    del loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def pooled_coverage_errors(metrics):
    result = {}
    for label, (_, nominal) in COVERAGE_LEVELS.items():
        empirical = metrics["all_steps"]["coverage"][label][
            "empirical_by_channel"
        ]
        result[label] = {
            channel: abs(empirical[channel] - nominal)
            for channel in CHANNELS
        }
    return result


def worst_horizon_coverage_error(metrics):
    worst = {}
    for label, (_, nominal) in COVERAGE_LEVELS.items():
        candidates = []
        for horizon, horizon_result in metrics["horizons"].items():
            empirical = horizon_result["coverage"][label][
                "empirical_by_channel"
            ]
            for channel in CHANNELS:
                candidates.append(
                    (abs(empirical[channel] - nominal), int(horizon), channel)
                )
        value, horizon, channel = max(candidates)
        worst[label] = {
            "absolute_error": value,
            "horizon": horizon,
            "channel": channel,
        }
    return worst


def plot_gap_comparison(old_inference, new_inference, output_path):
    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(KEY_TARGETS))
    width = 0.36
    for offset, label, results, color in (
        (-width / 2, "256 vs 256", old_inference, "tab:gray"),
        (width / 2, "1976 vs 744", new_inference, "tab:blue"),
    ):
        gaps = [results[target]["gap"] for target in KEY_TARGETS]
        lower = [
            gap - results[target]["cluster_bootstrap_95ci"][0]
            for target, gap in zip(KEY_TARGETS, gaps)
        ]
        upper = [
            results[target]["cluster_bootstrap_95ci"][1] - gap
            for target, gap in zip(KEY_TARGETS, gaps)
        ]
        axis.bar(x + offset, gaps, width, label=label, color=color)
        axis.errorbar(
            x + offset,
            gaps,
            yerr=np.asarray([lower, upper]),
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    axis.axhline(0.0, color="black", linestyle="--")
    axis.set_xticks(x, KEY_TARGETS, rotation=15)
    axis.set_ylabel("test - calibration standardized-error gap")
    axis.set_title("Small probability split vs expanded confirmation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def strip_classifier_probabilities(classifiers):
    return {
        feature_set: {
            model: {"auc": result["auc"]}
            for model, result in models.items()
        }
        for feature_set, models in classifiers.items()
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    source_summary_path = run_dir / "summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    source_run = Path(source_summary["source_run"])
    device = torch.device(args.device)

    expanded_files, confirm_files, intersections = audit_splits(
        source_run, run_dir
    )
    output_dir = (
        run_dir / args.output_name / datetime.now().strftime("%Y%m%dT%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_paths(output_dir / "expanded_calibration_files.txt", expanded_files)
    write_paths(output_dir / "confirm_test_files.txt", confirm_files)

    (
        wrapper,
        mean_checkpoint,
        model_args,
        mean_checkpoint_path,
        probability_checkpoint_path,
    ) = load_wrapper(source_summary, run_dir, device)
    stats = checkpoint_stats(mean_checkpoint, device)
    params = KinematicBicycleParams(**mean_checkpoint["params"])

    calibration = collect_files(
        expanded_files,
        wrapper,
        model_args,
        params,
        stats,
        device,
        args.eval_batch_size,
        compare_mean=False,
    )
    confirm = collect_files(
        confirm_files,
        wrapper,
        model_args,
        params,
        stats,
        device,
        args.eval_batch_size,
        compare_mean=True,
    )
    if confirm["maximum_mean_difference"] != 0.0:
        raise RuntimeError(
            "Frozen probability wrapper changed the deterministic mean: "
            f"{confirm['maximum_mean_difference']}"
        )

    calibration_error = torch.from_numpy(calibration["error"])
    calibration_sigma = torch.from_numpy(calibration["sigma"])
    confirm_error = torch.from_numpy(confirm["error"])
    confirm_sigma = torch.from_numpy(confirm["sigma"])
    global_temperature = channel_temperature(
        calibration_error, calibration_sigma
    )
    horizon_temperature = horizon_channel_temperature(
        calibration_error, calibration_sigma
    )
    global_confirm_sigma = confirm_sigma * global_temperature.view(1, 1, -1)
    horizon_confirm_sigma = confirm_sigma * horizon_temperature.view(
        1, confirm_sigma.shape[1], confirm_sigma.shape[2]
    )
    residual_std_tensor = stats["residual"][1].cpu()
    residual_std = residual_std_tensor.numpy()
    metrics = {
        "raw": all_horizon_metrics(
            confirm_error, confirm_sigma, residual_std_tensor
        ),
        "global_temperature": all_horizon_metrics(
            confirm_error, global_confirm_sigma, residual_std_tensor
        ),
        "horizon_temperature": all_horizon_metrics(
            confirm_error, horizon_confirm_sigma, residual_std_tensor
        ),
    }

    calibration_diagnostics = split_diagnostics(calibration["z"])
    confirm_diagnostics = split_diagnostics(confirm["z"])
    drift = drift_diagnostics(calibration["z"], confirm["z"])
    feature_results = feature_shift(calibration, confirm)
    component_results = error_sigma_components(
        calibration, confirm, residual_std
    )
    cluster_results = file_cluster_inference(
        calibration,
        confirm,
        args.seed,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
    )
    classifier_results = strip_classifier_probabilities(
        split_classifiers(calibration, confirm, args.seed)
    )

    prior_regime_dirs = sorted((run_dir / "regime_shift").glob("*/summary.json"))
    if not prior_regime_dirs:
        raise RuntimeError("No prior Step 13 regime summary found")
    prior_regime_path = prior_regime_dirs[-1]
    prior_regime = json.loads(prior_regime_path.read_text())
    prior_cluster = prior_regime["file_cluster_inference"]

    plot_histograms_and_qq(
        calibration["z"],
        confirm["z"],
        output_dir / "standardized_error_hist_qq.png",
    )
    plot_horizon_diagnostics(
        calibration_diagnostics,
        confirm_diagnostics,
        drift,
        output_dir / "horizon_error_diagnostics.png",
    )
    plot_temperature(
        horizon_temperature.numpy(), output_dir / "horizon_temperature.png"
    )
    plot_coverage_comparison(
        metrics, output_dir / "coverage_by_horizon.png"
    )
    plot_feature_shift(
        feature_results,
        classifier_results,
        output_dir / "feature_shift_and_classifier.png",
    )
    plot_gap_comparison(
        prior_cluster,
        cluster_results,
        output_dir / "small_vs_large_location_gap.png",
    )

    pooled_errors = pooled_coverage_errors(metrics["horizon_temperature"])
    worst_horizon_errors = worst_horizon_coverage_error(
        metrics["horizon_temperature"]
    )
    persistence = {}
    for target in KEY_TARGETS:
        inference = cluster_results[target]
        persistent = (
            inference["confidence_interval_excludes_zero"]
            and abs(inference["file_level_standardized_effect"]) >= 0.2
        )
        persistence[target] = {
            "small_gap": prior_cluster[target]["gap"],
            "small_95ci": prior_cluster[target][
                "cluster_bootstrap_95ci"
            ],
            "large_gap": inference["gap"],
            "large_95ci": inference["cluster_bootstrap_95ci"],
            "large_file_level_standardized_effect": inference[
                "file_level_standardized_effect"
            ],
            "persistent": persistent,
        }

    max_vehicle_smd = max(
        abs(feature_results[name]["standardized_mean_difference"])
        for name in VEHICLE_FEATURES
    )
    all_pooled_coverage_within_5pp = all(
        error < 0.05
        for label_results in pooled_errors.values()
        for error in label_results.values()
    )
    raw_nll = metrics["raw"]["all_steps"]["mean"]["nll"]
    horizon_nll = metrics["horizon_temperature"]["all_steps"]["mean"]["nll"]
    gates = {
        "mean_invariance_exact": (
            confirm["maximum_mean_difference"] == 0.0
        ),
        "horizon_nll_no_worse_than_raw": horizon_nll <= raw_nll,
        "all_pooled_channel_coverage_within_5pp": (
            all_pooled_coverage_within_5pp
        ),
        "horizon_calibration_passed": (
            horizon_nll <= raw_nll and all_pooled_coverage_within_5pp
        ),
        "all_four_location_gaps_persistent": all(
            result["persistent"] for result in persistence.values()
        ),
    }

    np.savez_compressed(
        output_dir / "confirmation_data.npz",
        expanded_calibration_error=calibration["error"],
        expanded_calibration_sigma=calibration["sigma"],
        confirm_test_error=confirm["error"],
        confirm_test_sigma=confirm["sigma"],
    )
    summary = {
        "protocol": "nuplan_probability_large_holdout_confirmation_v1",
        "source_run": str(run_dir.resolve()),
        "source_summary_sha256": sha256(source_summary_path),
        "prior_regime_summary": str(prior_regime_path.resolve()),
        "prior_regime_summary_sha256": sha256(prior_regime_path),
        "mean_checkpoint": str(mean_checkpoint_path.resolve()),
        "mean_checkpoint_sha256": sha256(mean_checkpoint_path),
        "probability_checkpoint": str(
            probability_checkpoint_path.resolve()
        ),
        "probability_checkpoint_sha256": sha256(
            probability_checkpoint_path
        ),
        "args": vars(args),
        "files": {
            "expanded_calibration": len(expanded_files),
            "confirm_test": len(confirm_files),
            "expanded_calibration_episodes": len(calibration["z"]),
            "confirm_test_episodes": len(confirm["z"]),
            "windows_per_file": calibration["windows_per_file"],
            "intersections": intersections,
        },
        "maximum_mean_difference": confirm["maximum_mean_difference"],
        "global_temperature": global_temperature.tolist(),
        "horizon_temperature": {
            "minimum": float(torch.min(horizon_temperature)),
            "maximum": float(torch.max(horizon_temperature)),
            "mean": float(torch.mean(horizon_temperature)),
            "values": horizon_temperature.tolist(),
        },
        "metrics": metrics,
        "diagnostics": {
            "expanded_calibration": calibration_diagnostics,
            "confirm_test": confirm_diagnostics,
            "calibration_confirm_drift": drift,
        },
        "feature_shift": feature_results,
        "classifier_results": classifier_results,
        "error_sigma_components": component_results,
        "file_cluster_inference": cluster_results,
        "small_vs_large_persistence": persistence,
        "gate_inputs": {
            "raw_nll": raw_nll,
            "global_temperature_nll": metrics["global_temperature"][
                "all_steps"
            ]["mean"]["nll"],
            "horizon_temperature_nll": horizon_nll,
            "pooled_channel_coverage_absolute_errors": pooled_errors,
            "worst_horizon_coverage_absolute_errors": (
                worst_horizon_errors
            ),
            "maximum_absolute_vehicle_smd": max_vehicle_smd,
            "vehicle_nonlinear_auc": classifier_results["vehicle_only"][
                "hist_gradient_boosting"
            ]["auc"],
        },
        "gates": gates,
        "limitations": [
            "The expanded calibration slice was not used by the sigma head or "
            "prior probability diagnostics, but the deterministic mean "
            "checkpoint was selected on the full original validation split.",
            "The confirm-test slice was not used by deterministic model "
            "selection or any prior probability analysis.",
        ],
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
                "files": summary["files"],
                "maximum_mean_difference": summary[
                    "maximum_mean_difference"
                ],
                "gate_inputs": summary["gate_inputs"],
                "gates": gates,
                "small_vs_large_persistence": persistence,
                "key_components": {
                    channel: component_results[channel]
                    for channel in ("dy_body", "dyawrate")
                },
                "key_file_cluster_inference": {
                    target: cluster_results[target]
                    for target in KEY_TARGETS
                },
                "classifier_results": classifier_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
