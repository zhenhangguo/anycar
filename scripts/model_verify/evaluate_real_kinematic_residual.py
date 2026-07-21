#!/usr/bin/env python3
"""Evaluate two frozen nuPlan checkpoints on real-car data.

The evaluation is intentionally zero-shot: checkpoint normalization statistics,
network weights, and kinematic parameters remain frozen.  The real-car pickle
files provide realized future ``acc_fused`` and steering-wheel angle signals,
so the reported result is an offline realized-input prediction benchmark rather
than a causal deployment benchmark.
"""

import argparse
import json
import math
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from car_foundation.dataset import MujocoDataset
from car_foundation.kinematic_residual import (
    CONSISTENT_TRANSITION_NAMES,
    KinematicBicycleParams,
    NuPlanKinematicResidualDataset,
    rollout_consistent_transition_sequence,
    rollout_kinematic_residual_consistent,
)
from train_kinematic_residual_ablation import (
    HORIZON_STEPS,
    LOSS_WEIGHTS,
    add_metrics,
    empty_metric_sums,
    forward_independent_history,
    make_model,
    prepare_batch,
    prepare_nominal_query,
    summarize_metrics,
)


class IndexedPredictionDataset(Dataset):
    """Index a prediction view while retaining its horizon metadata."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.prediction_length = dataset.prediction_length

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot real-car evaluation of two frozen model checkpoints."
    )
    parser.add_argument(
        "--dataset-path",
        default="/disk1/collect_data_from_anycar/data_from_bag/new_temp_data/pkg_file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join(
            REPO_ROOT,
            "outputs",
            "kinematic_residual_ablation",
            "20260714T171741",
        ),
        help="Fallback directory containing <variant>_best.pt checkpoints.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        default="",
        help="Explicit reference checkpoint path; overrides --checkpoint-dir.",
    )
    parser.add_argument(
        "--reference-variant",
        choices=("direct", "residual", "query"),
        default="direct",
    )
    parser.add_argument(
        "--direct-checkpoint",
        default="",
        help="Deprecated alias for --reference-checkpoint when --reference-variant=direct.",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        default="",
        help="Explicit residual/query checkpoint path; overrides --checkpoint-dir.",
    )
    parser.add_argument(
        "--candidate-variant",
        choices=("residual", "query"),
        default="residual",
    )
    parser.add_argument("--max-files", type=int, default=0, help="0 uses every pkl file.")
    parser.add_argument("--max-eval-episodes", type=int, default=0, help="0 uses all valid windows.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--steer-shifts",
        default="1",
        help="Comma-separated shifts. 1 matches the nuPlan checkpoint protocol.",
    )
    parser.add_argument("--quaternion-norm-tolerance", type=float, default=0.01)
    parser.add_argument("--position-jump-threshold", type=float, default=2.0)
    parser.add_argument("--yaw-consistency-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "real_kinematic_residual_evaluation"),
    )
    return parser.parse_args()


def discover_files(path, max_files):
    files = sorted(Path(path).glob("*.pkl"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No pkl files under {path}")
    return files


def raw_window_validity(
    files,
    sequence_length,
    quaternion_norm_tolerance,
    position_jump_threshold,
):
    """Build a validity mask matching MujocoDataset's non-overlapping windows."""
    masks = []
    audit = {
        "total_files": len(files),
        "total_windows": 0,
        "invalid_nonfinite_windows": 0,
        "invalid_quaternion_windows": 0,
        "invalid_position_jump_windows": 0,
    }
    required = (
        "xpos_x",
        "xpos_y",
        "xori_w",
        "xori_x",
        "xori_y",
        "xori_z",
        "xvel_x",
        "avel_z",
        "throttle",
        "steer",
    )
    for path in tqdm(files, desc="audit raw windows"):
        with open(path, "rb") as stream:
            logs = pickle.load(stream).data_logs
        values = {name: np.asarray(logs[name]) for name in required}
        length = len(values["xpos_x"])
        usable = length - length % sequence_length
        if usable == 0 or any(len(value) != length for value in values.values()):
            raise ValueError(f"Invalid channel length in {path}")
        windows = {
            name: value[:usable].reshape(-1, sequence_length)
            for name, value in values.items()
        }
        finite = np.ones(windows["xpos_x"].shape[0], dtype=bool)
        for value in windows.values():
            finite &= np.isfinite(value).all(axis=1)

        qnorm = np.sqrt(
            windows["xori_w"] ** 2
            + windows["xori_x"] ** 2
            + windows["xori_y"] ** 2
            + windows["xori_z"] ** 2
        )
        quaternion_ok = (np.abs(qnorm - 1.0) <= quaternion_norm_tolerance).all(axis=1)
        position_step = np.hypot(
            np.diff(windows["xpos_x"], axis=1),
            np.diff(windows["xpos_y"], axis=1),
        )
        position_ok = (position_step <= position_jump_threshold).all(axis=1)
        valid = finite & quaternion_ok & position_ok
        masks.append(valid)
        audit["total_windows"] += valid.size
        audit["invalid_nonfinite_windows"] += int(np.sum(~finite))
        audit["invalid_quaternion_windows"] += int(np.sum(~quaternion_ok))
        audit["invalid_position_jump_windows"] += int(np.sum(~position_ok))

    valid = np.concatenate(masks)
    audit["valid_raw_windows"] = int(np.sum(valid))
    audit["invalid_raw_windows"] = int(np.sum(~valid))
    return valid, audit


def add_yaw_consistency_filter(base_dataset, valid, dt, threshold):
    data = base_dataset.data
    yaw_delta = torch.atan2(
        torch.sin(data[:, 1:, 2] - data[:, :-1, 2]),
        torch.cos(data[:, 1:, 2] - data[:, :-1, 2]),
    )
    expected = data[:, :-1, 5] * dt
    consistency_ok = torch.max(torch.abs(yaw_delta - expected), dim=1).values <= threshold
    consistency_ok = consistency_ok.cpu().numpy()
    combined = valid & consistency_ok
    return combined, int(np.sum(valid & ~consistency_ok))


def checkpoint_stats(checkpoint, device):
    return {
        key: (value[0].to(device), value[1].to(device))
        for key, value in checkpoint["stats"].items()
    }


def checkpoint_params(checkpoint, dt):
    values = checkpoint["params"]
    return KinematicBicycleParams(
        dt=dt,
        wheelbase=float(values["wheelbase"]),
        steering_ratio=float(values["steering_ratio"]),
        steering_offset=float(values["steering_offset"]),
    )


def evaluate_model(variant, model, loader, stats, params, device, max_episodes):
    sums, count = empty_metric_sums()
    horizon_sums = {
        horizon: empty_metric_sums()
        for horizon in HORIZON_STEPS
        if horizon <= loader.dataset.prediction_length
    }
    target_squared_error = 0.0
    target_element_count = 0
    common_transition_squared_error = 0.0
    common_transition_element_count = 0
    transition_squared_error = torch.zeros(4, dtype=torch.float64)
    transition_count = 0
    seen = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {variant}"):
            if max_episodes > 0:
                remaining = max_episodes - seen
                if remaining <= 0:
                    break
                if batch["initial_state"].shape[0] > remaining:
                    batch = {key: value[:remaining] for key, value in batch.items()}

            initial = batch["initial_state"].to(device)
            truth = batch["truth"].to(device)
            history, action, context, mask = prepare_batch(
                batch,
                stats["history"][0],
                stats["history"][1],
                stats["context"][0],
                stats["context"][1],
                device,
            )
            nominal_state = None
            nominal_transition = None
            if variant == "query":
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
            target_key = "direct_target" if variant == "direct" else "residual_target"
            target_stats_key = "direct" if variant == "direct" else "residual"
            target_mean, target_std = stats[target_stats_key]
            target_normalized = (
                batch[target_key].to(device) - target_mean
            ) / target_std
            weighted_error = (
                (prediction_normalized - target_normalized) ** 2
                * LOSS_WEIGHTS.to(device)[None, None, :]
            )
            target_squared_error += weighted_error.sum().item()
            target_element_count += weighted_error.numel()

            prediction_value = prediction_normalized * target_std + target_mean
            if variant == "direct":
                composed_transition = prediction_value
                prediction = rollout_consistent_transition_sequence(
                    initial, prediction_value, params
                )
            else:
                composed_transition = (
                    batch["base_transition"].to(device) + prediction_value
                )
                prediction = rollout_kinematic_residual_consistent(
                    initial, action, prediction_value, params
                )

            transition_error = composed_transition - batch["direct_target"].to(device)
            transition_squared_error += (
                transition_error.double().square().sum(dim=(0, 1)).cpu()
            )
            transition_count += transition_error.shape[0] * transition_error.shape[1]
            common_normalized_error = transition_error / stats["direct"][1]
            common_weighted_error = (
                common_normalized_error.square()
                * LOSS_WEIGHTS.to(device)[None, None, :]
            )
            common_transition_squared_error += common_weighted_error.sum().item()
            common_transition_element_count += common_weighted_error.numel()

            count = add_metrics(sums, count, prediction, truth)
            for horizon, (horizon_values, horizon_count) in horizon_sums.items():
                horizon_count = add_metrics(
                    horizon_values,
                    horizon_count,
                    prediction[:, horizon - 1 : horizon],
                    truth[:, horizon - 1 : horizon],
                )
                horizon_sums[horizon] = (horizon_values, horizon_count)
            seen += prediction.shape[0]

    result = {
        "all_steps": summarize_metrics(sums, count),
        "horizons": {
            str(horizon): summarize_metrics(values, horizon_count)
            for horizon, (values, horizon_count) in horizon_sums.items()
        },
        "transition_rmse": {
            name: math.sqrt(float(value) / max(transition_count, 1))
            for name, value in zip(
                CONSISTENT_TRANSITION_NAMES, transition_squared_error
            )
        },
        "common_transition_loss": (
            common_transition_squared_error
            / max(common_transition_element_count, 1)
        ),
        "branch_normalized_target_loss_not_cross_comparable": (
            target_squared_error / max(target_element_count, 1)
        ),
        "evaluated_episodes": seen,
    }
    return result


def relative_improvement(reference, candidate):
    result = {}
    for key, reference_value in reference.items():
        candidate_value = candidate[key]
        result[key] = (
            100.0 * (reference_value - candidate_value) / reference_value
            if reference_value != 0.0
            else math.nan
        )
    return result


def scalar_relative_improvement(reference, candidate):
    return (
        100.0 * (reference - candidate) / reference
        if reference != 0.0
        else math.nan
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current model implementation.")
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    shifts = [int(value.strip()) for value in args.steer_shifts.split(",") if value.strip()]
    if not shifts:
        raise ValueError("At least one steering shift is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    reference_variant = args.reference_variant
    candidate_variant = args.candidate_variant
    if reference_variant == candidate_variant:
        raise ValueError("Reference and candidate variants must differ")
    if args.direct_checkpoint and reference_variant != "direct":
        raise ValueError(
            "--direct-checkpoint is only valid with --reference-variant=direct"
        )
    if args.reference_checkpoint and args.direct_checkpoint:
        raise ValueError(
            "Use only one of --reference-checkpoint and --direct-checkpoint"
        )
    checkpoint_dir = Path(args.checkpoint_dir)
    reference_checkpoint = (
        args.reference_checkpoint
        or args.direct_checkpoint
        or str(checkpoint_dir / f"{reference_variant}_best.pt")
    )
    checkpoint_paths = {
        reference_variant: Path(reference_checkpoint),
        candidate_variant: Path(args.candidate_checkpoint)
        if args.candidate_checkpoint
        else checkpoint_dir / f"{candidate_variant}_best.pt",
    }
    checkpoints = {
        variant: torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        for variant, checkpoint_path in checkpoint_paths.items()
    }
    for variant, checkpoint in checkpoints.items():
        if checkpoint.get("variant") != variant:
            raise ValueError(f"Checkpoint variant mismatch for {variant}")
        checkpoint_dt = float(checkpoint["params"]["dt"])
        if not math.isclose(checkpoint_dt, args.dt, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Requested dt={args.dt} differs from {variant} checkpoint dt={checkpoint_dt}"
            )

    history_length = int(checkpoints[reference_variant]["args"]["history_length"])
    prediction_length = int(
        checkpoints[reference_variant]["args"]["prediction_length"]
    )
    for name in ("history_length", "prediction_length"):
        reference_value = int(checkpoints[reference_variant]["args"][name])
        candidate_value = int(checkpoints[candidate_variant]["args"][name])
        if reference_value != candidate_value:
            raise ValueError(
                f"Checkpoint {name} mismatch: {reference_variant}={reference_value}, "
                f"{candidate_variant}={candidate_value}"
            )
    sequence_length = history_length + 1 + prediction_length
    files = discover_files(args.dataset_path, args.max_files)
    raw_valid, audit = raw_window_validity(
        files,
        sequence_length,
        args.quaternion_norm_tolerance,
        args.position_jump_threshold,
    )
    base = MujocoDataset(
        [str(path) for path in files],
        history_length + 1,
        prediction_length,
        delays=None,
        teacher_forcing=False,
        binary_mask=False,
        attack=False,
        use_zero_point=True,
    )
    if len(raw_valid) != len(base):
        raise RuntimeError(
            f"Raw audit produced {len(raw_valid)} windows but loader produced {len(base)}"
        )
    valid, invalid_yaw = add_yaw_consistency_filter(
        base, raw_valid, args.dt, args.yaw_consistency_threshold
    )
    valid_indices = np.flatnonzero(valid)
    audit["invalid_yaw_consistency_windows"] = invalid_yaw
    audit["valid_windows"] = int(valid_indices.size)
    audit["invalid_windows"] = int(len(valid) - valid_indices.size)

    results = {}
    for shift in shifts:
        shift_results = {}
        for variant in (reference_variant, candidate_variant):
            checkpoint = checkpoints[variant]
            params = checkpoint_params(checkpoint, args.dt)
            view = NuPlanKinematicResidualDataset(
                base,
                history_length,
                prediction_length,
                params,
                steer_shift=shift,
            )
            indexed_view = IndexedPredictionDataset(view, valid_indices)
            loader = DataLoader(
                indexed_view,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
            )
            model_args = SimpleNamespace(**checkpoint["args"])
            model_args.device = args.device
            model = make_model(model_args, device, variant=variant)
            model.load_state_dict(checkpoint["model_state_dict"])
            stats = checkpoint_stats(checkpoint, device)
            shift_results[variant] = evaluate_model(
                variant,
                model,
                loader,
                stats,
                params,
                device,
                args.max_eval_episodes,
            )
            shift_results[variant]["checkpoint"] = str(checkpoint_paths[variant])
            del model
            torch.cuda.empty_cache()
        shift_results[f"{candidate_variant}_improvement_percent"] = {
            "all_steps": relative_improvement(
                shift_results[reference_variant]["all_steps"],
                shift_results[candidate_variant]["all_steps"],
            ),
            "horizons": {
                horizon: relative_improvement(
                    shift_results[reference_variant]["horizons"][horizon],
                    shift_results[candidate_variant]["horizons"][horizon],
                )
                for horizon in shift_results[reference_variant]["horizons"]
            },
            "transition_rmse": relative_improvement(
                shift_results[reference_variant]["transition_rmse"],
                shift_results[candidate_variant]["transition_rmse"],
            ),
            "common_transition_loss": scalar_relative_improvement(
                shift_results[reference_variant]["common_transition_loss"],
                shift_results[candidate_variant]["common_transition_loss"],
            ),
        }
        results[str(shift)] = shift_results

    summary = {
        "protocol": "realized_input_zero_shot_pair_v4",
        "dataset_path": args.dataset_path,
        "checkpoint_dir": str(checkpoint_dir),
        "reference_variant": reference_variant,
        "candidate_variant": candidate_variant,
        "checkpoints": {
            variant: str(path) for variant, path in checkpoint_paths.items()
        },
        "dt": args.dt,
        "history_length": history_length,
        "prediction_length": prediction_length,
        "steer_shifts": shifts,
        "input_semantics": {
            "longitudinal": "vehicle/status acc_fused (realized measurement)",
            "steering": "steering_report steering_wheel_angle (realized measurement)",
        },
        "files": len(files),
        "audit": audit,
        "results": results,
    }
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"summary: {output_path}")


if __name__ == "__main__":
    main()
