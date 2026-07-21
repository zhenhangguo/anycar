#!/usr/bin/env python3
"""Fine-tune Residual-v2 and Query Residual on grouped real-car data.

The real files are split by acquisition-minute groups before any window is
constructed.  This prevents adjacent pkl fragments from being assigned to
different splits.  Both branches start from their frozen nuPlan checkpoints
and use the same real train/validation/test split and optimization protocol.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from car_foundation.dataset import MujocoDataset
from car_foundation.kinematic_residual import (
    KinematicBicycleParams,
    NuPlanKinematicResidualDataset,
)
from evaluate_real_kinematic_residual import (
    IndexedPredictionDataset,
    add_yaw_consistency_filter,
    checkpoint_params,
    checkpoint_stats,
    evaluate_model,
    raw_window_validity,
    relative_improvement,
    scalar_relative_improvement,
)
from train_kinematic_residual_ablation import (
    DEFAULT_ROLLOUT_SCALE_FLOORS,
    ROLLOUT_METRIC_NAMES,
    forward_independent_history,
    make_model,
    parse_rollout_horizons,
    parse_rollout_scale_floors,
    prepare_batch,
    prepare_nominal_query,
    rollout_scale_stats,
    run_epoch,
    set_seed,
    streaming_stats,
)


PROTOCOL_VERSION = "real_finetune_consistent_v2"
MINUTE_GROUP_PATTERN = re.compile(
    r"^bag_data_(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Residual-v2 and Query Residual on real-car data."
    )
    parser.add_argument(
        "--dataset-path",
        default="/disk1/collect_data_from_anycar/data_from_bag/new_temp_data/pkg_file",
    )
    parser.add_argument(
        "--residual-checkpoint",
        default=os.path.join(
            REPO_ROOT,
            "outputs/formal_residual_vs_query_20k/20260716T142015/residual_best.pt",
        ),
    )
    parser.add_argument(
        "--query-checkpoint",
        default=os.path.join(
            REPO_ROOT,
            "outputs/formal_residual_vs_query_20k/20260716T142015/query_best.pt",
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="0 uses all acquisition-minute groups; intended for smoke tests.",
    )
    parser.add_argument(
        "--max-episodes-per-split",
        type=int,
        default=0,
        help="0 uses every valid episode; intended for smoke tests.",
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steer-shift", type=int, default=1)
    parser.add_argument("--quaternion-norm-tolerance", type=float, default=0.01)
    parser.add_argument("--position-jump-threshold", type=float, default=2.0)
    parser.add_argument("--yaw-consistency-threshold", type=float, default=0.01)
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Maximum epochs; 0 trains until early stopping.",
    )
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.01)
    parser.add_argument("--rollout-horizons", default="1,5,10,20,50")
    parser.add_argument(
        "--rollout-scale-floors",
        default="position=0.05,yaw=0.005,vx=0.1,yawrate=0.01",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            REPO_ROOT, "outputs/formal_real_finetune_residual_vs_query"
        ),
    )
    return parser.parse_args()


def acquisition_minute(path):
    match = MINUTE_GROUP_PATTERN.match(Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot parse acquisition minute from {path}")
    date = match.group("date")
    key = f"{date}T{match.group('hour')}:{match.group('minute')}"
    return date, key


def discover_minute_groups(dataset_path):
    files = sorted(Path(dataset_path).glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pkl files under {dataset_path}")
    groups = defaultdict(list)
    group_dates = {}
    for path in files:
        date, key = acquisition_minute(path)
        groups[key].append(str(path))
        group_dates[key] = date
    return {
        key: sorted(group_files)
        for key, group_files in sorted(groups.items())
    }, group_dates


def split_minute_groups(groups, group_dates, train_ratio, val_ratio, seed, max_groups=0):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be within (0, 1)")
    if not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("--val-ratio must be positive and leave a non-empty test ratio")

    grouped_by_date = defaultdict(list)
    for key in groups:
        grouped_by_date[group_dates[key]].append(key)

    rng = random.Random(seed)
    for date in grouped_by_date:
        grouped_by_date[date].sort()
        rng.shuffle(grouped_by_date[date])

    if max_groups > 0:
        selected = []
        dates = sorted(grouped_by_date)
        while len(selected) < max_groups:
            changed = False
            for date in dates:
                if grouped_by_date[date] and len(selected) < max_groups:
                    selected.append(grouped_by_date[date].pop())
                    changed = True
            if not changed:
                break
        selected_set = set(selected)
        grouped_by_date = defaultdict(list)
        for key in selected:
            grouped_by_date[group_dates[key]].append(key)
        if len(selected_set) < 3:
            raise ValueError("--max-groups must retain at least three groups")

    split_groups = {"train": [], "val": [], "test": []}
    for date in sorted(grouped_by_date):
        keys = grouped_by_date[date]
        if len(keys) < 3:
            raise ValueError(
                f"Acquisition date {date} needs at least three minute groups"
            )
        train_count = min(max(1, round(len(keys) * train_ratio)), len(keys) - 2)
        val_count = min(
            max(1, round(len(keys) * val_ratio)),
            len(keys) - train_count - 1,
        )
        split_groups["train"].extend(keys[:train_count])
        split_groups["val"].extend(keys[train_count : train_count + val_count])
        split_groups["test"].extend(keys[train_count + val_count :])

    split_files = {}
    for split, keys in split_groups.items():
        split_groups[split] = sorted(keys)
        split_files[split] = [
            path for key in split_groups[split] for path in groups[key]
        ]
    return split_groups, split_files


def build_filtered_view(files, args, params, split):
    sequence_length = (
        args.history_length + 1 + args.prediction_length
    )
    raw_valid, audit = raw_window_validity(
        [Path(path) for path in files],
        sequence_length,
        args.quaternion_norm_tolerance,
        args.position_jump_threshold,
    )
    base = MujocoDataset(
        files,
        args.history_length + 1,
        args.prediction_length,
        delays=None,
        teacher_forcing=False,
        binary_mask=False,
        attack=False,
        use_zero_point=True,
    )
    if len(raw_valid) != len(base):
        raise RuntimeError(
            f"{split}: raw audit produced {len(raw_valid)} windows "
            f"but loader produced {len(base)}"
        )
    valid, invalid_yaw = add_yaw_consistency_filter(
        base,
        raw_valid,
        args.dt,
        args.yaw_consistency_threshold,
    )
    valid_indices = np.flatnonzero(valid)
    if args.max_episodes_per_split > 0:
        valid_indices = valid_indices[: args.max_episodes_per_split]
    audit["invalid_yaw_consistency_windows"] = invalid_yaw
    audit["valid_windows_before_smoke_limit"] = int(np.sum(valid))
    audit["selected_valid_windows"] = int(valid_indices.size)
    audit["invalid_windows"] = int(len(valid) - np.sum(valid))

    view = NuPlanKinematicResidualDataset(
        base,
        args.history_length,
        args.prediction_length,
        params,
        steer_shift=args.steer_shift,
    )
    return IndexedPredictionDataset(view, valid_indices), audit


def validate_checkpoints(checkpoints, args):
    for variant, checkpoint in checkpoints.items():
        if checkpoint.get("variant") != variant:
            raise ValueError(
                f"{variant} checkpoint declares variant={checkpoint.get('variant')}"
            )
        for name in ("history_length", "prediction_length"):
            if int(checkpoint["args"][name]) != int(getattr(args, name)):
                raise ValueError(
                    f"{variant} checkpoint {name}={checkpoint['args'][name]} "
                    f"does not match requested {getattr(args, name)}"
                )
        if not math.isclose(
            float(checkpoint["params"]["dt"]),
            args.dt,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{variant} checkpoint dt differs from --dt")

    reference = checkpoints["residual"]
    candidate = checkpoints["query"]
    for key in ("history", "context", "residual", "direct"):
        for index in (0, 1):
            if not torch.equal(
                reference["stats"][key][index],
                candidate["stats"][key][index],
            ):
                raise ValueError(f"Pretrained checkpoints differ in shared {key} stats")
    for key in ("dt", "wheelbase", "steering_ratio", "steering_offset"):
        if not math.isclose(
            float(reference["params"][key]),
            float(candidate["params"][key]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Pretrained checkpoints differ in parameter {key}")


def make_finetune_stats(checkpoint, real_residual, real_direct, device):
    stats = checkpoint_stats(checkpoint, device)
    stats["residual"] = tuple(value.to(device) for value in real_residual)
    stats["direct"] = tuple(value.to(device) for value in real_direct)
    return stats


def make_frozen_evaluation_stats(checkpoint, real_direct, device):
    stats = checkpoint_stats(checkpoint, device)
    stats["direct"] = tuple(value.to(device) for value in real_direct)
    return stats


def retarget_output_normalization(model, old_stats, new_stats):
    """Change normalized output coordinates without changing physical output."""
    old_mean, old_std = old_stats
    new_mean, new_std = new_stats
    scale = old_std / new_std
    shift = (old_mean - new_mean) / new_std
    output = model.embedding["output"]
    with torch.no_grad():
        output.weight.mul_(scale[:, None])
        output.bias.mul_(scale).add_(shift)


def normalization_equivalence_audit(
    variant,
    model,
    batch,
    old_stats,
    new_stats,
    params,
    device,
):
    model.eval()
    with torch.no_grad():
        history, action, context, mask = prepare_batch(
            batch,
            old_stats["history"][0],
            old_stats["history"][1],
            old_stats["context"][0],
            old_stats["context"][1],
            device,
        )
        nominal_state = nominal_transition = None
        if variant == "query":
            nominal_state, nominal_transition = prepare_nominal_query(
                batch, old_stats, device
            )
        old_normalized = forward_independent_history(
            model,
            history,
            action,
            context,
            mask,
            nominal_state,
            nominal_transition,
        )
        old_physical = (
            old_normalized * old_stats["residual"][1]
            + old_stats["residual"][0]
        )

        retarget_output_normalization(
            model, old_stats["residual"], new_stats["residual"]
        )
        new_normalized = forward_independent_history(
            model,
            history,
            action,
            context,
            mask,
            nominal_state,
            nominal_transition,
        )
        new_physical = (
            new_normalized * new_stats["residual"][1]
            + new_stats["residual"][0]
        )
    return {
        "max_abs_physical_residual_difference": float(
            torch.max(torch.abs(old_physical - new_physical)).item()
        ),
        "mean_abs_physical_residual_difference": float(
            torch.mean(torch.abs(old_physical - new_physical)).item()
        ),
    }


def checkpoint_payload(
    variant,
    model,
    source_checkpoint,
    source_path,
    stats,
    params,
    args,
    epoch,
    selection_value,
    val_losses,
    val_metrics,
    rollout_horizons,
    rollout_scales,
    learning_rate,
):
    return {
        "model_state_dict": model.state_dict(),
        "protocol": PROTOCOL_VERSION,
        "variant": variant,
        "stats": {
            key: [value[0].detach().cpu(), value[1].detach().cpu()]
            for key, value in stats.items()
        },
        "params": dict(source_checkpoint["params"]),
        "args": dict(source_checkpoint["args"]),
        "epoch": epoch,
        "selection_metric": "real_validation_rollout_score",
        "selection_value": selection_value,
        "val_loss": val_losses["transition"],
        "val_objective": val_losses["objective"],
        "val_rollout_score": val_losses["rollout"],
        "val_metrics": val_metrics,
        "rollout_horizons": rollout_horizons,
        "rollout_scales": rollout_scales.detach().cpu(),
        "learning_rate": learning_rate,
        "pretrained_checkpoint": str(source_path),
        "pretrained_epoch": int(source_checkpoint["epoch"]),
        "real_finetune_args": vars(args),
    }


def save_training_curves(results, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(20.8, 4.5))
    for variant, result in results.items():
        history = result["history"]
        train = [item for item in history if "train_objective" in item]
        validation = [item for item in history if "val_loss" in item]
        axes[0].plot(
            [item["epoch"] for item in train],
            [item["train_objective"] for item in train],
            label=variant,
        )
        axes[1].plot(
            [item["epoch"] for item in validation],
            [item["val_loss"] for item in validation],
            marker="o",
            label=variant,
        )
        axes[2].plot(
            [item["epoch"] for item in validation],
            [item["val_rollout_score"] for item in validation],
            marker="o",
            label=variant,
        )
        axes[3].plot(
            [item["epoch"] for item in train],
            [item["learning_rate"] for item in train],
            label=variant,
        )
    axes[0].set(title="Real train objective", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Real validation transition", xlabel="Epoch", ylabel="Loss")
    axes[2].set(title="Real validation rollout", xlabel="Epoch", ylabel="Score")
    axes[3].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
    axes[0].set_yscale("log")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def comparison(reference, candidate):
    return {
        "all_steps": relative_improvement(
            reference["all_steps"], candidate["all_steps"]
        ),
        "horizons": {
            horizon: relative_improvement(
                reference["horizons"][horizon],
                candidate["horizons"][horizon],
            )
            for horizon in reference["horizons"]
        },
        "transition_rmse": relative_improvement(
            reference["transition_rmse"], candidate["transition_rmse"]
        ),
        "common_transition_loss": scalar_relative_improvement(
            reference["common_transition_loss"],
            candidate["common_transition_loss"],
        ),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current model implementation")
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    if args.val_every < 1:
        raise ValueError("--val-every must be positive")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be positive")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be non-negative")
    if not 0.0 < args.lr_decay <= 1.0:
        raise ValueError("--lr-decay must be within (0, 1]")
    if args.rollout_loss_weight < 0.0:
        raise ValueError("--rollout-loss-weight must be non-negative")

    device = torch.device(args.device)
    checkpoint_paths = {
        "residual": Path(args.residual_checkpoint),
        "query": Path(args.query_checkpoint),
    }
    checkpoints = {
        variant: torch.load(path, map_location=device, weights_only=False)
        for variant, path in checkpoint_paths.items()
    }
    args.history_length = int(checkpoints["residual"]["args"]["history_length"])
    args.prediction_length = int(
        checkpoints["residual"]["args"]["prediction_length"]
    )
    validate_checkpoints(checkpoints, args)
    params = checkpoint_params(checkpoints["residual"], args.dt)
    rollout_horizons = parse_rollout_horizons(
        args.rollout_horizons, args.prediction_length
    )
    rollout_scale_floors = parse_rollout_scale_floors(
        args.rollout_scale_floors
    )

    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, group_dates = discover_minute_groups(args.dataset_path)
    split_groups, split_files = split_minute_groups(
        groups,
        group_dates,
        args.train_ratio,
        args.val_ratio,
        args.seed,
        args.max_groups,
    )
    for split in ("train", "val", "test"):
        (output_dir / f"{split}_groups.txt").write_text(
            "\n".join(split_groups[split]) + "\n"
        )
        (output_dir / f"{split}_files.txt").write_text(
            "\n".join(split_files[split]) + "\n"
        )

    datasets = {}
    audits = {}
    for split in ("train", "val", "test"):
        datasets[split], audits[split] = build_filtered_view(
            split_files[split], args, params, split
        )
    real_residual = streaming_stats(
        datasets["train"], "residual_target", args.eval_batch_size
    )
    real_direct = streaming_stats(
        datasets["train"], "direct_target", args.eval_batch_size
    )
    rollout_scales = rollout_scale_stats(
        datasets["train"],
        rollout_horizons,
        rollout_scale_floors,
        args.eval_batch_size,
    ).to(device)

    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    audit_batch = next(iter(val_loader))

    training_results = {}
    for variant in ("residual", "query"):
        set_seed(args.seed)
        source = checkpoints[variant]
        model_args = SimpleNamespace(**source["args"])
        model_args.device = args.device
        model = make_model(model_args, device, variant=variant)
        model.load_state_dict(source["model_state_dict"])
        old_stats = checkpoint_stats(source, device)
        stats = make_finetune_stats(
            source, real_residual, real_direct, device
        )
        normalization_audit = normalization_equivalence_audit(
            variant,
            model,
            audit_batch,
            old_stats,
            stats,
            params,
            device,
        )
        if normalization_audit["max_abs_physical_residual_difference"] > 1e-5:
            raise RuntimeError(
                f"{variant} output normalization retarget changed physical output: "
                f"{normalization_audit}"
            )

        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_loader = DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_decay
        )
        best_path = output_dir / f"{variant}_best.pt"

        epoch_zero_losses = run_epoch(
            model,
            val_loader,
            None,
            variant,
            stats,
            device,
            train=False,
            params=params,
            rollout_horizons=rollout_horizons,
            rollout_scales=rollout_scales,
            rollout_loss_weight=args.rollout_loss_weight,
        )
        epoch_zero_metrics = evaluate_model(
            variant,
            model,
            val_loader,
            stats,
            params,
            device,
            max_episodes=0,
        )
        best_selection_value = epoch_zero_losses["rollout"]
        best_epoch = 0
        torch.save(
            checkpoint_payload(
                variant,
                model,
                source,
                checkpoint_paths[variant],
                stats,
                params,
                args,
                0,
                best_selection_value,
                epoch_zero_losses,
                epoch_zero_metrics,
                rollout_horizons,
                rollout_scales,
                args.lr,
            ),
            best_path,
        )
        history = [
            {
                "epoch": 0,
                "val_loss": epoch_zero_losses["transition"],
                "val_objective": epoch_zero_losses["objective"],
                "val_rollout_score": epoch_zero_losses["rollout"],
                "val_metrics": epoch_zero_metrics,
                "selection_value": best_selection_value,
            }
        ]
        checks_without_improvement = 0
        early_stopped = False
        epoch = 0
        print(
            f"{variant} epoch=0 validation "
            f"transition={epoch_zero_losses['transition']:.6f} "
            f"rollout={epoch_zero_losses['rollout']:.6f}"
        )

        while args.epochs == 0 or epoch < args.epochs:
            epoch += 1
            learning_rate = optimizer.param_groups[0]["lr"]
            train_losses = run_epoch(
                model,
                train_loader,
                optimizer,
                variant,
                stats,
                device,
                train=True,
                params=params,
                rollout_horizons=rollout_horizons,
                rollout_scales=rollout_scales,
                rollout_loss_weight=args.rollout_loss_weight,
            )
            history_entry = {
                "epoch": epoch,
                "train_objective": train_losses["objective"],
                "train_transition_loss": train_losses["transition"],
                "train_rollout_score": train_losses["rollout"],
                "learning_rate": learning_rate,
            }
            history.append(history_entry)
            print(
                f"{variant} epoch={epoch} "
                f"train={train_losses['objective']:.6f} "
                f"transition={train_losses['transition']:.6f} "
                f"rollout={train_losses['rollout']:.6f} "
                f"lr={learning_rate:.8f}"
            )
            should_validate = epoch % args.val_every == 0 or (
                args.epochs > 0 and epoch == args.epochs
            )
            if should_validate:
                val_losses = run_epoch(
                    model,
                    val_loader,
                    None,
                    variant,
                    stats,
                    device,
                    train=False,
                    params=params,
                    rollout_horizons=rollout_horizons,
                    rollout_scales=rollout_scales,
                    rollout_loss_weight=args.rollout_loss_weight,
                )
                val_metrics = evaluate_model(
                    variant,
                    model,
                    val_loader,
                    stats,
                    params,
                    device,
                    max_episodes=0,
                )
                selection_value = val_losses["rollout"]
                history_entry.update(
                    {
                        "val_loss": val_losses["transition"],
                        "val_objective": val_losses["objective"],
                        "val_rollout_score": selection_value,
                        "val_metrics": val_metrics,
                        "selection_value": selection_value,
                    }
                )
                print(
                    f"{variant} validation epoch={epoch} "
                    f"transition={val_losses['transition']:.6f} "
                    f"rollout={selection_value:.6f} "
                    f"position={val_metrics['all_steps']['position_rmse']:.6f} "
                    f"yaw={val_metrics['all_steps']['yaw_rmse']:.6f} "
                    f"yawrate={val_metrics['all_steps']['yawrate_rmse']:.6f}"
                )
                if (
                    selection_value
                    < best_selection_value - args.early_stopping_min_delta
                ):
                    best_selection_value = selection_value
                    best_epoch = epoch
                    checks_without_improvement = 0
                    torch.save(
                        checkpoint_payload(
                            variant,
                            model,
                            source,
                            checkpoint_paths[variant],
                            stats,
                            params,
                            args,
                            epoch,
                            selection_value,
                            val_losses,
                            val_metrics,
                            rollout_horizons,
                            rollout_scales,
                            learning_rate,
                        ),
                        best_path,
                    )
                else:
                    checks_without_improvement += 1
                if checks_without_improvement >= args.early_stopping_patience:
                    early_stopped = True
                    print(
                        f"{variant} early stopping at epoch {epoch}; "
                        f"best validation rollout={best_selection_value:.6f} "
                        f"at epoch {best_epoch}"
                    )
                    break
            scheduler.step()

        training_results[variant] = {
            "best_epoch": best_epoch,
            "stopped_epoch": epoch,
            "early_stopped": early_stopped,
            "best_validation_rollout_score": best_selection_value,
            "normalization_equivalence": normalization_audit,
            "checkpoint": str(best_path),
            "history": history,
        }
        del model, optimizer, scheduler, train_loader
        torch.cuda.empty_cache()

    test_results = {}
    for variant in ("residual", "query"):
        source = checkpoints[variant]
        model_args = SimpleNamespace(**source["args"])
        model_args.device = args.device

        frozen_model = make_model(model_args, device, variant=variant)
        frozen_model.load_state_dict(source["model_state_dict"])
        frozen_stats = make_frozen_evaluation_stats(
            source, real_direct, device
        )
        test_results[f"{variant}_frozen"] = evaluate_model(
            variant,
            frozen_model,
            test_loader,
            frozen_stats,
            params,
            device,
            max_episodes=0,
        )
        test_results[f"{variant}_frozen"]["checkpoint"] = str(
            checkpoint_paths[variant]
        )
        del frozen_model
        torch.cuda.empty_cache()

        fine_checkpoint = torch.load(
            training_results[variant]["checkpoint"],
            map_location=device,
            weights_only=False,
        )
        fine_model = make_model(model_args, device, variant=variant)
        fine_model.load_state_dict(fine_checkpoint["model_state_dict"])
        fine_stats = checkpoint_stats(fine_checkpoint, device)
        test_results[f"{variant}_finetuned"] = evaluate_model(
            variant,
            fine_model,
            test_loader,
            fine_stats,
            params,
            device,
            max_episodes=0,
        )
        test_results[f"{variant}_finetuned"]["checkpoint"] = training_results[
            variant
        ]["checkpoint"]
        del fine_model
        torch.cuda.empty_cache()

    comparisons = {
        "residual_finetune_improvement_percent": comparison(
            test_results["residual_frozen"],
            test_results["residual_finetuned"],
        ),
        "query_finetune_improvement_percent": comparison(
            test_results["query_frozen"],
            test_results["query_finetuned"],
        ),
        "finetuned_query_vs_finetuned_residual_improvement_percent": comparison(
            test_results["residual_finetuned"],
            test_results["query_finetuned"],
        ),
        "frozen_query_vs_frozen_residual_improvement_percent": comparison(
            test_results["residual_frozen"],
            test_results["query_frozen"],
        ),
    }

    split_summary = {
        split: {
            "minute_groups": len(split_groups[split]),
            "files": len(split_files[split]),
            "audit": audits[split],
        }
        for split in ("train", "val", "test")
    }
    summary = {
        "protocol": PROTOCOL_VERSION,
        "args": vars(args),
        "dataset_path": args.dataset_path,
        "split_unit": "acquisition_date_hour_minute",
        "split": split_summary,
        "params": dict(checkpoints["residual"]["params"]),
        "pretrained_checkpoints": {
            variant: str(path) for variant, path in checkpoint_paths.items()
        },
        "real_train_stats": {
            "residual": {
                "mean": real_residual[0].tolist(),
                "std": real_residual[1].tolist(),
            },
            "direct": {
                "mean": real_direct[0].tolist(),
                "std": real_direct[1].tolist(),
            },
            "rollout_scales": {
                str(horizon): {
                    metric: float(rollout_scales[horizon_index, metric_index])
                    for metric_index, metric in enumerate(ROLLOUT_METRIC_NAMES)
                }
                for horizon_index, horizon in enumerate(rollout_horizons)
            },
            "rollout_scale_floors": {
                key: rollout_scale_floors.get(
                    key, DEFAULT_ROLLOUT_SCALE_FLOORS[key]
                )
                for key in ROLLOUT_METRIC_NAMES
            },
        },
        "training": training_results,
        "test": test_results,
        "comparisons": comparisons,
        "training_curves": str(output_dir / "training_curves.png"),
    }
    save_training_curves(training_results, output_dir / "training_curves.png")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"test": test_results, "comparisons": comparisons}, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
