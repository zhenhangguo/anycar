#!/usr/bin/env python3
"""Train and compare physically consistent direct and residual predictors.

This experiment intentionally uses the observable five-state protocol
``[x, y, yaw, vx, yawrate]``.  nuPlan ``vy`` remains available for offline
analysis but is neither an input nor a supervised output here.
Yaw is integrated from the current yawrate under the nuPlan v2 protocol and is
not predicted as an independent transition channel.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, REPO_ROOT)

from car_foundation.dataset import MujocoDataset
from car_foundation.kinematic_residual import (
    KinematicBicycleParams,
    NuPlanKinematicResidualDataset,
    OBSERVABLE_STATE_INDICES,
    ROLLOUT_METRIC_NAMES,
    multi_horizon_rollout_loss,
    rollout_metric_squared_errors,
    rollout_consistent_transition_sequence,
    rollout_kinematic_residual_consistent,
    states_relative_to_initial_body,
)
from car_foundation.models import (
    TorchTransformerDecoderCurrentStateMLP,
    TorchTransformerDecoderKinematicQueryMLP,
)


LOSS_WEIGHTS = torch.tensor([0.5, 0.5, 0.5, 2.5], dtype=torch.float32)
HORIZON_STEPS = (1, 5, 10, 20, 50)
DEFAULT_ROLLOUT_SCALE_FLOORS = {
    "position": 0.05,
    "yaw": 0.005,
    "vx": 0.1,
    "yawrate": 0.01,
}
PROTOCOL_VERSION = "consistent_v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare nuPlan direct and kinematic-residual prediction."
    )
    parser.add_argument(
        "--dataset-path",
        default="/disk1/collect_data_from_anycar/New_demo/new_data_with_x_mean_zero/total_data_1",
    )
    parser.add_argument("--max-files", type=int, default=0, help="0 uses all pkl files.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--history-length", type=int, default=250)
    parser.add_argument("--prediction-length", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--wheelbase", type=float, default=3.9)
    parser.add_argument("--steering-ratio", type=float, default=25.0)
    parser.add_argument("--steering-offset-deg", type=float, default=5.0)
    parser.add_argument(
        "--steer-shift",
        type=int,
        default=1,
        help="Use steer[t + shift] for transition t->t+1; nuPlan generator uses 1.",
    )
    parser.add_argument("--models", default="direct,residual")
    parser.add_argument("--include-kinematic", action="store_true")
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Maximum training epochs; 0 trains until early stopping.",
    )
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument(
        "--selection-metric",
        choices=("val_loss", "rollout_score"),
        default="val_loss",
        help=(
            "Metric used for checkpoint selection and early stopping. "
            "rollout_score is the shared dimensionless multi-horizon rollout MSE."
        ),
    )
    parser.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of dimensionless multi-horizon rollout loss added to the "
            "normalized transition loss; 0 preserves the R0/RQ objective."
        ),
    )
    parser.add_argument(
        "--rollout-horizons",
        default="1,5,10,20,50",
        help="One-based future steps used by rollout loss and rollout_score.",
    )
    parser.add_argument(
        "--rollout-scale-floors",
        default="position=0.05,yaw=0.005,vx=0.1,yawrate=0.01",
        help=(
            "Physical lower bounds for train-set nominal-error RMSE scales. "
            "Order-independent metric=value pairs."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help=(
            "Stop after this many validation checks without improvement; "
            "0 disables early stopping for a finite run."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Required metric decrease to reset early-stopping patience.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.99,
        help="Multiplicative learning-rate decay applied after every epoch.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--query-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-eval-episodes", type=int, default=0)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "kinematic_residual_ablation"),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_rollout_horizons(spec, prediction_length):
    horizons = tuple(int(item.strip()) for item in spec.split(",") if item.strip())
    if not horizons:
        raise ValueError("--rollout-horizons must contain at least one step.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("--rollout-horizons must not contain duplicates.")
    if any(horizon < 1 or horizon > prediction_length for horizon in horizons):
        raise ValueError(
            f"--rollout-horizons must be within [1, {prediction_length}]."
        )
    return tuple(sorted(horizons))


def parse_rollout_scale_floors(spec):
    floors = dict(DEFAULT_ROLLOUT_SCALE_FLOORS)
    for item in spec.split(","):
        if not item.strip():
            continue
        key, separator, value = item.partition("=")
        key = key.strip()
        if not separator or key not in floors:
            raise ValueError(f"Unknown rollout scale floor: {item}")
        floors[key] = float(value)
    if any(value <= 0 for value in floors.values()):
        raise ValueError("All rollout scale floors must be strictly positive.")
    return floors


def params_dict(params):
    return {
        "dt": float(params.dt),
        "wheelbase": float(params.wheelbase),
        "steering_ratio": float(params.steering_ratio),
        "steering_offset": float(params.steering_offset),
    }


def split_files(args):
    files = sorted(Path(args.dataset_path).glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pkl files under {args.dataset_path}")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.max_files > 0:
        files = files[: args.max_files]
    if len(files) < 3:
        raise ValueError("At least three pkl files are required for train/val/test.")
    train_end = min(max(1, int(len(files) * args.train_ratio)), len(files) - 2)
    requested_val = max(1, int(len(files) * args.val_ratio))
    val_end = min(train_end + requested_val, len(files) - 1)
    return [str(p) for p in files[:train_end]], [str(p) for p in files[train_end:val_end]], [str(p) for p in files[val_end:]]


def make_base_dataset(files, args, mean=None, std=None):
    return MujocoDataset(
        files,
        args.history_length + 1,
        args.prediction_length,
        delays=None,
        mean=mean,
        std=std,
        teacher_forcing=False,
        binary_mask=False,
        attack=False,
        use_zero_point=True,
    )


def make_view(base, args, params):
    return NuPlanKinematicResidualDataset(
        base,
        args.history_length,
        args.prediction_length,
        params,
        steer_shift=args.steer_shift,
    )


def streaming_stats(dataset, key, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    output_dim = dataset[0][key].shape[-1]
    total = torch.zeros(output_dim, dtype=torch.float64)
    total_sq = torch.zeros(output_dim, dtype=torch.float64)
    count = 0
    for batch in tqdm(loader, desc=f"stats {key}", leave=False):
        value = batch[key].to(torch.float64).reshape(-1, output_dim)
        total += value.sum(dim=0)
        total_sq += (value * value).sum(dim=0)
        count += value.shape[0]
    mean = total / count
    variance = (total_sq / count - mean * mean).clamp_min(1e-12)
    return mean.float(), variance.sqrt().float()


def context_stats(dataset):
    start = dataset.model_history_length
    data = dataset.dataset.data
    context = torch.stack(
        (data[:, start, 3], data[:, start, 5], data[:, start, 6], data[:, start, 7]),
        dim=-1,
    )
    return context.mean(dim=0), context.std(dim=0).clamp_min(1e-6)


def rollout_scale_stats(dataset, horizons, floors, batch_size):
    """Estimate train-only physical scales from nominal rollout error RMSE.

    Each horizon/metric is normalized by the nominal kinematic model's train
    RMSE, with a physical floor to prevent exact or nearly exact channels (for
    example nuPlan ``vx``) from amplifying floating-point noise.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    squared_error_sum = torch.zeros(
        len(horizons), len(ROLLOUT_METRIC_NAMES), dtype=torch.float64
    )
    count = 0
    for batch in tqdm(loader, desc="stats rollout_scale", leave=False):
        initial = batch["initial_state"]
        truth_relative = states_relative_to_initial_body(initial, batch["truth"])
        squared_errors = rollout_metric_squared_errors(
            batch["nominal_state_rel"], truth_relative, horizons
        )
        squared_error_sum += squared_errors.to(torch.float64).sum(dim=0)
        count += squared_errors.shape[0]
    scales = torch.sqrt(squared_error_sum / max(count, 1)).to(torch.float32)
    floor_tensor = torch.tensor(
        [floors[name] for name in ROLLOUT_METRIC_NAMES], dtype=torch.float32
    )
    scales = torch.maximum(scales, floor_tensor.unsqueeze(0))
    return scales


def make_model(args, device, variant=None):
    model_class = (
        TorchTransformerDecoderKinematicQueryMLP
        if variant == "query"
        else TorchTransformerDecoderCurrentStateMLP
    )
    model_kwargs = dict(
        state_dim=5,
        action_dim=2,
        output_dim=4,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        device=device,
        dropout=args.dropout,
        history_length=args.history_length,
        prediction_length=args.prediction_length,
        compressed_history_length=42,
        current_dim=4,
        fusion_hidden_dim=args.fusion_hidden_dim,
    )
    if variant == "query":
        model_kwargs.update(
            nominal_state_dim=5,
            nominal_transition_dim=4,
            query_hidden_dim=args.query_hidden_dim,
        )
    return model_class(**model_kwargs).to(device)


def forward_independent_history(
    model,
    history,
    action,
    context,
    mask=None,
    nominal_state=None,
    nominal_transition=None,
):
    history_emb = model.position_encoding["history"](model._build_history_emb(history))
    if nominal_state is None and nominal_transition is None:
        action_embedding = model._build_action_emb(history, action, context)
    else:
        action_embedding = model._build_action_emb(
            history,
            action,
            context,
            nominal_state,
            nominal_transition,
        )
    action_emb = model.position_encoding["action"](action_embedding)
    out = model.transformer_decoder(
        tgt=action_emb,
        memory=history_emb,
        tgt_mask=model.tgt_mask,
        tgt_key_padding_mask=mask.to(model.device) if mask is not None else None,
    )
    return model.embedding["output"](out)


def prepare_batch(batch, history_mean, history_std, context_mean, context_std, device):
    history = batch["history"].to(device).clone()
    history[:, :, :5] = (history[:, :, :5] - history_mean) / history_std
    history = history[:, 1:, :]
    action = batch["action"].to(device)
    context = (batch["current_context"].to(device) - context_mean) / context_std
    mask = batch["mask"].to(device)
    return history, action, context, mask


def prepare_nominal_query(batch, stats, device):
    nominal_state = batch["nominal_state_rel"].to(device)
    nominal_state = (
        nominal_state - stats["nominal_state"][0]
    ) / stats["nominal_state"][1]
    nominal_transition = batch["nominal_transition"].to(device)
    nominal_transition = (
        nominal_transition - stats["nominal_transition"][0]
    ) / stats["nominal_transition"][1]
    return nominal_state, nominal_transition


def physical_rollout(variant, initial, action, transition, params):
    if variant == "direct":
        return rollout_consistent_transition_sequence(initial, transition, params)
    return rollout_kinematic_residual_consistent(
        initial, action, transition, params
    )


def run_epoch(
    model,
    loader,
    optimizer,
    variant,
    stats,
    device,
    train,
    params,
    rollout_horizons=(),
    rollout_scales=None,
    rollout_loss_weight=0.0,
):
    model.train(train)
    target_variant = "direct" if variant == "direct" else "residual"
    target_key = "direct_target" if variant == "direct" else "residual_target"
    target_mean, target_std = stats[target_variant]
    objective_total = 0.0
    transition_total = 0.0
    batch_count = 0
    rollout_total = 0.0
    rollout_count = 0
    context_manager = torch.enable_grad() if train else torch.no_grad()
    with context_manager:
        for batch in tqdm(loader, desc=f"{variant} {'train' if train else 'val'}", leave=False):
            history, action, context, mask = prepare_batch(
                batch,
                stats["history"][0],
                stats["history"][1],
                stats["context"][0],
                stats["context"][1],
                device,
            )
            target = batch[target_key].to(device)
            target = (target - target_mean) / target_std
            if variant == "query":
                nominal_state, nominal_transition = prepare_nominal_query(
                    batch, stats, device
                )
                if train:
                    prediction = model(
                        history,
                        action,
                        current_state=context,
                        nominal_state=nominal_state,
                        nominal_transition=nominal_transition,
                        action_padding_mask=mask,
                    )
                else:
                    prediction = forward_independent_history(
                        model,
                        history,
                        action,
                        context,
                        mask,
                        nominal_state,
                        nominal_transition,
                    )
            elif train:
                prediction = model(
                    history,
                    action,
                    current_state=context,
                    action_padding_mask=mask,
                )
            else:
                prediction = forward_independent_history(model, history, action, context, mask)
            weights = LOSS_WEIGHTS.to(device)[None, None, :]
            transition_loss = torch.mean((prediction - target) ** 2 * weights)
            rollout_loss = None
            if rollout_scales is not None:
                pred_value = prediction * target_std + target_mean
                if train and rollout_loss_weight == 0.0:
                    with torch.no_grad():
                        predicted_states = physical_rollout(
                            variant,
                            batch["initial_state"].to(device),
                            action,
                            pred_value.detach(),
                            params,
                        )
                        rollout_loss = multi_horizon_rollout_loss(
                            predicted_states,
                            batch["truth"].to(device),
                            rollout_scales,
                            rollout_horizons,
                        )
                else:
                    predicted_states = physical_rollout(
                        variant,
                        batch["initial_state"].to(device),
                        action,
                        pred_value,
                        params,
                    )
                    rollout_loss = multi_horizon_rollout_loss(
                        predicted_states,
                        batch["truth"].to(device),
                        rollout_scales,
                        rollout_horizons,
                    )
            objective = transition_loss
            if rollout_loss is not None and rollout_loss_weight > 0.0:
                objective = objective + rollout_loss_weight * rollout_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
            objective_total += objective.detach().item()
            transition_total += transition_loss.detach().item()
            batch_count += 1
            if rollout_loss is not None:
                current_batch = prediction.shape[0]
                rollout_total += rollout_loss.detach().item() * current_batch
                rollout_count += current_batch
    result = {
        "objective": objective_total / max(batch_count, 1),
        "transition": transition_total / max(batch_count, 1),
    }
    if rollout_scales is not None:
        result["rollout"] = rollout_total / max(rollout_count, 1)
    return result


def empty_metric_sums():
    return {name: 0.0 for name in ("position", "yaw", "vx", "yawrate")}, 0


def add_metrics(sums, count, prediction, truth):
    diff = prediction - truth
    yaw_diff = torch.atan2(torch.sin(diff[..., 2]), torch.cos(diff[..., 2]))
    sums["position"] += torch.sum(diff[..., 0] ** 2 + diff[..., 1] ** 2).item()
    sums["yaw"] += torch.sum(yaw_diff**2).item()
    sums["vx"] += torch.sum(diff[..., 3] ** 2).item()
    sums["yawrate"] += torch.sum(diff[..., 4] ** 2).item()
    return count + prediction.shape[0] * prediction.shape[1]


def summarize_metrics(sums, count):
    return {f"{name}_rmse": float(np.sqrt(value / max(count, 1))) for name, value in sums.items()}


def save_training_curves(results, output_path):
    has_rollout = any(
        any("val_rollout_score" in item for item in result.get("history", []))
        for result in results.values()
    )
    column_count = 4 if has_rollout else 3
    fig, axes = plt.subplots(1, column_count, figsize=(5.2 * column_count, 4.5))
    for variant, result in results.items():
        history = result.get("history", [])
        if not history:
            continue
        epochs = [item["epoch"] for item in history]
        axes[0].plot(
            epochs,
            [item["train_loss"] for item in history],
            label=variant,
        )
        validation = [item for item in history if "val_loss" in item]
        axes[1].plot(
            [item["epoch"] for item in validation],
            [item["val_loss"] for item in validation],
            marker="o",
            label=variant,
        )
        learning_rate_axis = 3 if has_rollout else 2
        if has_rollout:
            rollout_validation = [
                item for item in validation if "val_rollout_score" in item
            ]
            if rollout_validation:
                axes[2].plot(
                    [item["epoch"] for item in rollout_validation],
                    [item["val_rollout_score"] for item in rollout_validation],
                    marker="o",
                    label=variant,
                )
        axes[learning_rate_axis].plot(
            epochs,
            [item["learning_rate"] for item in history],
            label=variant,
        )

    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation transition loss", xlabel="Epoch", ylabel="Loss")
    if has_rollout:
        axes[2].set(title="Validation rollout score", xlabel="Epoch", ylabel="Score")
    axes[-1].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
    axes[0].set_yscale("log")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def evaluate(name, model, loader, stats, params, device, max_episodes=0):
    sums, count = empty_metric_sums()
    horizon_sums = {h: empty_metric_sums() for h in HORIZON_STEPS if h <= loader.dataset.prediction_length}
    seen = 0
    if model is not None:
        model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {name}", leave=False):
            initial = batch["initial_state"].to(device)
            truth = batch["truth"].to(device)
            action = batch["action"].to(device)
            if max_episodes > 0 and seen >= max_episodes:
                break
            if name == "kinematic":
                residual = torch.zeros_like(batch["residual_target"], device=device)
                prediction = rollout_kinematic_residual_consistent(
                    initial, action, residual, params
                )
            else:
                history, action, context, mask = prepare_batch(
                    batch,
                    stats["history"][0],
                    stats["history"][1],
                    stats["context"][0],
                    stats["context"][1],
                    device,
                )
                if name == "query":
                    nominal_state, nominal_transition = prepare_nominal_query(
                        batch, stats, device
                    )
                    pred_norm = forward_independent_history(
                        model,
                        history,
                        action,
                        context,
                        mask,
                        nominal_state,
                        nominal_transition,
                    )
                else:
                    pred_norm = forward_independent_history(
                        model, history, action, context, mask
                    )
                target_variant = "direct" if name == "direct" else "residual"
                target_mean, target_std = stats[target_variant]
                pred_value = pred_norm * target_std + target_mean
                if name == "direct":
                    prediction = rollout_consistent_transition_sequence(
                        initial, pred_value, params
                    )
                else:
                    prediction = rollout_kinematic_residual_consistent(
                        initial, action, pred_value, params
                    )
            count = add_metrics(sums, count, prediction, truth)
            for horizon, (h_sums, h_count) in horizon_sums.items():
                h_count = add_metrics(
                    h_sums, h_count, prediction[:, horizon - 1 : horizon], truth[:, horizon - 1 : horizon]
                )
                horizon_sums[horizon] = (h_sums, h_count)
            seen += prediction.shape[0]
    result = {"all_steps": summarize_metrics(sums, count), "evaluated_episodes": seen}
    result["horizons"] = {
        str(h): summarize_metrics(h_sums, h_count)
        for h, (h_sums, h_count) in horizon_sums.items()
    }
    return result


def main():
    args = parse_args()
    if args.history_length != 250:
        raise ValueError(
            "The current CNN history compressor is configured for 250 input frames."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the current TorchTransformerDecoder implementation.")
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative.")
    if args.val_every < 1:
        raise ValueError("--val-every must be positive.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative.")
    if args.epochs == 0 and args.early_stopping_patience == 0:
        raise ValueError("Unlimited training requires positive early-stopping patience.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be non-negative.")
    if args.rollout_loss_weight < 0:
        raise ValueError("--rollout-loss-weight must be non-negative.")
    if not 0.0 < args.lr_decay <= 1.0:
        raise ValueError("--lr-decay must be within (0, 1].")
    rollout_horizons = parse_rollout_horizons(
        args.rollout_horizons, args.prediction_length
    )
    rollout_scale_floors = parse_rollout_scale_floors(
        args.rollout_scale_floors
    )
    use_rollout_objective = args.rollout_loss_weight > 0.0
    use_rollout_score = (
        use_rollout_objective or args.selection_metric == "rollout_score"
    )
    set_seed(args.seed)
    device = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files, test_files = split_files(args)
    for name, files in (("train", train_files), ("val", val_files), ("test", test_files)):
        (output_dir / f"{name}_files.txt").write_text("\n".join(files) + "\n")

    params = KinematicBicycleParams(
        dt=args.dt,
        wheelbase=args.wheelbase,
        steering_ratio=args.steering_ratio,
        steering_offset=np.deg2rad(args.steering_offset_deg),
    )
    train_base = make_base_dataset(train_files, args)
    val_base = make_base_dataset(val_files, args, train_base.mean, train_base.std)
    test_base = make_base_dataset(test_files, args, train_base.mean, train_base.std)
    train_view = make_view(train_base, args, params)
    val_view = make_view(val_base, args, params)
    test_view = make_view(test_base, args, params)

    observable = torch.tensor(OBSERVABLE_STATE_INDICES, dtype=torch.long)
    history_mean = train_base.mean[observable].to(device)
    history_std = train_base.std[observable].clamp_min(1e-6).to(device)
    direct_mean, direct_std = streaming_stats(
        train_view, "direct_target", args.eval_batch_size
    )
    residual_mean, residual_std = streaming_stats(
        train_view, "residual_target", args.eval_batch_size
    )
    context_mean, context_std = context_stats(train_view)
    stats = {
        "history": (history_mean, history_std),
        "context": (context_mean.to(device), context_std.to(device)),
        "direct": (direct_mean.to(device), direct_std.to(device)),
        "residual": (residual_mean.to(device), residual_std.to(device)),
    }
    rollout_scales = None
    if use_rollout_score:
        rollout_scales = rollout_scale_stats(
            train_view,
            rollout_horizons,
            rollout_scale_floors,
            args.eval_batch_size,
        ).to(device)

    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    if "query" in requested:
        nominal_state_mean, nominal_state_std = streaming_stats(
            train_view, "nominal_state_rel", args.eval_batch_size
        )
        nominal_transition_mean, nominal_transition_std = streaming_stats(
            train_view, "nominal_transition", args.eval_batch_size
        )
        stats["nominal_state"] = (
            nominal_state_mean.to(device),
            nominal_state_std.to(device),
        )
        stats["nominal_transition"] = (
            nominal_transition_mean.to(device),
            nominal_transition_std.to(device),
        )

    train_loader = DataLoader(
        train_view,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_view, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_view, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    results = {}
    if args.include_kinematic:
        results["kinematic"] = evaluate(
            "kinematic", None, test_loader, stats, params, device, args.max_eval_episodes
        )
    for variant in requested:
        if variant not in ("direct", "residual", "query"):
            raise ValueError(f"Unknown model variant: {variant}")
        set_seed(args.seed)
        model = make_model(args, device, variant)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_decay
        )
        # The query-only branch allocates additional randomly initialized
        # parameters after all shared R0 parameters.  Reset the random stream so
        # sampler order and dropout masks remain comparable across variants.
        if "query" in requested:
            set_seed(args.seed)
        best_selection_value = float("inf")
        best_epoch = 0
        checks_without_improvement = 0
        early_stopped = False
        best_path = output_dir / f"{variant}_best.pt"
        history = []
        epoch = 0
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
                "train_loss": train_losses["objective"],
                "train_transition_loss": train_losses["transition"],
                "learning_rate": learning_rate,
            }
            if "rollout" in train_losses:
                history_entry["train_rollout_score"] = train_losses["rollout"]
            history.append(history_entry)
            train_rollout_text = (
                f"rollout={train_losses['rollout']:.6f} "
                if "rollout" in train_losses
                else ""
            )
            print(
                f"{variant} epoch={epoch} train={train_losses['objective']:.6f} "
                f"transition={train_losses['transition']:.6f} "
                f"{train_rollout_text}"
                f"lr={learning_rate:.8f}"
            )
            should_validate = epoch % args.val_every == 0 or (
                args.epochs > 0 and epoch == args.epochs
            )
            if not should_validate:
                scheduler.step()
                continue

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
            val_loss = val_losses["transition"]
            val_metrics = evaluate(variant, model, val_loader, stats, params, device)
            val_all_steps = val_metrics["all_steps"]
            selection_value = (
                val_losses["rollout"]
                if args.selection_metric == "rollout_score"
                else val_loss
            )
            history_entry.update(
                {
                    "val_loss": val_loss,
                    "val_objective": val_losses["objective"],
                    "val_metrics": val_metrics,
                    "selection_value": selection_value,
                }
            )
            if "rollout" in val_losses:
                history_entry["val_rollout_score"] = val_losses["rollout"]
            val_rollout_text = (
                f"rollout={val_losses['rollout']:.6f} "
                if "rollout" in val_losses
                else ""
            )
            print(
                f"{variant} validation epoch={epoch} val={val_loss:.6f} "
                f"{val_rollout_text}"
                f"val_position={val_all_steps['position_rmse']:.6f} "
                f"val_yaw={val_all_steps['yaw_rmse']:.6f} "
                f"val_yawrate={val_all_steps['yawrate_rmse']:.6f}"
            )
            improved = (
                selection_value
                < best_selection_value - args.early_stopping_min_delta
            )
            if improved:
                best_selection_value = selection_value
                best_epoch = epoch
                checks_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "protocol": PROTOCOL_VERSION,
                        "variant": variant,
                        "stats": {key: [value[0].cpu(), value[1].cpu()] for key, value in stats.items()},
                        "params": params_dict(params),
                        "args": vars(args),
                        "epoch": best_epoch,
                        "selection_metric": args.selection_metric,
                        "selection_value": best_selection_value,
                        "val_loss": val_loss,
                        "val_objective": val_losses["objective"],
                        "val_rollout_score": val_losses.get("rollout"),
                        "val_metrics": val_metrics,
                        "rollout_horizons": rollout_horizons,
                        "rollout_scales": (
                            rollout_scales.cpu()
                            if rollout_scales is not None
                            else None
                        ),
                        "learning_rate": learning_rate,
                    },
                    best_path,
                )
            else:
                checks_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and checks_without_improvement >= args.early_stopping_patience
            ):
                early_stopped = True
                print(
                    f"{variant} early stopping at epoch {epoch}: "
                    f"best {args.selection_metric}={best_selection_value:.6f} "
                    f"at epoch {best_epoch}; no improvement for "
                    f"{checks_without_improvement} validation checks"
                )
                break
            scheduler.step()
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        results[variant] = evaluate(
            variant, model, test_loader, stats, params, device, args.max_eval_episodes
        )
        results[variant]["selection_metric"] = args.selection_metric
        results[variant]["best_selection_value"] = best_selection_value
        results[variant]["best_epoch"] = best_epoch
        results[variant]["stopped_epoch"] = epoch
        results[variant]["early_stopped"] = early_stopped
        validation_history = [item for item in history if "val_loss" in item]
        results[variant]["validation_checks"] = len(validation_history)
        results[variant]["best_val_loss"] = min(
            item["val_loss"] for item in validation_history
        )
        results[variant]["checkpoint_val_loss"] = checkpoint["val_loss"]
        if rollout_scales is not None:
            results[variant]["min_val_rollout_score"] = min(
                item["val_rollout_score"] for item in validation_history
            )
            results[variant]["checkpoint_rollout_score"] = checkpoint[
                "val_rollout_score"
            ]
        results[variant]["checkpoint"] = str(best_path)
        results[variant]["history"] = history

    summary = {
        "protocol": PROTOCOL_VERSION,
        "args": vars(args),
        "params": params_dict(params),
        "rollout": {
            "horizons": rollout_horizons,
            "loss_weight": args.rollout_loss_weight,
            "scale_source": "train_nominal_error_rmse_with_physical_floor",
            "scale_floors": rollout_scale_floors,
            "scales": (
                {
                    str(horizon): {
                        name: float(rollout_scales[index, metric_index].item())
                        for metric_index, name in enumerate(ROLLOUT_METRIC_NAMES)
                    }
                    for index, horizon in enumerate(rollout_horizons)
                }
                if rollout_scales is not None
                else None
            ),
        },
        "dataset": {
            "train_files": len(train_files),
            "val_files": len(val_files),
            "test_files": len(test_files),
            "train_episodes": len(train_view),
            "val_episodes": len(val_view),
            "test_episodes": len(test_view),
        },
        "results": results,
        "training_curves": str(output_dir / "training_curves.png"),
    }
    save_training_curves(results, output_dir / "training_curves.png")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    print(f"summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
