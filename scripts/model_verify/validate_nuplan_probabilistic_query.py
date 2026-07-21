#!/usr/bin/env python3
"""Small frozen-mean probabilistic Query validation on fixed nuPlan splits."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    gaussian_nll,
)
from train_kinematic_residual_ablation import (
    LOSS_WEIGHTS,
    forward_independent_history,
    make_base_dataset,
    make_model,
    make_view,
    prepare_batch,
    prepare_nominal_query,
)


CHANNELS = ("dx_body", "dy_body", "dvx", "dyawrate")
HORIZONS = (1, 5, 10, 20, 50)
COVERAGE_LEVELS = {
    "68": (1.0, 0.682689492137),
    "90": (1.6448536269514722, 0.9),
    "95": (1.959963984540054, 0.95),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a frozen-mean probabilistic nuPlan Query residual."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            REPO_ROOT,
            "outputs/formal_residual_vs_query_20k/20260716T142015/query_best.pt",
        ),
    )
    parser.add_argument(
        "--source-run",
        default=os.path.join(
            REPO_ROOT,
            "outputs/formal_residual_vs_query_20k/20260716T142015",
        ),
    )
    parser.add_argument("--sigma-train-files", type=int, default=512)
    parser.add_argument("--sigma-val-files", type=int, default=256)
    parser.add_argument("--calibration-files", type=int, default=256)
    parser.add_argument("--test-files", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigma-floor", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            REPO_ROOT, "outputs", "nuplan_probabilistic_query_smoke"
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_file_list(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def fixed_splits(args):
    val_files = read_file_list(Path(args.source_run) / "val_files.txt")
    test_files = read_file_list(Path(args.source_run) / "test_files.txt")
    val_required = (
        args.sigma_train_files + args.sigma_val_files + args.calibration_files
    )
    if val_required > len(val_files):
        raise ValueError(
            f"Requested {val_required} validation files, only {len(val_files)} exist"
        )
    if args.test_files > len(test_files):
        raise ValueError(
            f"Requested {args.test_files} test files, only {len(test_files)} exist"
        )
    train_end = args.sigma_train_files
    val_end = train_end + args.sigma_val_files
    calibration_end = val_end + args.calibration_files
    splits = {
        "sigma_train": val_files[:train_end],
        "sigma_val": val_files[train_end:val_end],
        "calibration": val_files[val_end:calibration_end],
        "probability_test": test_files[: args.test_files],
    }
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise RuntimeError(f"{left}/{right} overlap: {len(overlap)} files")
    return splits


def checkpoint_stats(checkpoint, device):
    stats = {}
    for name, pair in checkpoint["stats"].items():
        stats[name] = tuple(value.to(device) for value in pair)
    return stats


def make_loader(files, model_args, params, batch_size, workers, shuffle):
    base = make_base_dataset(files, model_args)
    view = make_view(base, model_args, params)
    loader = DataLoader(
        view,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    return view, loader


def forward_batch(wrapper, batch, stats, device):
    history, action, context, mask = prepare_batch(
        batch,
        stats["history"][0],
        stats["history"][1],
        stats["context"][0],
        stats["context"][1],
        device,
    )
    nominal_state, nominal_transition = prepare_nominal_query(batch, stats, device)
    mean, sigma = wrapper(
        history,
        action,
        context,
        mask,
        nominal_state,
        nominal_transition,
    )
    target = batch["residual_target"].to(device)
    target = (target - stats["residual"][0]) / stats["residual"][1]
    return mean, sigma, target, (
        history,
        action,
        context,
        mask,
        nominal_state,
        nominal_transition,
    )


def collect_predictions(wrapper, loader, stats, device, compare_mean=False):
    errors = []
    sigmas = []
    maximum_mean_difference = 0.0
    wrapper.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="collect", leave=False):
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
            errors.append((target - mean).cpu())
            sigmas.append(sigma.cpu())
    return (
        torch.cat(errors, dim=0),
        torch.cat(sigmas, dim=0),
        maximum_mean_difference,
    )


def initialize_sigma(wrapper, loader, stats, device):
    errors, _, _ = collect_predictions(wrapper, loader, stats, device)
    sigma = torch.sqrt(torch.mean(errors.square(), dim=(0, 1))).clamp_min(
        wrapper.sigma_floor * 1.01
    )
    wrapper.initialize_constant_sigma(sigma)
    return sigma


def run_epoch(wrapper, loader, stats, device, optimizer=None):
    train = optimizer is not None
    wrapper.train(train)
    total = 0.0
    examples = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(
            loader, desc="sigma train" if train else "sigma val", leave=False
        ):
            mean, sigma, target, _ = forward_batch(wrapper, batch, stats, device)
            loss = gaussian_nll(
                target - mean,
                sigma,
                weights=LOSS_WEIGHTS.to(device),
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count = target.shape[0]
            total += loss.detach().item() * count
            examples += count
    return total / max(examples, 1)


def safe_correlation(x, y, method):
    x_scale = max(float(np.max(np.abs(x))), 1.0)
    y_scale = max(float(np.max(np.abs(y))), 1.0)
    if (
        float(np.ptp(x)) <= np.finfo(x.dtype).eps * x_scale
        or float(np.ptp(y)) <= np.finfo(y.dtype).eps * y_scale
    ):
        return None
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def distribution_metrics(errors, sigmas, residual_std):
    error = errors.numpy()
    sigma = sigmas.numpy()
    physical_std = residual_std.cpu().numpy().reshape(1, 1, -1)
    result = {"channels": {}, "coverage": {}, "mean": {}}
    nll_values = 0.5 * (
        np.square(error / sigma)
        + 2.0 * np.log(sigma)
        + math.log(2.0 * math.pi)
    )
    standardized_rms = np.sqrt(np.mean(np.square(error / sigma), axis=(0, 1)))
    for channel_index, channel in enumerate(CHANNELS):
        abs_error = np.abs(error[..., channel_index]).reshape(-1)
        channel_sigma = sigma[..., channel_index].reshape(-1)
        result["channels"][channel] = {
            "nll": float(np.mean(nll_values[..., channel_index])),
            "standardized_residual_rms": float(
                standardized_rms[channel_index]
            ),
            "sigma_physical_mean": float(
                np.mean(channel_sigma * physical_std[..., channel_index])
            ),
            "pearson_sigma_abs_error": safe_correlation(
                channel_sigma, abs_error, "pearson"
            ),
            "spearman_sigma_abs_error": safe_correlation(
                channel_sigma, abs_error, "spearman"
            ),
        }
    result["mean"]["nll"] = float(np.mean(nll_values))
    correlations = [
        result["channels"][channel]["spearman_sigma_abs_error"]
        for channel in CHANNELS
        if result["channels"][channel]["spearman_sigma_abs_error"] is not None
    ]
    result["mean"]["spearman_sigma_abs_error"] = (
        float(np.mean(correlations)) if correlations else None
    )
    for label, (z_value, nominal) in COVERAGE_LEVELS.items():
        covered = np.mean(np.abs(error) <= z_value * sigma, axis=(0, 1))
        widths = np.mean(
            2.0 * z_value * sigma * physical_std, axis=(0, 1)
        )
        result["coverage"][label] = {
            "nominal": nominal,
            "empirical_by_channel": {
                channel: float(covered[index])
                for index, channel in enumerate(CHANNELS)
            },
            "empirical_mean": float(np.mean(covered)),
            "absolute_error_mean": float(np.mean(np.abs(covered - nominal))),
            "physical_width_by_channel": {
                channel: float(widths[index])
                for index, channel in enumerate(CHANNELS)
            },
        }
    return result


def horizon_metrics(errors, sigmas, residual_std):
    result = {}
    for horizon in HORIZONS:
        if horizon > errors.shape[1]:
            continue
        result[str(horizon)] = distribution_metrics(
            errors[:, horizon - 1 : horizon],
            sigmas[:, horizon - 1 : horizon],
            residual_std,
        )
    return result


def scale_metrics(errors, sigmas, residual_std):
    return {
        "all_steps": distribution_metrics(errors, sigmas, residual_std),
        "horizons": horizon_metrics(errors, sigmas, residual_std),
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_history(history, output_path):
    training = [entry for entry in history if "train_nll" in entry]
    validation = [entry for entry in history if "val_nll" in entry]
    plt.figure(figsize=(8, 5))
    plt.plot(
        [entry["epoch"] for entry in training],
        [entry["train_nll"] for entry in training],
        label="train",
    )
    plt.plot(
        [entry["epoch"] for entry in validation],
        [entry["val_nll"] for entry in validation],
        marker="o",
        label="validation",
    )
    plt.xlabel("epoch")
    plt.ylabel("weighted normalized Gaussian NLL")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    if args.epochs == 0 and args.early_stopping_patience <= 0:
        raise ValueError("Unlimited epochs require positive early-stop patience")
    if not 0.0 < args.lr_decay <= 1.0:
        raise ValueError("--lr-decay must be within (0, 1]")
    if args.val_every <= 0:
        raise ValueError("--val-every must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    splits = fixed_splits(args)
    for name, files in splits.items():
        (output_dir / f"{name}_files.txt").write_text("\n".join(files) + "\n")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["variant"] != "query":
        raise ValueError("Checkpoint must be a Query residual model")
    if checkpoint["protocol"] != "consistent_v2":
        raise ValueError("Checkpoint must use consistent_v2")
    model_args = SimpleNamespace(**checkpoint["args"])
    model = make_model(model_args, device, "query")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stats = checkpoint_stats(checkpoint, device)
    params = KinematicBicycleParams(**checkpoint["params"])

    datasets = {}
    loaders = {}
    for name, files in splits.items():
        shuffle = name == "sigma_train"
        batch_size = args.batch_size if shuffle else args.eval_batch_size
        datasets[name], loaders[name] = make_loader(
            files,
            model_args,
            params,
            batch_size,
            args.num_workers if shuffle else 0,
            shuffle,
        )

    wrapper = FrozenMeanGaussianResidual(
        model, output_dim=4, sigma_floor=args.sigma_floor
    ).to(device)
    initial_sigma = initialize_sigma(
        wrapper, loaders["sigma_train"], stats, device
    )
    constant_train_sigma = initial_sigma.cpu().view(1, 1, -1)

    initial_val_nll = run_epoch(
        wrapper, loaders["sigma_val"], stats, device, optimizer=None
    )
    optimizer = torch.optim.AdamW(
        wrapper.sigma_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=args.lr_decay
    )
    best_path = output_dir / "probabilistic_query_best.pt"
    best_val = initial_val_nll
    best_epoch = 0
    checks_without_improvement = 0
    early_stopped = False
    torch.save(
        {
            "sigma_head_state_dict": wrapper.sigma_head.state_dict(),
            "initial_sigma": initial_sigma.cpu(),
            "epoch": 0,
            "val_nll": initial_val_nll,
        },
        best_path,
    )
    history = [{"epoch": 0, "val_nll": initial_val_nll, "learning_rate": args.lr}]
    epoch = 0
    while args.epochs == 0 or epoch < args.epochs:
        epoch += 1
        learning_rate = optimizer.param_groups[0]["lr"]
        train_nll = run_epoch(
            wrapper, loaders["sigma_train"], stats, device, optimizer
        )
        entry = {
            "epoch": epoch,
            "train_nll": train_nll,
            "learning_rate": learning_rate,
        }
        should_validate = epoch % args.val_every == 0 or (
            args.epochs > 0 and epoch == args.epochs
        )
        if should_validate:
            val_nll = run_epoch(
                wrapper, loaders["sigma_val"], stats, device, optimizer=None
            )
            entry["val_nll"] = val_nll
            print(
                f"epoch={epoch} train_nll={train_nll:.7f} "
                f"val_nll={val_nll:.7f} lr={learning_rate:.8f}"
            )
            if val_nll < best_val - args.early_stopping_min_delta:
                best_val = val_nll
                best_epoch = epoch
                checks_without_improvement = 0
                torch.save(
                    {
                        "sigma_head_state_dict": wrapper.sigma_head.state_dict(),
                        "initial_sigma": initial_sigma.cpu(),
                        "epoch": epoch,
                        "val_nll": val_nll,
                    },
                    best_path,
                )
            else:
                checks_without_improvement += 1
                if (
                    args.early_stopping_patience > 0
                    and checks_without_improvement
                    >= args.early_stopping_patience
                ):
                    early_stopped = True
                    history.append(entry)
                    break
        history.append(entry)
        scheduler.step()

    best = torch.load(best_path, map_location=device, weights_only=False)
    wrapper.sigma_head.load_state_dict(best["sigma_head_state_dict"])
    calibration_error, calibration_sigma, _ = collect_predictions(
        wrapper, loaders["calibration"], stats, device
    )
    learned_temperature = channel_temperature(
        calibration_error, calibration_sigma
    )
    constant_calibration_sigma = constant_train_sigma.expand_as(calibration_error)
    constant_temperature = channel_temperature(
        calibration_error, constant_calibration_sigma
    )

    test_error, learned_test_sigma, maximum_mean_difference = collect_predictions(
        wrapper,
        loaders["probability_test"],
        stats,
        device,
        compare_mean=True,
    )
    constant_test_sigma = constant_train_sigma.expand_as(test_error)
    learned_calibrated_sigma = learned_test_sigma * learned_temperature.view(
        1, 1, -1
    )
    constant_calibrated_sigma = constant_test_sigma * constant_temperature.view(
        1, 1, -1
    )
    residual_std = stats["residual"][1].cpu()
    metrics = {
        "constant_raw": scale_metrics(
            test_error, constant_test_sigma, residual_std
        ),
        "constant_calibrated": scale_metrics(
            test_error, constant_calibrated_sigma, residual_std
        ),
        "learned_raw": scale_metrics(
            test_error, learned_test_sigma, residual_std
        ),
        "learned_calibrated": scale_metrics(
            test_error, learned_calibrated_sigma, residual_std
        ),
    }
    physical_error = test_error * residual_std.view(1, 1, -1)
    mean_transition_rmse = torch.sqrt(
        torch.mean(physical_error.square(), dim=(0, 1))
    )

    plot_history(history, output_dir / "training_curve.png")
    summary = {
        "protocol": "frozen_mean_nuplan_probabilistic_query_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_run": str(Path(args.source_run).resolve()),
        "args": vars(args),
        "split_file_counts": {
            name: len(files) for name, files in splits.items()
        },
        "split_episode_counts": {
            name: len(dataset) for name, dataset in datasets.items()
        },
        "split_intersections": {
            f"{left}/{right}": len(set(splits[left]) & set(splits[right]))
            for index, left in enumerate(splits)
            for right in list(splits)[index + 1 :]
        },
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in wrapper.parameters()
            if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in wrapper.parameters()
            if not parameter.requires_grad
        ),
        "initial_sigma_normalized": initial_sigma.cpu().tolist(),
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "early_stopped": early_stopped,
        "best_val_nll": best_val,
        "history": history,
        "calibration": {
            "constant_temperature": constant_temperature.tolist(),
            "learned_temperature": learned_temperature.tolist(),
        },
        "mean_invariance": {
            "max_abs_normalized_difference": maximum_mean_difference,
            "transition_rmse_physical": {
                channel: float(mean_transition_rmse[index])
                for index, channel in enumerate(CHANNELS)
            },
        },
        "test_metrics": metrics,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "best_epoch": best_epoch,
            "stopped_epoch": epoch,
            "best_val_nll": best_val,
            "max_mean_difference": maximum_mean_difference,
            "constant_calibrated": metrics["constant_calibrated"]["all_steps"],
            "learned_calibrated": metrics["learned_calibrated"]["all_steps"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
