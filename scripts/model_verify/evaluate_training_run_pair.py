#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
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

from car_foundation.dataset import MujocoDataset
from car_foundation.models import (
    TorchTransformerDecoder,
    TorchTransformerDecoderCurrentState,
    TorchTransformerDecoderCurrentStateMLP,
)


STATE_WEIGHTS = torch.tensor([0.5, 0.5, 2.0, 0.5, 0.0, 2.5], dtype=torch.float32)
STATE_NAMES = ("x", "y", "yaw", "vx", "vy", "yawrate")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two AnyCar torch training runs on the same test split."
    )
    parser.add_argument(
        "--baseline-summary",
        default="/home/gzh/anycar/outputs/checkpoints/2026-07-02T18:55:52.344-model_checkpoint/run_summary.json",
    )
    parser.add_argument(
        "--current-summary",
        default="/home/gzh/anycar/outputs/checkpoints/2026-07-06T10:05:09.092-model_checkpoint/run_summary.json",
    )
    parser.add_argument("--checkpoint-choice", choices=("best", "latest"), default="best")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all test episodes.")
    parser.add_argument("--num-plot-episodes", type=int, default=12)
    parser.add_argument("--profile-warmup", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=100)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "training_run_comparison"),
    )
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def read_split(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def assert_same_splits(baseline, current):
    result = {}
    for split in ("train", "val", "test"):
        b_files = read_split(baseline["dataset"]["split_files"][split])
        c_files = read_split(current["dataset"]["split_files"][split])
        same = b_files == c_files
        result[split] = {
            "same": same,
            "baseline_count": len(b_files),
            "current_count": len(c_files),
        }
        if not same:
            raise ValueError(f"{split}_files are not identical; refusing direct comparison.")
    return result


def selected_checkpoint(summary, choice):
    key = "best_val_checkpoint" if choice == "best" else "latest_checkpoint"
    path = summary.get(key, "")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"{key} does not exist: {path}")
    return path


def make_model(summary, checkpoint_path, device):
    config = summary["config"]
    variant = config.get("model_variant", "baseline")
    kwargs = dict(
        state_dim=config.get("state_dim", 6),
        action_dim=config.get("action_dim", 2),
        output_dim=config.get("state_dim", 6),
        latent_dim=config.get("latent_dim", 256),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 3),
        device=device,
        dropout=config.get("dropout", 0.1),
        history_length=config.get("history_length", 250),
        prediction_length=config.get("prediction_length", 50),
    )
    if variant == "current_concat_linear":
        model = TorchTransformerDecoderCurrentState(
            **kwargs,
            current_dim=config.get("current_dim", kwargs["state_dim"] + kwargs["action_dim"]),
        )
    elif variant == "current_concat_mlp":
        model = TorchTransformerDecoderCurrentStateMLP(
            **kwargs,
            current_dim=config.get("current_dim", kwargs["state_dim"] + kwargs["action_dim"]),
            fusion_hidden_dim=config.get("fusion_hidden_dim", 128),
        )
    elif variant == "baseline":
        model = TorchTransformerDecoder(**kwargs)
    else:
        raise ValueError(f"Unsupported model_variant: {variant}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def model_forward_independent_history(model, history, action, action_padding_mask=None):
    history_emb = model.position_encoding["history"](model._build_history_emb(history))
    if isinstance(model, TorchTransformerDecoderCurrentState):
        action_emb_raw = model._build_action_emb(history, action)
    else:
        action_emb_raw = model.embedding["action"](action)
    action_emb = model.position_encoding["action"](action_emb_raw)
    out = model.transformer_decoder(
        tgt=action_emb,
        memory=history_emb,
        tgt_mask=model.tgt_mask,
        tgt_key_padding_mask=action_padding_mask.to(model.device)
        if action_padding_mask is not None
        else None,
    )
    return model.embedding["output"](out)


def prepare_batch(history, action, y, mask, mean, std, device):
    history = history.to(device).clone()
    action = action.to(device)
    y = y.to(device)
    mask = mask.to(device) if mask is not None else None
    history[:, :, :6] = (history[:, :, :6] - mean) / std
    y_norm = (y[:, :, :6] - mean) / std
    return history[:, 1:, :].detach(), action.detach(), y_norm.detach(), mask


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def unwrap_yaw_diff(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def update_metric_sums(sums, truth, pred):
    count = truth.shape[0] * truth.shape[1]
    diff = pred - truth
    yaw_diff = unwrap_yaw_diff(pred[:, :, 2], truth[:, :, 2])
    sums["position_sse"] += torch.sum(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2).item()
    sums["velocity_sse"] += torch.sum(diff[:, :, 3] ** 2 + diff[:, :, 4] ** 2).item()
    sums["yaw_sse"] += torch.sum(yaw_diff**2).item()
    sums["yawrate_sse"] += torch.sum(diff[:, :, 5] ** 2).item()
    for idx, name in enumerate(STATE_NAMES):
        if name == "yaw":
            sums[f"{name}_sse"] += torch.sum(yaw_diff**2).item()
        else:
            sums[f"{name}_sse"] += torch.sum(diff[:, :, idx] ** 2).item()
    sums["count"] += count


def metric_summary(sums):
    count = max(sums["count"], 1)
    result = {
        "position_rmse": float(np.sqrt(sums["position_sse"] / count)),
        "velocity_rmse": float(np.sqrt(sums["velocity_sse"] / count)),
        "yaw_rmse": float(np.sqrt(sums["yaw_sse"] / count)),
        "yawrate_rmse": float(np.sqrt(sums["yawrate_sse"] / count)),
    }
    for name in STATE_NAMES:
        result[f"{name}_rmse"] = float(np.sqrt(sums[f"{name}_sse"] / count))
    return result


def rollout_from_delta(pred_delta, last_state):
    last_pose = last_state[:, :6].clone()
    pred_abs = pred_delta.clone()
    for i in range(pred_abs.shape[1]):
        dx = pred_delta[:, i, 0] * torch.cos(last_pose[:, 2]) - pred_delta[
            :, i, 1
        ] * torch.sin(last_pose[:, 2])
        dy = pred_delta[:, i, 0] * torch.sin(last_pose[:, 2]) + pred_delta[
            :, i, 1
        ] * torch.cos(last_pose[:, 2])
        pred_abs[:, i, 0] = last_pose[:, 0] + dx
        pred_abs[:, i, 1] = last_pose[:, 1] + dy
        pred_abs[:, i, 2:6] = last_pose[:, 2:6] + pred_delta[:, i, 2:6]
        last_pose = pred_abs[:, i, :6]
    return pred_abs


def per_episode_rows(start_idx, truth, pred):
    rows = []
    diff = pred - truth
    yaw_diff = torch.atan2(torch.sin(pred[:, :, 2] - truth[:, :, 2]), torch.cos(pred[:, :, 2] - truth[:, :, 2]))
    for batch_idx in range(truth.shape[0]):
        row = {"episode": start_idx + batch_idx}
        d = diff[batch_idx]
        yd = yaw_diff[batch_idx]
        row["position_rmse"] = float(torch.sqrt(torch.mean(d[:, 0] ** 2 + d[:, 1] ** 2)).item())
        row["velocity_rmse"] = float(torch.sqrt(torch.mean(d[:, 3] ** 2 + d[:, 4] ** 2)).item())
        row["yaw_rmse"] = float(torch.sqrt(torch.mean(yd**2)).item())
        row["yawrate_rmse"] = float(torch.sqrt(torch.mean(d[:, 5] ** 2)).item())
        for idx, name in enumerate(STATE_NAMES):
            if name == "yaw":
                value = torch.sqrt(torch.mean(yd**2))
            else:
                value = torch.sqrt(torch.mean(d[:, idx] ** 2))
            row[f"{name}_rmse"] = float(value.item())
        rows.append(row)
    return rows


def evaluate_model(name, model, checkpoint, dataset, args, output_dir):
    device = torch.device(args.device)
    mean = checkpoint["input_mean"].to(device)
    std = checkpoint["input_std"].to(device)
    max_episodes = len(dataset) if args.max_episodes <= 0 else min(args.max_episodes, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(max_episodes))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    loss_sse = 0.0
    loss_count = 0
    metric_sums = {key: 0.0 for key in (
        "position_sse",
        "velocity_sse",
        "yaw_sse",
        "yawrate_sse",
        "x_sse",
        "y_sse",
        "yaw_sse",
        "vx_sse",
        "vy_sse",
        "yawrate_sse",
    )}
    metric_sums["count"] = 0
    rows = []
    plot_cache = {}

    history_length = args.model_history_length
    prediction_length = args.prediction_length
    weights = STATE_WEIGHTS.to(device)[None, None, :]

    model.eval()
    with torch.no_grad():
        seen = 0
        for history, action, y, mask in tqdm(loader, desc=f"eval {name}", leave=False):
            batch_size = history.shape[0]
            x, action, y_norm, mask = prepare_batch(history, action, y, mask, mean, std, device)
            pred_norm = model_forward_independent_history(model, x, action, mask)
            valid = 1.0 if mask is None else (mask == 0)[:, :, None].to(pred_norm.dtype)
            loss_sse += torch.sum(((pred_norm - y_norm) ** 2) * valid * weights).item()
            loss_count += pred_norm.numel()

            pred_delta = pred_norm * std + mean
            episodes = dataset.data[seen : seen + batch_size, :, :6].to(device)
            last_state = episodes[:, history_length, :6]
            truth = episodes[:, history_length + 1 : history_length + 1 + prediction_length, :6]
            pred_abs = rollout_from_delta(pred_delta, last_state)
            update_metric_sums(metric_sums, truth, pred_abs)
            rows.extend(per_episode_rows(seen, truth.cpu(), pred_abs.cpu()))

            if len(plot_cache) < args.num_plot_episodes:
                take = min(args.num_plot_episodes - len(plot_cache), batch_size)
                for i in range(take):
                    plot_cache[seen + i] = {
                        "truth": truth[i].detach().cpu().numpy(),
                        "pred": pred_abs[i].detach().cpu().numpy(),
                        "history": episodes[i, : history_length + 1, :6].detach().cpu().numpy(),
                    }
            seen += batch_size

    metrics_path = os.path.join(output_dir, f"{name}_rollout_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "test_weighted_mse": float(loss_sse / max(loss_count, 1)),
        "rollout": metric_summary(metric_sums),
        "metrics_csv": metrics_path,
        "plot_cache": plot_cache,
        "evaluated_episodes": max_episodes,
    }


def profile_model(model, dataset, checkpoint, args):
    device = torch.device(args.device)
    mean = checkpoint["input_mean"].to(device)
    std = checkpoint["input_std"].to(device)
    count = min(args.batch_size, len(dataset))
    history, action, y, mask = dataset[:count]
    x, action, _, mask = prepare_batch(history, action, y, mask, mean, std, device)
    model.eval()
    with torch.no_grad():
        for _ in range(args.profile_warmup):
            model_forward_independent_history(model, x, action, mask)
        sync(device)
        times = []
        for _ in range(args.profile_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model_forward_independent_history(model, x, action, mask)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return {
        "batch_size": int(x.shape[0]),
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "p99_ms": float(np.percentile(times, 99)),
    }


def plot_comparison(output_dir, baseline_cache, current_cache):
    common = sorted(set(baseline_cache) & set(current_cache))
    plot_paths = []
    for idx in common:
        b = baseline_cache[idx]
        c = current_cache[idx]
        truth = b["truth"]
        history = b["history"]
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        axs[0, 0].plot(history[:, 0], history[:, 1], label="history")
        axs[0, 0].plot(truth[:, 0], truth[:, 1], label="truth", marker="o", markersize=2)
        axs[0, 0].plot(b["pred"][:, 0], b["pred"][:, 1], label="baseline", marker="x", markersize=2)
        axs[0, 0].plot(c["pred"][:, 0], c["pred"][:, 1], label="current", marker="+", markersize=2)
        axs[0, 0].axis("equal")
        axs[0, 0].legend()
        axs[0, 0].set_title("xy")

        horizon = np.arange(truth.shape[0])
        axs[0, 1].plot(horizon, truth[:, 3], label="truth vx")
        axs[0, 1].plot(horizon, b["pred"][:, 3], label="baseline vx")
        axs[0, 1].plot(horizon, c["pred"][:, 3], label="current vx")
        axs[0, 1].legend()
        axs[0, 1].set_title("vx")

        axs[1, 0].plot(horizon, truth[:, 2] * 57.3, label="truth yaw")
        axs[1, 0].plot(horizon, b["pred"][:, 2] * 57.3, label="baseline yaw")
        axs[1, 0].plot(horizon, c["pred"][:, 2] * 57.3, label="current yaw")
        axs[1, 0].legend()
        axs[1, 0].set_title("yaw deg")

        axs[1, 1].plot(horizon, truth[:, 5] * 57.3, label="truth yawrate")
        axs[1, 1].plot(horizon, b["pred"][:, 5] * 57.3, label="baseline yawrate")
        axs[1, 1].plot(horizon, c["pred"][:, 5] * 57.3, label="current yawrate")
        axs[1, 1].legend()
        axs[1, 1].set_title("yawrate deg/s")

        fig.tight_layout()
        path = os.path.join(output_dir, f"rollout_compare_{idx:04d}.png")
        fig.savefig(path)
        plt.close(fig)
        plot_paths.append(path)
    return plot_paths


def without_plot_cache(result):
    result = dict(result)
    result.pop("plot_cache", None)
    return result


def relative_change(new, old):
    return None if old == 0 else (new / old - 1.0)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TorchTransformerDecoder._build_history_emb.")

    baseline_summary = load_json(args.baseline_summary)
    current_summary = load_json(args.current_summary)
    split_check = assert_same_splits(baseline_summary, current_summary)

    baseline_ckpt = selected_checkpoint(baseline_summary, args.checkpoint_choice)
    current_ckpt = selected_checkpoint(current_summary, args.checkpoint_choice)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    baseline_model, baseline_checkpoint = make_model(baseline_summary, baseline_ckpt, device)
    current_model, current_checkpoint = make_model(current_summary, current_ckpt, device)

    test_files = read_split(baseline_summary["dataset"]["split_files"]["test"])
    test_files_path = os.path.join(output_dir, "test_files.txt")
    with open(test_files_path, "w") as f:
        f.write("\n".join(test_files) + "\n")

    config = baseline_summary["config"]
    args.model_history_length = config.get("history_length", 250)
    args.prediction_length = config.get("prediction_length", 50)
    dataset = MujocoDataset(
        test_files,
        args.model_history_length + 1,
        args.prediction_length,
        delays=None,
        mean=baseline_checkpoint["input_mean"].cpu(),
        std=baseline_checkpoint["input_std"].cpu(),
        teacher_forcing=False,
        binary_mask=False,
        attack=False,
        use_zero_point=True,
    )

    start = time.perf_counter()
    baseline_eval = evaluate_model("baseline", baseline_model, baseline_checkpoint, dataset, args, output_dir)
    current_eval = evaluate_model("current", current_model, current_checkpoint, dataset, args, output_dir)
    baseline_latency = profile_model(baseline_model, dataset, baseline_checkpoint, args)
    current_latency = profile_model(current_model, dataset, current_checkpoint, args)
    plot_paths = plot_comparison(output_dir, baseline_eval["plot_cache"], current_eval["plot_cache"])

    comparisons = {
        "test_weighted_mse_relative": relative_change(
            current_eval["test_weighted_mse"], baseline_eval["test_weighted_mse"]
        ),
        "latency_mean_relative": relative_change(
            current_latency["mean_ms"], baseline_latency["mean_ms"]
        ),
        "rollout_relative": {
            key: relative_change(current_eval["rollout"][key], baseline_eval["rollout"][key])
            for key in baseline_eval["rollout"].keys()
        },
    }
    summary = {
        "baseline_summary": args.baseline_summary,
        "current_summary": args.current_summary,
        "checkpoint_choice": args.checkpoint_choice,
        "baseline_checkpoint": baseline_ckpt,
        "current_checkpoint": current_ckpt,
        "output_dir": output_dir,
        "test_files": test_files_path,
        "split_check": split_check,
        "dataset_len": len(dataset),
        "evaluated_episodes": baseline_eval["evaluated_episodes"],
        "elapsed_seconds": time.perf_counter() - start,
        "baseline": without_plot_cache(baseline_eval) | {"latency": baseline_latency},
        "current": without_plot_cache(current_eval) | {"latency": current_latency},
        "comparison": comparisons,
        "plots": plot_paths,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"output_dir: {output_dir}")
    print(f"summary: {summary_path}")
    print(f"dataset_len: {len(dataset)} evaluated={baseline_eval['evaluated_episodes']}")
    print(f"baseline test_weighted_mse={baseline_eval['test_weighted_mse']:.9f}")
    print(f"current  test_weighted_mse={current_eval['test_weighted_mse']:.9f}")
    print(f"relative test_weighted_mse={comparisons['test_weighted_mse_relative'] * 100:.2f}%")
    for key in ("position_rmse", "velocity_rmse", "yaw_rmse", "yawrate_rmse"):
        b = baseline_eval["rollout"][key]
        c = current_eval["rollout"][key]
        rel = comparisons["rollout_relative"][key] * 100
        print(f"{key:<16} baseline={b:.9f} current={c:.9f} rel={rel:.2f}%")
    print(
        "latency_mean_ms "
        f"baseline={baseline_latency['mean_ms']:.3f} "
        f"current={current_latency['mean_ms']:.3f} "
        f"rel={comparisons['latency_mean_relative'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
