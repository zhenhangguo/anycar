#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import random
import re
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
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, REPO_ROOT)

from car_foundation.dataset import MujocoDataset
from car_foundation.models import LearnedPositionalEncoding


DEFAULT_DATASET_PATH = (
    "/disk1/collect_data_from_anycar/New_demo/new_data_with_x_mean_zero/total_data_1"
)
STATE_WEIGHTS = torch.tensor([0.5, 0.5, 2.0, 0.5, 0.0, 2.5], dtype=torch.float32)
STATE_NAMES = ("x", "y", "yaw", "vx", "vy", "yawrate")


class AblationTransformerDecoder(nn.Module):
    def __init__(
        self,
        variant,
        state_dim,
        action_dim,
        output_dim,
        latent_dim,
        num_heads,
        num_layers,
        dropout,
        history_length,
        prediction_length,
        compressed_history_length,
        current_dim,
        fusion_hidden_dim,
    ):
        super().__init__()
        if variant not in ("baseline", "current_concat_linear", "current_concat_mlp"):
            raise ValueError(f"Unknown variant: {variant}")

        self.variant = variant
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.compressed_history_length = compressed_history_length
        self.current_dim = current_dim
        self.fusion_hidden_dim = fusion_hidden_dim

        self.embedding = nn.ModuleDict(
            {
                "state": nn.Linear(state_dim, latent_dim),
                "action": nn.Linear(action_dim, latent_dim),
                "output": nn.Linear(latent_dim, output_dim),
            }
        )
        self.compressor = nn.ModuleDict(
            {
                "state": nn.Sequential(
                    nn.Conv1d(state_dim, state_dim, kernel_size=5, stride=3, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(state_dim, state_dim, kernel_size=3, stride=2, padding=1),
                ),
                "action": nn.Sequential(
                    nn.Conv1d(action_dim, action_dim, kernel_size=5, stride=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(action_dim, action_dim, kernel_size=3, stride=2, padding=1),
                ),
            }
        )
        self.position_encoding = nn.ModuleDict(
            {
                "history": LearnedPositionalEncoding(
                    latent_dim, compressed_history_length * 2 - 1, flip=True
                ),
                "action": LearnedPositionalEncoding(latent_dim, prediction_length),
            }
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=512,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.register_buffer(
            "tgt_mask",
            nn.Transformer.generate_square_subsequent_mask(prediction_length),
        )

        if variant == "current_concat_linear":
            self.action_fusion = nn.Linear(action_dim + current_dim, latent_dim)
        elif variant == "current_concat_mlp":
            self.action_fusion = nn.Sequential(
                nn.Linear(action_dim + current_dim, fusion_hidden_dim),
                nn.SiLU(),
                nn.Linear(fusion_hidden_dim, latent_dim),
            )
            self.init_fusion_from_action_embedding()
        else:
            self.action_fusion = None

    def _build_history_emb(self, history):
        state = history[..., : self.state_dim].permute(0, 2, 1).contiguous()
        action = history[..., self.state_dim :].permute(0, 2, 1).contiguous()

        state_compressed = self.compressor["state"](state).transpose(1, 2)
        action_compressed = self.compressor["action"](action).transpose(1, 2)
        state_emb = self.embedding["state"](state_compressed)
        action_emb = self.embedding["action"](action_compressed)

        interleaved = torch.stack([state_emb, action_emb], dim=2)
        interleaved = interleaved.view(interleaved.size(0), -1, interleaved.size(-1))
        return interleaved[:, :-1, :]

    def _build_action_emb(self, action, current_state):
        if self.variant == "baseline":
            return self.embedding["action"](action)
        if current_state is None:
            raise ValueError(f"{self.variant} requires current_state.")
        current = current_state[:, None, :].expand(-1, action.shape[1], -1)
        fused = self.action_fusion(torch.cat([action, current], dim=-1))
        if self.variant == "current_concat_mlp":
            return self.embedding["action"](action) + fused
        return fused

    def init_fusion_from_action_embedding(self):
        with torch.no_grad():
            if self.variant == "current_concat_linear":
                self.action_fusion.weight.zero_()
                self.action_fusion.bias.copy_(self.embedding["action"].bias)
                self.action_fusion.weight[:, : self.action_dim].copy_(
                    self.embedding["action"].weight
                )
            elif self.variant == "current_concat_mlp":
                nn.init.zeros_(self.action_fusion[-1].weight)
                nn.init.zeros_(self.action_fusion[-1].bias)

    def forward(
        self,
        history,
        action,
        current_state=None,
        history_padding_mask=None,
        action_padding_mask=None,
    ):
        history_emb = self.position_encoding["history"](self._build_history_emb(history))
        action_emb = self.position_encoding["action"](
            self._build_action_emb(action, current_state)
        )
        out = self.transformer_decoder(
            tgt=action_emb,
            memory=history_emb,
            tgt_mask=self.tgt_mask,
            tgt_key_padding_mask=action_padding_mask,
            memory_key_padding_mask=history_padding_mask,
        )
        return self.embedding["output"](out)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate current-state conditioning ablations."
    )
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--models",
        default="baseline,current_concat_linear,current_concat_mlp",
        help="Comma-separated: baseline,current_concat_linear,current_concat_mlp",
    )
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-history-length", type=int, default=250)
    parser.add_argument("--prediction-length", type=int, default=50)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--compressed-history-length", type=int, default=42)
    parser.add_argument("--current-dim", type=int, default=8)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--max-val-episodes", type=int, default=50)
    parser.add_argument("--num-plot-episodes", type=int, default=5)
    parser.add_argument(
        "--selection-metric",
        choices=("val_loss", "rollout_score"),
        default="val_loss",
        help="Metric used to save best_model.pt.",
    )
    parser.add_argument(
        "--selection-episodes",
        type=int,
        default=50,
        help="Number of validation episodes used for rollout_score selection.",
    )
    parser.add_argument(
        "--rollout-score-weights",
        default="position=1.0,velocity=0.5,yaw=5.0,yawrate=5.0",
        help="Comma-separated rollout score weights.",
    )
    parser.add_argument("--profile-warmup", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=100)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "current_state_ablation"),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(path):
    if not path:
        return ""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    matches = glob.glob(os.path.join(path, "**", "torch_model_*"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"No torch_model_* checkpoint found under {path}")

    def key(item):
        match = re.search(r"torch_model_(\d+)$", item)
        return int(match.group(1)) if match else -1

    return sorted(matches, key=key)[-1]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_file_split(args):
    files = sorted(glob.glob(os.path.join(args.dataset_path, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No pkl files found in {args.dataset_path}")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.max_files > 0:
        files = files[: args.max_files]
    split = max(1, int(len(files) * args.train_ratio))
    if split >= len(files):
        split = len(files) - 1
    if split <= 0:
        raise ValueError("Need at least two pkl files for train/val split.")
    return files[:split], files[split:]


def make_dataset(files, args, mean=None, std=None):
    return MujocoDataset(
        files,
        args.model_history_length + 1,
        args.prediction_length,
        delays=None,
        mean=mean,
        std=std,
        teacher_forcing=False,
        binary_mask=False,
        attack=False,
        use_zero_point=True,
    )


def make_model(args, variant, device, checkpoint_path=""):
    model = AblationTransformerDecoder(
        variant=variant,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        output_dim=args.state_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        history_length=args.model_history_length,
        prediction_length=args.prediction_length,
        compressed_history_length=args.compressed_history_length,
        current_dim=args.current_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
    ).to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"{variant}: loaded {checkpoint_path}; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if variant in ("current_concat_linear", "current_concat_mlp") and any(
            key.startswith("action_fusion.") for key in missing
        ):
            model.init_fusion_from_action_embedding()
    return model


def prepare_batch(history, action, y, mask, mean, std, device):
    history = history.to(device).clone()
    action = action.to(device)
    y = y.to(device)
    mask = mask.to(device) if mask is not None else None
    history[:, :, :6] = (history[:, :, :6] - mean) / std
    y_norm = (y[:, :, :6] - mean) / std
    x = history[:, 1:, :].detach()
    current_state = x[:, -1, :].detach()
    return x, action.detach(), current_state, y_norm.detach(), mask


def weighted_mse_loss(pred, target, mask, device):
    weights = STATE_WEIGHTS.to(device)[None, None, :]
    if mask is None:
        valid = 1.0
    else:
        valid = (mask == 0)[:, :, None].to(pred.dtype)
    return torch.mean(((pred - target) ** 2) * valid * weights)


def run_epoch(model, loader, optimizer, mean, std, device, train):
    model.train(train)
    total_loss = 0.0
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for history, action, y, mask in tqdm(loader, leave=False):
            x, action, current_state, y_norm, mask = prepare_batch(
                history, action, y, mask, mean, std, device
            )
            pred = model(x, action, current_state, action_padding_mask=mask)
            loss = weighted_mse_loss(pred, y_norm, mask, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.detach().item()
            count += 1
    return total_loss / max(count, 1)


def rollout_prediction(model, history, action, last_state, mean, std, device):
    model.eval()
    with torch.no_grad():
        y_placeholder = torch.zeros(
            action.shape[0], action.shape[1], 6, dtype=history.dtype
        )
        x, action, current_state, _, _ = prepare_batch(
            history,
            action,
            y_placeholder,
            torch.zeros(action.shape[0], action.shape[1], dtype=history.dtype),
            mean,
            std,
            device,
        )
        pred_delta = model(x, action, current_state) * std + mean
        last_pose = last_state[:, :6].to(device).clone()
        pred_abs = pred_delta.clone()
        for i in range(pred_abs.shape[1]):
            dx = pred_abs[:, i, 0] * torch.cos(last_pose[:, 2]) - pred_abs[
                :, i, 1
            ] * torch.sin(last_pose[:, 2])
            dy = pred_abs[:, i, 0] * torch.sin(last_pose[:, 2]) + pred_abs[
                :, i, 1
            ] * torch.cos(last_pose[:, 2])
            pred_abs[:, i, 0] = dx
            pred_abs[:, i, 1] = dy
            pred_abs[:, i, :6] += last_pose
            last_pose = pred_abs[:, i, :6]
        return pred_abs.cpu()


def vector_rmse(a0, a1, b0, b1):
    return float(np.sqrt(np.mean((a0 - b0) ** 2 + (a1 - b1) ** 2)))


def scalar_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def yaw_rmse(a, b):
    diff = np.arctan2(np.sin(a - b), np.cos(a - b))
    return float(np.sqrt(np.mean(diff**2)))


def parse_rollout_score_weights(spec):
    weights = {
        "position": 1.0,
        "velocity": 0.5,
        "yaw": 5.0,
        "yawrate": 5.0,
    }
    if not spec:
        return weights
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in weights:
            raise ValueError(f"Unknown rollout score weight: {key}")
        weights[key] = float(value)
    return weights


def rollout_score(summary, weights):
    return (
        weights["position"] * summary["position_rmse"]
        + weights["velocity"] * summary["velocity_rmse"]
        + weights["yaw"] * summary["yaw_rmse"]
        + weights["yawrate"] * summary["yawrate_rmse"]
    )


def collect_rollout_metrics(model, dataset, args, mean, std, device, max_episodes):
    metrics = []
    max_count = min(max_episodes, len(dataset))
    for idx in range(max_count):
        history, action, _, _ = dataset[idx : idx + 1]
        episode = dataset.get_episode(idx).unsqueeze(0)
        last_state = episode[:, args.model_history_length, :6]
        pred = rollout_prediction(model, history, action, last_state, mean, std, device)
        pred_np = pred.numpy()[0]
        episode_np = episode.numpy()[0]
        truth = episode_np[
            args.model_history_length + 1 : args.model_history_length
            + 1
            + args.prediction_length,
            :6,
        ]

        row = {
            "episode": idx,
            "position_rmse": vector_rmse(truth[:, 0], truth[:, 1], pred_np[:, 0], pred_np[:, 1]),
            "velocity_rmse": vector_rmse(truth[:, 3], truth[:, 4], pred_np[:, 3], pred_np[:, 4]),
            "yaw_rmse": yaw_rmse(truth[:, 2], pred_np[:, 2]),
            "yawrate_rmse": scalar_rmse(truth[:, 5], pred_np[:, 5]),
        }
        for state_idx, name in enumerate(STATE_NAMES):
            if name == "yaw":
                row[f"{name}_rmse"] = yaw_rmse(truth[:, state_idx], pred_np[:, state_idx])
            else:
                row[f"{name}_rmse"] = scalar_rmse(truth[:, state_idx], pred_np[:, state_idx])
        metrics.append(row)

    summary = {}
    for key in metrics[0].keys():
        if key == "episode":
            continue
        summary[key] = float(np.mean([row[key] for row in metrics]))
    return metrics, summary


def evaluate_rollout(model, dataset, args, mean, std, device, model_dir):
    metrics, summary = collect_rollout_metrics(
        model,
        dataset,
        args,
        mean,
        std,
        device,
        args.max_val_episodes,
    )
    plot_count = min(args.num_plot_episodes, len(metrics))
    for row in metrics[:plot_count]:
        idx = row["episode"]
        history, action, _, _ = dataset[idx : idx + 1]
        episode = dataset.get_episode(idx).unsqueeze(0)
        last_state = episode[:, args.model_history_length, :6]
        pred = rollout_prediction(model, history, action, last_state, mean, std, device)
        pred_np = pred.numpy()[0]
        episode_np = episode.numpy()[0]
        truth = episode_np[
            args.model_history_length + 1 : args.model_history_length
            + 1
            + args.prediction_length,
            :6,
        ]
        if idx < plot_count:
            plot_rollout(model_dir, idx, episode_np, truth, pred_np, args)
    write_metrics_csv(os.path.join(model_dir, "rollout_metrics.csv"), metrics)
    plot_metric_summary(model_dir, metrics, summary)
    return summary


def plot_rollout(model_dir, idx, episode, truth, pred, args):
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    history_xy = episode[: args.model_history_length + 1, :2]
    axs[0, 0].plot(history_xy[:, 0], history_xy[:, 1], label="history")
    axs[0, 0].plot(truth[:, 0], truth[:, 1], label="truth", marker="o", markersize=3)
    axs[0, 0].plot(pred[:, 0], pred[:, 1], label="pred", marker="x", markersize=3)
    axs[0, 0].set_title("xy")
    axs[0, 0].axis("equal")
    axs[0, 0].legend()

    horizon = np.arange(args.prediction_length)
    axs[0, 1].plot(horizon, truth[:, 3], label="truth vx")
    axs[0, 1].plot(horizon, pred[:, 3], label="pred vx")
    axs[0, 1].plot(horizon, truth[:, 4], label="truth vy")
    axs[0, 1].plot(horizon, pred[:, 4], label="pred vy")
    axs[0, 1].set_title("velocity")
    axs[0, 1].legend()

    axs[1, 0].plot(horizon, truth[:, 2] * 57.3, label="truth yaw")
    axs[1, 0].plot(horizon, pred[:, 2] * 57.3, label="pred yaw")
    axs[1, 0].set_title("yaw deg")
    axs[1, 0].legend()

    axs[1, 1].plot(horizon, truth[:, 5] * 57.3, label="truth yawrate")
    axs[1, 1].plot(horizon, pred[:, 5] * 57.3, label="pred yawrate")
    axs[1, 1].set_title("yawrate deg/s")
    axs[1, 1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, f"rollout_{idx:03d}.png"))
    plt.close(fig)


def write_metrics_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_summary(model_dir, metrics, summary):
    fig, axs = plt.subplots(2, 1, figsize=(10, 9))
    keys = ("position_rmse", "velocity_rmse", "yaw_rmse", "yawrate_rmse")
    for key in keys:
        axs[0].scatter(range(len(metrics)), [row[key] for row in metrics], label=key)
    axs[0].set_title("rollout RMSE per episode")
    axs[0].legend()

    axs[1].barh(list(keys), [summary[key] for key in keys])
    axs[1].set_title("mean rollout RMSE")
    fig.tight_layout()
    fig.savefig(os.path.join(model_dir, "rmse_summary.png"))
    plt.close(fig)


def profile_forward(model, loader, mean, std, device, warmup, iters):
    model.eval()
    history, action, y, mask = next(iter(loader))
    x, action, current_state, _, mask = prepare_batch(history, action, y, mask, mean, std, device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x, action, current_state, action_padding_mask=mask)
        sync(device)
        times = []
        for _ in range(iters):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(x, action, current_state, action_padding_mask=mask)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
            else:
                start = time.perf_counter()
                model(x, action, current_state, action_padding_mask=mask)
                times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "p99_ms": float(np.percentile(times, 99)),
        "batch_size": int(x.shape[0]),
    }


def plot_loss_curves(output_dir, histories):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, history in histories.items():
        ax.plot(history["train_loss"], label=f"{name} train")
        ax.plot(history["val_loss"], label=f"{name} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"))
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_checkpoint(args.resume_checkpoint)
    train_files, val_files = load_file_split(args)
    print(f"dataset_path: {args.dataset_path}")
    print(f"train_files:  {len(train_files)}")
    print(f"val_files:    {len(val_files)}")
    print(f"output_dir:   {output_dir}")
    print(f"device:       {device}")
    if checkpoint_path:
        print(f"checkpoint:   {checkpoint_path}")

    train_dataset = make_dataset(train_files, args)
    val_dataset = make_dataset(val_files, args, mean=train_dataset.mean, std=train_dataset.std)
    mean = train_dataset.mean.to(device)
    std = train_dataset.std.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    config = vars(args).copy()
    config.update(
        {
            "output_dir": output_dir,
            "train_files": train_files,
            "val_files": val_files,
            "checkpoint_path": checkpoint_path,
            "train_dataset_len": len(train_dataset),
            "val_dataset_len": len(val_dataset),
            "input_mean": train_dataset.mean.tolist(),
            "input_std": train_dataset.std.tolist(),
        }
    )
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    score_weights = parse_rollout_score_weights(args.rollout_score_weights)
    summaries = {}
    histories = {}
    variants = [item.strip() for item in args.models.split(",") if item.strip()]
    for variant in variants:
        print(f"\n=== {variant} ===")
        set_seed(args.seed)
        model_dir = os.path.join(output_dir, variant)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model = make_model(args, variant, device, checkpoint_path)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        history = {"train_loss": [], "val_loss": [], "selection_score": []}
        best_selection = float("inf")
        best_val_loss = float("inf")
        best_epoch = 0
        selected_val_loss = None
        best_rollout_selection = None
        best_path = os.path.join(model_dir, "best_model.pt")
        for epoch in range(args.epochs):
            train_loss = run_epoch(model, train_loader, optimizer, mean, std, device, train=True)
            val_loss = run_epoch(model, val_loader, None, mean, std, device, train=False)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            best_val_loss = min(best_val_loss, val_loss)

            rollout_selection = None
            if args.selection_metric == "rollout_score":
                _, rollout_selection = collect_rollout_metrics(
                    model,
                    val_dataset,
                    args,
                    mean,
                    std,
                    device,
                    args.selection_episodes,
                )
                selection_value = rollout_score(rollout_selection, score_weights)
            else:
                selection_value = val_loss
            history["selection_score"].append(selection_value)

            print(
                f"{variant} epoch={epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"{args.selection_metric}={selection_value:.6f}"
            )
            if selection_value < best_selection:
                best_selection = selection_value
                best_epoch = epoch + 1
                selected_val_loss = val_loss
                best_rollout_selection = rollout_selection
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "input_mean": mean.detach().cpu(),
                        "input_std": std.detach().cpu(),
                        "variant": variant,
                        "epoch": epoch + 1,
                        "selection_metric": args.selection_metric,
                        "selection_score": selection_value,
                        "val_loss": val_loss,
                        "rollout_selection": rollout_selection,
                    },
                    best_path,
                )

        model.load_state_dict(torch.load(best_path, map_location=device)["model_state_dict"])
        rollout_summary = evaluate_rollout(model, val_dataset, args, mean, std, device, model_dir)
        latency = profile_forward(
            model,
            val_loader,
            mean,
            std,
            device,
            args.profile_warmup,
            args.profile_iters,
        )
        summaries[variant] = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "selected_val_loss": selected_val_loss,
            "selection_metric": args.selection_metric,
            "best_selection_score": best_selection,
            "best_rollout_selection": best_rollout_selection,
            "rollout_score_weights": score_weights,
            "rollout": rollout_summary,
            "latency": latency,
            "best_checkpoint": best_path,
        }
        histories[variant] = history
        with open(os.path.join(model_dir, "summary.json"), "w") as f:
            json.dump(summaries[variant], f, indent=2)

    plot_loss_curves(output_dir, histories)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print("\nSummary:")
    for variant, summary in summaries.items():
        rollout = summary["rollout"]
        latency = summary["latency"]
        print(
            f"{variant:<24} best_epoch={summary['best_epoch']:<3} "
            f"selected_val={summary['selected_val_loss']:.6f} "
            f"min_val={summary['best_val_loss']:.6f} "
            f"select={summary['best_selection_score']:.6f} "
            f"pos={rollout['position_rmse']:.4f} "
            f"vel={rollout['velocity_rmse']:.4f} "
            f"yawrate={rollout['yawrate_rmse']:.4f} "
            f"lat={latency['mean_ms']:.3f} ms"
        )
    print(f"\nsummary: {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
