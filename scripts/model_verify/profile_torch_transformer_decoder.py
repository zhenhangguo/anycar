#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, REPO_ROOT)

from car_foundation.dataset import MujocoDataset
from car_foundation.models import TorchTransformerDecoder


DEFAULT_CHECKPOINT = (
    "/home/gzh/anycar/outputs/checkpoints/"
    "2026-07-01T11:09:19.608-model_checkpoint/400"
)
DEFAULT_DATA_DIR = "/disk1/collect_data_from_anycar/New_demo/check_data_with_offset"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile TorchTransformerDecoder forward and cacheable history work."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-history-length", type=int, default=251)
    parser.add_argument("--model-history-length", type=int, default=250)
    parser.add_argument("--prediction-length", type=int, default=50)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--compressed-history-length", type=int, default=42)
    parser.add_argument(
        "--share-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repeat the first history across the batch, matching sampled-action inference.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Export a torch.profiler chrome trace for one full forward.",
    )
    parser.add_argument(
        "--trace-dir",
        default=os.path.join(REPO_ROOT, "outputs", "profiles"),
    )
    return parser.parse_args()


def resolve_checkpoint(path):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    dirname = os.path.basename(path.rstrip(os.sep))
    if dirname.isdigit():
        candidate = os.path.join(path, f"torch_model_{dirname}")
        if os.path.isfile(candidate):
            return candidate

    candidates = glob.glob(os.path.join(path, "torch_model_*"))
    if not candidates:
        raise FileNotFoundError(f"No torch_model_* file under {path}")

    def sort_key(candidate):
        match = re.search(r"torch_model_(\d+)$", candidate)
        return int(match.group(1)) if match else -1

    return sorted(candidates, key=sort_key)[-1]


def tile_to_batch(tensor, batch_size):
    if tensor.shape[0] == batch_size:
        return tensor
    repeats = int(np.ceil(batch_size / tensor.shape[0]))
    shape = [repeats] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*shape)[:batch_size]


def load_inputs(args, checkpoint, device):
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.pkl")))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No pkl files found under {args.data_dir}")

    dataset = MujocoDataset(
        files,
        args.dataset_history_length,
        args.prediction_length,
        teacher_forcing=False,
        binary_mask=False,
        use_zero_point=True,
    )
    take = min(args.batch_size, len(dataset))
    history, action, _, _ = dataset[:take]
    history = tile_to_batch(history, args.batch_size).to(device)
    action = tile_to_batch(action, args.batch_size).to(device)

    input_mean = checkpoint.get("input_mean")
    input_std = checkpoint.get("input_std")
    if input_mean is None or input_std is None:
        input_mean = torch.tensor(dataset.mean, dtype=torch.float32)
        input_std = torch.tensor(dataset.std, dtype=torch.float32)
        print("checkpoint has no input_mean/input_std; using dataset statistics")

    input_mean = input_mean.to(device=device, dtype=torch.float32)
    input_std = input_std.to(device=device, dtype=torch.float32)

    history = history.clone()
    history[:, :, : args.state_dim] = (
        history[:, :, : args.state_dim] - input_mean
    ) / input_std
    history = history[:, -args.model_history_length :, :].contiguous()

    if args.share_history:
        history = history[0:1].repeat(args.batch_size, 1, 1).contiguous()

    return history, action.contiguous(), len(dataset), len(files)


def load_model(args, checkpoint, device):
    model = TorchTransformerDecoder(
        args.state_dim,
        args.action_dim,
        args.state_dim,
        args.latent_dim,
        args.num_heads,
        args.num_layers,
        device,
        args.dropout,
        args.model_history_length,
        args.prediction_length,
        args.compressed_history_length,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def encode_history(model, history, share_history):
    source = history[0:1].contiguous() if share_history else history
    memory = model._build_history_emb(source)
    memory = model.position_encoding["history"](memory)
    if share_history and history.shape[0] > 1:
        memory = memory.repeat(history.shape[0], 1, 1).contiguous()
    return memory


@torch.no_grad()
def decode_with_memory(model, memory, action):
    action_emb = model.position_encoding["action"](model.embedding["action"](action))
    out = model.transformer_decoder(
        tgt=action_emb,
        memory=memory,
        tgt_mask=model.tgt_mask,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    )
    return model.embedding["output"](out)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_cuda(fn, warmup, iters, device):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync(device)

        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            sync(device)
            times.append(start.elapsed_time(end))
    return np.asarray(times, dtype=np.float64)


def summarize(name, times):
    print(
        f"{name:<28} mean={times.mean():8.3f} ms  "
        f"p50={np.percentile(times, 50):8.3f} ms  "
        f"p90={np.percentile(times, 90):8.3f} ms  "
        f"p99={np.percentile(times, 99):8.3f} ms"
    )


def export_trace(model, history, action, trace_dir):
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(
        trace_dir, f"torch_transformer_decoder_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with torch.profiler.record_function("model_forward"):
            with torch.no_grad():
                model(history, action)
        sync(history.device)
    prof.export_chrome_trace(trace_path)
    print("\nTop CUDA ops:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(f"\nchrome trace: {trace_path}")


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu only for import checks.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This profiler uses CUDA events; run with --device cuda.")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_model(args, checkpoint, device)
    history, action, dataset_len, file_count = load_inputs(args, checkpoint, device)

    print(f"checkpoint: {checkpoint_path}")
    print(f"data_dir:    {args.data_dir}")
    print(f"pkl_files:   {file_count}")
    print(f"dataset_len: {dataset_len}")
    print(f"batch_size:  {args.batch_size}")
    print(f"history:     {tuple(history.shape)}")
    print(f"action:      {tuple(action.shape)}")
    print(f"share_history: {args.share_history}")
    print()

    cached_memory = encode_history(model, history, args.share_history)
    sync(device)

    full = measure_cuda(lambda: model(history, action), args.warmup, args.iters, device)
    hist = measure_cuda(lambda: encode_history(model, history, args.share_history), args.warmup, args.iters, device)
    dec = measure_cuda(lambda: decode_with_memory(model, cached_memory, action), args.warmup, args.iters, device)

    summarize("full_forward", full)
    summarize("history_encoder", hist)
    summarize("cached_decoder", dec)
    print()
    print(f"history_encoder / full_forward: {hist.mean() / full.mean():.3f}")
    print(f"cached_decoder  / full_forward: {dec.mean() / full.mean():.3f}")

    if args.trace:
        export_trace(model, history, action, args.trace_dir)


if __name__ == "__main__":
    main()
