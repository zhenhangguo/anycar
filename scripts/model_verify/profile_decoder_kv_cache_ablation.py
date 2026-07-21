#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn.functional as F


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
        description=(
            "Compare the original TorchTransformerDecoder forward with a manual "
            "decoder forward that caches cross-attention K/V projections."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
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
        help=(
            "Use one history for the whole action batch. This matches the current "
            "eval forward path and MPPI-style sampled-action inference."
        ),
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


def split_in_proj(mha):
    d_model = mha.embed_dim
    weight = mha.in_proj_weight
    bias = mha.in_proj_bias
    return {
        "wq": weight[:d_model],
        "wk": weight[d_model : 2 * d_model],
        "wv": weight[2 * d_model :],
        "bq": None if bias is None else bias[:d_model],
        "bk": None if bias is None else bias[d_model : 2 * d_model],
        "bv": None if bias is None else bias[2 * d_model :],
    }


def to_heads(x, num_heads):
    batch_size, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def from_heads(x):
    batch_size, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


@torch.no_grad()
def encode_memory_base(model, history, share_history):
    source = history[0:1].contiguous() if share_history else history
    memory = model._build_history_emb(source)
    return model.position_encoding["history"](memory)


def repeat_memory_if_needed(memory_base, batch_size, share_history):
    if share_history and memory_base.shape[0] == 1 and batch_size > 1:
        return memory_base.repeat(batch_size, 1, 1).contiguous()
    return memory_base


@torch.no_grad()
def build_cross_kv_cache(model, memory_base):
    cache = []
    for layer in model.transformer_decoder.layers:
        mha = layer.multihead_attn
        parts = split_in_proj(mha)
        k = F.linear(memory_base, parts["wk"], parts["bk"])
        v = F.linear(memory_base, parts["wv"], parts["bv"])
        cache.append(
            {
                "kh": to_heads(k, mha.num_heads),
                "vh": to_heads(v, mha.num_heads),
            }
        )
    return cache


@torch.no_grad()
def build_action_emb(model, action):
    return model.position_encoding["action"](model.embedding["action"](action))


@torch.no_grad()
def cached_cross_attention(mha, tgt, kv_cache):
    parts = split_in_proj(mha)
    q = F.linear(tgt, parts["wq"], parts["bq"])
    qh = to_heads(q, mha.num_heads)
    kh = kv_cache["kh"]
    vh = kv_cache["vh"]
    if kh.shape[0] == 1 and qh.shape[0] > 1:
        kh = kh.expand(qh.shape[0], -1, -1, -1)
        vh = vh.expand(qh.shape[0], -1, -1, -1)

    attn_out = F.scaled_dot_product_attention(
        qh,
        kh,
        vh,
        dropout_p=0.0,
        is_causal=False,
    )
    return mha.out_proj(from_heads(attn_out))


@torch.no_grad()
def feed_forward(layer, x):
    return layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))


@torch.no_grad()
def cached_decoder_layer(layer, tgt, kv_cache, tgt_mask):
    if layer.norm_first:
        sa_input = layer.norm1(tgt)
        sa_out = layer.self_attn(
            sa_input,
            sa_input,
            sa_input,
            attn_mask=tgt_mask,
            key_padding_mask=None,
            need_weights=False,
        )[0]
        tgt = tgt + layer.dropout1(sa_out)
        tgt = tgt + layer.dropout2(cached_cross_attention(layer.multihead_attn, layer.norm2(tgt), kv_cache))
        tgt = tgt + layer.dropout3(feed_forward(layer, layer.norm3(tgt)))
        return tgt

    sa_out = layer.self_attn(
        tgt,
        tgt,
        tgt,
        attn_mask=tgt_mask,
        key_padding_mask=None,
        need_weights=False,
    )[0]
    tgt = layer.norm1(tgt + layer.dropout1(sa_out))
    ca_out = cached_cross_attention(layer.multihead_attn, tgt, kv_cache)
    tgt = layer.norm2(tgt + layer.dropout2(ca_out))
    return layer.norm3(tgt + layer.dropout3(feed_forward(layer, tgt)))


@torch.no_grad()
def decode_uncached_with_memory(model, memory, action):
    tgt = build_action_emb(model, action)
    out = model.transformer_decoder(
        tgt=tgt,
        memory=memory,
        tgt_mask=model.tgt_mask,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    )
    return model.embedding["output"](out)


@torch.no_grad()
def decode_cached_kv(model, kv_cache, action):
    tgt = build_action_emb(model, action)
    for layer, layer_cache in zip(model.transformer_decoder.layers, kv_cache):
        tgt = cached_decoder_layer(layer, tgt, layer_cache, model.tgt_mask)
    if model.transformer_decoder.norm is not None:
        tgt = model.transformer_decoder.norm(tgt)
    return model.embedding["output"](tgt)


@torch.no_grad()
def uncached_full_forward(model, history, action, share_history):
    memory_base = encode_memory_base(model, history, share_history)
    memory = repeat_memory_if_needed(memory_base, action.shape[0], share_history)
    return decode_uncached_with_memory(model, memory, action)


@torch.no_grad()
def cached_kv_full_forward(model, history, action, share_history):
    memory_base = encode_memory_base(model, history, share_history)
    kv_cache = build_cross_kv_cache(model, memory_base)
    return decode_cached_kv(model, kv_cache, action)


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
        f"{name:<32} mean={times.mean():8.4f} ms  "
        f"p50={np.percentile(times, 50):8.4f} ms  "
        f"p90={np.percentile(times, 90):8.4f} ms  "
        f"p99={np.percentile(times, 99):8.4f} ms"
    )


def print_ratio(name, numerator, denominator):
    print(f"{name:<42} {numerator.mean() / denominator.mean():.3f}")


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This ablation uses CUDA events; run with --device cuda.")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_model(args, checkpoint, device)
    history, action, dataset_len, file_count = load_inputs(args, checkpoint, device)

    memory_base = encode_memory_base(model, history, args.share_history)
    memory = repeat_memory_if_needed(memory_base, action.shape[0], args.share_history)
    kv_cache = build_cross_kv_cache(model, memory_base)
    sync(device)

    ref = decode_uncached_with_memory(model, memory, action)
    cached = decode_cached_kv(model, kv_cache, action)
    diff = (ref - cached).abs()
    sync(device)

    print(f"checkpoint:      {checkpoint_path}")
    print(f"data_dir:        {args.data_dir}")
    print(f"pkl_files:       {file_count}")
    print(f"dataset_len:     {dataset_len}")
    print(f"batch_size:      {args.batch_size}")
    print(f"history:         {tuple(history.shape)}")
    print(f"action:          {tuple(action.shape)}")
    print(f"share_history:   {args.share_history}")
    print(f"memory_base:     {tuple(memory_base.shape)}")
    print(f"memory_reference:{tuple(memory.shape)}")
    print(
        "kv_cache[0]:     "
        f"K{tuple(kv_cache[0]['kh'].shape)} V{tuple(kv_cache[0]['vh'].shape)}"
    )
    print(
        f"output_check:    max_abs={diff.max().item():.6e} "
        f"mean_abs={diff.mean().item():.6e}"
    )
    print()

    measurements = {
        "original_full_forward": measure_cuda(
            lambda: uncached_full_forward(model, history, action, args.share_history),
            args.warmup,
            args.iters,
            device,
        ),
        "original_decoder_only": measure_cuda(
            lambda: decode_uncached_with_memory(model, memory, action),
            args.warmup,
            args.iters,
            device,
        ),
        "prepare_history_memory": measure_cuda(
            lambda: encode_memory_base(model, history, args.share_history),
            args.warmup,
            args.iters,
            device,
        ),
        "prepare_cross_kv_cache": measure_cuda(
            lambda: build_cross_kv_cache(model, memory_base),
            args.warmup,
            args.iters,
            device,
        ),
        "prepare_history_kv_cache": measure_cuda(
            lambda: build_cross_kv_cache(
                model, encode_memory_base(model, history, args.share_history)
            ),
            args.warmup,
            args.iters,
            device,
        ),
        "cached_kv_decoder_only": measure_cuda(
            lambda: decode_cached_kv(model, kv_cache, action),
            args.warmup,
            args.iters,
            device,
        ),
        "cached_kv_full_forward": measure_cuda(
            lambda: cached_kv_full_forward(model, history, action, args.share_history),
            args.warmup,
            args.iters,
            device,
        ),
    }

    for name, times in measurements.items():
        summarize(name, times)

    print()
    print_ratio(
        "cached_kv_decoder_only / original_decoder_only",
        measurements["cached_kv_decoder_only"],
        measurements["original_decoder_only"],
    )
    print_ratio(
        "cached_kv_full_forward / original_full_forward",
        measurements["cached_kv_full_forward"],
        measurements["original_full_forward"],
    )
    print_ratio(
        "prepare_history_kv_cache / original_full_forward",
        measurements["prepare_history_kv_cache"],
        measurements["original_full_forward"],
    )
    decoder_speedup = (
        measurements["original_decoder_only"].mean()
        / measurements["cached_kv_decoder_only"].mean()
    )
    full_speedup = (
        measurements["original_full_forward"].mean()
        / measurements["cached_kv_full_forward"].mean()
    )
    print(f"{'decoder speedup':<42} {decoder_speedup:.3f}x")
    print(f"{'full forward speedup':<42} {full_speedup:.3f}x")


if __name__ == "__main__":
    main()
