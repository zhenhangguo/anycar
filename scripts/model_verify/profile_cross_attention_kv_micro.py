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
        description="Micro-benchmark one decoder layer cross-attention Q/K/V projection cost."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--layer", type=int, default=0, help="Decoder layer index, 0-based.")
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
        help="Repeat the first history across the batch, matching sampled-action inference.",
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


@torch.no_grad()
def build_memory(model, history, share_history):
    source = history[0:1].contiguous() if share_history else history
    memory = model._build_history_emb(source)
    memory = model.position_encoding["history"](memory)
    if share_history and history.shape[0] > 1:
        memory = memory.repeat(history.shape[0], 1, 1).contiguous()
    return memory


@torch.no_grad()
def build_tgt_before_cross_attention(model, action, memory, layer_idx):
    tgt = model.position_encoding["action"](model.embedding["action"](action))
    for i in range(layer_idx):
        tgt = model.transformer_decoder.layers[i](
            tgt,
            memory,
            tgt_mask=model.tgt_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
        )

    layer = model.transformer_decoder.layers[layer_idx]
    if layer.norm_first:
        sa_input = layer.norm1(tgt)
        sa_out = layer.self_attn(
            sa_input,
            sa_input,
            sa_input,
            attn_mask=model.tgt_mask,
            key_padding_mask=None,
            need_weights=False,
        )[0]
        tgt_after_self = tgt + layer.dropout1(sa_out)
        return layer.norm2(tgt_after_self)

    sa_out = layer.self_attn(
        tgt,
        tgt,
        tgt,
        attn_mask=model.tgt_mask,
        key_padding_mask=None,
        need_weights=False,
    )[0]
    return layer.norm1(tgt + layer.dropout1(sa_out))


def split_in_proj(mha):
    d = mha.embed_dim
    weight = mha.in_proj_weight
    bias = mha.in_proj_bias
    return {
        "wq": weight[:d],
        "wk": weight[d : 2 * d],
        "wv": weight[2 * d :],
        "bq": None if bias is None else bias[:d],
        "bk": None if bias is None else bias[d : 2 * d],
        "bv": None if bias is None else bias[2 * d :],
    }


def to_heads(x, num_heads):
    batch_size, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


@torch.no_grad()
def cross_q(mha, parts, tgt):
    return F.linear(tgt, parts["wq"], parts["bq"])


@torch.no_grad()
def cross_k(mha, parts, memory):
    return F.linear(memory, parts["wk"], parts["bk"])


@torch.no_grad()
def cross_v(mha, parts, memory):
    return F.linear(memory, parts["wv"], parts["bv"])


@torch.no_grad()
def cross_kv(mha, parts, memory):
    return (
        F.linear(memory, parts["wk"], parts["bk"]),
        F.linear(memory, parts["wv"], parts["bv"]),
    )


@torch.no_grad()
def cross_attention_only(mha, q, k, v):
    qh = to_heads(q, mha.num_heads)
    kh = to_heads(k, mha.num_heads)
    vh = to_heads(v, mha.num_heads)
    out = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).contiguous().view(q.shape)


@torch.no_grad()
def cross_out_proj(mha, attn_out):
    return F.linear(attn_out, mha.out_proj.weight, mha.out_proj.bias)


@torch.no_grad()
def cross_all_manual(mha, parts, tgt, memory):
    q = cross_q(mha, parts, tgt)
    k, v = cross_kv(mha, parts, memory)
    attn_out = cross_attention_only(mha, q, k, v)
    return cross_out_proj(mha, attn_out)


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
        f"{name:<28} mean={times.mean():8.4f} ms  "
        f"p50={np.percentile(times, 50):8.4f} ms  "
        f"p90={np.percentile(times, 90):8.4f} ms  "
        f"p99={np.percentile(times, 99):8.4f} ms"
    )


@torch.no_grad()
def check_manual_cross_attention(layer, mha, parts, tgt, memory):
    official = layer.multihead_attn(
        tgt,
        memory,
        memory,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
    )[0]
    manual = cross_all_manual(mha, parts, tgt, memory)
    diff = (official - manual).abs()
    print(
        f"manual_cross_check max_abs={diff.max().item():.6e} "
        f"mean_abs={diff.mean().item():.6e}"
    )


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This micro-benchmark uses CUDA events; run with --device cuda.")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_model(args, checkpoint, device)
    if args.layer < 0 or args.layer >= len(model.transformer_decoder.layers):
        raise ValueError(f"--layer must be in [0, {len(model.transformer_decoder.layers) - 1}]")

    history, action, dataset_len, file_count = load_inputs(args, checkpoint, device)
    memory = build_memory(model, history, args.share_history)
    tgt = build_tgt_before_cross_attention(model, action, memory, args.layer)

    layer = model.transformer_decoder.layers[args.layer]
    mha = layer.multihead_attn
    parts = split_in_proj(mha)

    q = cross_q(mha, parts, tgt)
    k, v = cross_kv(mha, parts, memory)
    attn_out = cross_attention_only(mha, q, k, v)
    sync(device)

    print(f"checkpoint: {checkpoint_path}")
    print(f"data_dir:    {args.data_dir}")
    print(f"pkl_files:   {file_count}")
    print(f"dataset_len: {dataset_len}")
    print(f"layer:       {args.layer}")
    print(f"batch_size:  {args.batch_size}")
    print(f"history:     {tuple(history.shape)}")
    print(f"memory:      {tuple(memory.shape)}")
    print(f"tgt_before_cross_attn: {tuple(tgt.shape)}")
    print(f"q/k/v:       {tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}")
    print(f"share_history: {args.share_history}")
    check_manual_cross_attention(layer, mha, parts, tgt, memory)
    print()

    measurements = {
        "cross_q_proj": measure_cuda(lambda: cross_q(mha, parts, tgt), args.warmup, args.iters, device),
        "cross_k_proj": measure_cuda(lambda: cross_k(mha, parts, memory), args.warmup, args.iters, device),
        "cross_v_proj": measure_cuda(lambda: cross_v(mha, parts, memory), args.warmup, args.iters, device),
        "cross_kv_proj": measure_cuda(lambda: cross_kv(mha, parts, memory), args.warmup, args.iters, device),
        "cross_attention": measure_cuda(lambda: cross_attention_only(mha, q, k, v), args.warmup, args.iters, device),
        "cross_out_proj": measure_cuda(lambda: cross_out_proj(mha, attn_out), args.warmup, args.iters, device),
        "cross_all_manual": measure_cuda(lambda: cross_all_manual(mha, parts, tgt, memory), args.warmup, args.iters, device),
        "cross_official_mha": measure_cuda(
            lambda: layer.multihead_attn(
                tgt,
                memory,
                memory,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=False,
            )[0],
            args.warmup,
            args.iters,
            device,
        ),
    }

    for name, times in measurements.items():
        summarize(name, times)

    kv_mean = measurements["cross_kv_proj"].mean()
    official_mean = measurements["cross_official_mha"].mean()
    print()
    print(f"cross_kv_proj / cross_official_mha: {kv_mean / official_mean:.3f}")
    print(f"estimated 3-layer cross_kv_proj: {kv_mean * len(model.transformer_decoder.layers):.4f} ms")


if __name__ == "__main__":
    main()
