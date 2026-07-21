#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, os.path.join(REPO_ROOT, package_dir))
sys.path.insert(0, REPO_ROOT)

from profile_decoder_kv_cache_ablation import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    build_cross_kv_cache,
    build_action_emb,
    encode_memory_base,
    from_heads,
    load_inputs,
    load_model,
    repeat_memory_if_needed,
    resolve_checkpoint,
    split_in_proj,
    sync,
    to_heads,
)


class FullForwardWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, history, action):
        return self.model(history, action)


class DecoderWithMemoryWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, memory, action):
        action_emb = self.model.position_encoding["action"](
            self.model.embedding["action"](action)
        )
        out = self.model.transformer_decoder(
            tgt=action_emb,
            memory=memory,
            tgt_mask=self.model.tgt_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
        )
        return self.model.embedding["output"](out)


class CachedKvDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _cross_attention(self, layer, tgt, kh, vh):
        mha = layer.multihead_attn
        parts = split_in_proj(mha)
        q = F.linear(tgt, parts["wq"], parts["bq"])
        qh = to_heads(q, mha.num_heads)
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

    def _feed_forward(self, layer, x):
        return layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))

    def _layer_forward(self, layer, tgt, kh, vh):
        if layer.norm_first:
            raise RuntimeError("This wrapper expects norm_first=False.")

        sa_out = layer.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=self.model.tgt_mask,
            key_padding_mask=None,
            need_weights=False,
        )[0]
        tgt = layer.norm1(tgt + layer.dropout1(sa_out))
        ca_out = self._cross_attention(layer, tgt, kh, vh)
        tgt = layer.norm2(tgt + layer.dropout2(ca_out))
        return layer.norm3(tgt + layer.dropout3(self._feed_forward(layer, tgt)))

    def forward(self, action, k0, v0, k1, v1, k2, v2):
        tgt = build_action_emb(self.model, action)
        caches = ((k0, v0), (k1, v1), (k2, v2))
        for layer, (kh, vh) in zip(self.model.transformer_decoder.layers, caches):
            tgt = self._layer_forward(layer, tgt, kh, vh)
        if self.model.transformer_decoder.norm is not None:
            tgt = self.model.transformer_decoder.norm(tgt)
        return self.model.embedding["output"](tgt)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export/profile ONNX variants of TorchTransformerDecoder."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
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
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--provider",
        choices=("cuda", "tensorrt", "cpu"),
        default="cuda",
        help="ONNX Runtime provider used for timing.",
    )
    parser.add_argument(
        "--share-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one history for the whole action batch.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "outputs", "onnx_profiles"),
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse existing ONNX files in --output-dir.",
    )
    parser.add_argument(
        "--io-binding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Bind inputs/outputs as CUDA OrtValues to avoid repeated CPU/GPU feed copies.",
    )
    return parser.parse_args()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def export_onnx(module, inputs, input_names, output_path, opset):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            tuple(inputs),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes={},
            dynamo=False,
        )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    return output_path


def provider_list(provider):
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider == "tensorrt":
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session(onnx_path, provider, profile_prefix):
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = profile_prefix
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        onnx_path,
        sess_options=options,
        providers=provider_list(provider),
    )
    active_providers = session.get_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            "Requested CUDAExecutionProvider, but ONNX Runtime did not activate it. "
            "Check CUDA/cuDNN runtime libraries in LD_LIBRARY_PATH."
        )
    if provider == "tensorrt" and "TensorrtExecutionProvider" not in active_providers:
        raise RuntimeError(
            "Requested TensorrtExecutionProvider, but ONNX Runtime did not activate it. "
            "Check TensorRT/CUDA/cuDNN runtime libraries in LD_LIBRARY_PATH."
        )
    return session


def make_io_binding(session, feed):
    binding = session.io_binding()
    ort_values = []
    for name, array in feed.items():
        value = ort.OrtValue.ortvalue_from_numpy(array, "cuda", 0)
        ort_values.append(value)
        binding.bind_ortvalue_input(name, value)
    binding.bind_output("output", "cuda", 0)
    return binding, ort_values


def run_once(session, feed, use_io_binding):
    if not use_io_binding:
        return session.run(None, feed)[0]
    binding, ort_values = make_io_binding(session, feed)
    session.run_with_iobinding(binding)
    output = binding.copy_outputs_to_cpu()[0]
    del ort_values
    return output


def run_ort(session, feed, warmup, iters, use_io_binding=False):
    binding = None
    ort_values = None
    if use_io_binding:
        binding, ort_values = make_io_binding(session, feed)

    for _ in range(warmup):
        if use_io_binding:
            session.run_with_iobinding(binding)
        else:
            session.run(None, feed)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        if use_io_binding:
            session.run_with_iobinding(binding)
        else:
            session.run(None, feed)
        times.append((time.perf_counter() - start) * 1000.0)
    del ort_values
    profile_path = session.end_profiling()
    return np.asarray(times, dtype=np.float64), profile_path


def summarize_times(name, times):
    print(
        f"{name:<28} mean={times.mean():8.3f} ms  "
        f"p50={np.percentile(times, 50):8.3f} ms  "
        f"p90={np.percentile(times, 90):8.3f} ms  "
        f"p99={np.percentile(times, 99):8.3f} ms"
    )


def model_run_windows(events, skip_model_runs):
    windows = [
        (event["ts"], event["ts"] + event["dur"])
        for event in events
        if event.get("cat") == "Session"
        and event.get("ph") == "X"
        and event.get("name") == "model_run"
    ]
    windows.sort()
    return windows[skip_model_runs:]


def node_events(profile_path, skip_model_runs=0):
    with open(profile_path) as f:
        events = json.load(f)
    windows = model_run_windows(events, skip_model_runs)

    def in_selected_runs(event):
        if not windows:
            return True
        ts = event.get("ts", 0)
        return any(start <= ts <= end for start, end in windows)

    return [
        e
        for e in events
        if e.get("cat") == "Node" and e.get("ph") == "X" and e.get("dur", 0) > 0
        and in_selected_runs(e)
    ]


def clean_node_name(name):
    return name.removesuffix("_kernel_time")


def categorize_node(event):
    name = clean_node_name(event.get("name", ""))
    op = event.get("args", {}).get("op_name", "")
    low = name.lower()
    op_low = op.lower()

    if "compressor" in low or "history" in low or op_low == "conv":
        return "history/conv_embedding"
    if "embedding/action" in low or "action_embedding" in low:
        return "action_embedding"
    if "self_attn" in low:
        if op_low in ("matmul", "gemm"):
            return "self_attn/projection_or_out"
        if op_low in ("softmax", "fusedmatmul") or "attention" in low:
            return "self_attn/attention"
        return "self_attn/layout_elementwise"
    if "multihead_attn" in low or "cross_attn" in low:
        if op_low in ("matmul", "gemm"):
            return "cross_attn/projection_or_out"
        if op_low in ("softmax", "fusedmatmul") or "attention" in low:
            return "cross_attn/attention"
        return "cross_attn/layout_elementwise"
    if "linear1" in low or "linear2" in low:
        return "ffn"
    if "norm" in low or op_low in ("layernormalization", "simplifiedlayernormalization"):
        return "layernorm"
    if op_low in ("reshape", "transpose", "squeeze", "unsqueeze", "concat", "slice", "gather"):
        return "layout_indexing"
    if op_low in ("add", "mul", "div", "sub", "relu", "where", "cast", "expand", "tile"):
        return "elementwise"
    if "output" in low:
        return "output_projection"
    if op_low in ("matmul", "gemm"):
        return "matmul_gemm_uncategorized"
    return f"other/{op or 'unknown'}"


def summarize_profile(profile_path, skip_model_runs=0, top_n=20):
    with open(profile_path) as f:
        all_events = json.load(f)
    run_count = len(model_run_windows(all_events, 0))
    events = [
        e
        for e in all_events
        if e.get("cat") == "Node" and e.get("ph") == "X" and e.get("dur", 0) > 0
    ]
    if skip_model_runs > 0:
        selected_windows = model_run_windows(all_events, skip_model_runs)
        events = [
            e
            for e in events
            if any(start <= e.get("ts", 0) <= end for start, end in selected_windows)
        ]
    by_op = defaultdict(float)
    by_cat = defaultdict(float)
    by_name = defaultdict(float)
    total = sum(e.get("dur", 0.0) for e in events)
    if total <= 0:
        print(f"profile: {profile_path}")
        print("node_total=0.000 ms  node_events=0")
        return by_cat, by_op, total

    for event in events:
        dur = event.get("dur", 0.0)
        op = event.get("args", {}).get("op_name", "unknown")
        by_op[op] += dur
        by_cat[categorize_node(event)] += dur
        by_name[clean_node_name(event.get("name", ""))] += dur

    print(f"profile: {profile_path}")
    print(
        f"node_total={total / 1000.0:.3f} ms  node_events={len(events)}  "
        f"model_runs={run_count}  skipped_runs={min(skip_model_runs, run_count)}"
    )
    print("\nBy category:")
    for key, dur in sorted(by_cat.items(), key=lambda item: item[1], reverse=True):
        print(f"  {key:<34} {dur / 1000.0:8.3f} ms  {dur / total * 100:6.2f}%")

    print("\nBy op type:")
    for key, dur in sorted(by_op.items(), key=lambda item: item[1], reverse=True)[:top_n]:
        print(f"  {key:<24} {dur / 1000.0:8.3f} ms  {dur / total * 100:6.2f}%")

    print("\nTop nodes:")
    for key, dur in sorted(by_name.items(), key=lambda item: item[1], reverse=True)[:top_n]:
        print(f"  {dur / 1000.0:8.3f} ms  {dur / total * 100:6.2f}%  {key[:140]}")
    return by_cat, by_op, total


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device(args.device)
    if device.type != "cuda" and args.provider != "cpu":
        raise RuntimeError("Use --device cuda for CUDA/TensorRT providers.")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_model(args, checkpoint, device)
    history, action, dataset_len, file_count = load_inputs(args, checkpoint, device)

    memory_base = encode_memory_base(model, history, args.share_history)
    memory = repeat_memory_if_needed(memory_base, action.shape[0], args.share_history)
    kv_cache = build_cross_kv_cache(model, memory_base)
    sync(device)

    full_wrapper = FullForwardWrapper(model).to(device).eval()
    decoder_wrapper = DecoderWithMemoryWrapper(model).to(device).eval()
    cached_wrapper = CachedKvDecoderWrapper(model).to(device).eval()

    with torch.no_grad():
        torch_full = full_wrapper(history, action)
        torch_decoder = decoder_wrapper(memory, action)
        torch_cached = cached_wrapper(
            action,
            kv_cache[0]["kh"],
            kv_cache[0]["vh"],
            kv_cache[1]["kh"],
            kv_cache[1]["vh"],
            kv_cache[2]["kh"],
            kv_cache[2]["vh"],
        )
    sync(device)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    model_tag = f"b{args.batch_size}_{args.provider}_opset{args.opset}"
    output_dir = os.path.join(args.output_dir, f"{timestamp}_{model_tag}")
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "original_full": os.path.join(output_dir, "original_full.onnx"),
        "original_decoder_only": os.path.join(output_dir, "original_decoder_only.onnx"),
        "cached_kv_decoder_only": os.path.join(output_dir, "cached_kv_decoder_only.onnx"),
    }

    if not args.skip_export:
        print("Exporting ONNX models...")
        export_onnx(
            full_wrapper,
            (history, action),
            ("history_input", "prediction_input"),
            paths["original_full"],
            args.opset,
        )
        export_onnx(
            decoder_wrapper,
            (memory, action),
            ("memory", "prediction_input"),
            paths["original_decoder_only"],
            args.opset,
        )
        export_onnx(
            cached_wrapper,
            (
                action,
                kv_cache[0]["kh"],
                kv_cache[0]["vh"],
                kv_cache[1]["kh"],
                kv_cache[1]["vh"],
                kv_cache[2]["kh"],
                kv_cache[2]["vh"],
            ),
            ("prediction_input", "k0", "v0", "k1", "v1", "k2", "v2"),
            paths["cached_kv_decoder_only"],
            args.opset,
        )

    feeds = {
        "original_full": {
            "history_input": to_numpy(history),
            "prediction_input": to_numpy(action),
        },
        "original_decoder_only": {
            "memory": to_numpy(memory),
            "prediction_input": to_numpy(action),
        },
        "cached_kv_decoder_only": {
            "prediction_input": to_numpy(action),
            "k0": to_numpy(kv_cache[0]["kh"]),
            "v0": to_numpy(kv_cache[0]["vh"]),
            "k1": to_numpy(kv_cache[1]["kh"]),
            "v1": to_numpy(kv_cache[1]["vh"]),
            "k2": to_numpy(kv_cache[2]["kh"]),
            "v2": to_numpy(kv_cache[2]["vh"]),
        },
    }

    print(f"checkpoint:    {checkpoint_path}")
    print(f"data_dir:      {args.data_dir}")
    print(f"pkl_files:     {file_count}")
    print(f"dataset_len:   {dataset_len}")
    print(f"batch_size:    {args.batch_size}")
    print(f"provider:      {args.provider}")
    print(f"providers:     {ort.get_available_providers()}")
    print(f"io_binding:    {args.io_binding}")
    print(f"history:       {tuple(history.shape)}")
    print(f"action:        {tuple(action.shape)}")
    print(f"memory:        {tuple(memory.shape)}")
    print(f"kv0:           K{tuple(kv_cache[0]['kh'].shape)} V{tuple(kv_cache[0]['vh'].shape)}")
    print(f"output_dir:    {output_dir}")
    print()

    results = {}
    profiles = {}
    for name, onnx_path in paths.items():
        session = make_session(
            onnx_path,
            args.provider,
            os.path.join(output_dir, f"profile_{name}"),
        )
        print(f"{name}_active_providers: {session.get_providers()}")
        output = run_once(session, feeds[name], args.io_binding)
        ref = {
            "original_full": torch_full,
            "original_decoder_only": torch_decoder,
            "cached_kv_decoder_only": torch_cached,
        }[name]
        diff = np.abs(output - to_numpy(ref))
        print(
            f"{name}_check max_abs={diff.max():.6e} "
            f"mean_abs={diff.mean():.6e}"
        )
        times, profile_path = run_ort(
            session,
            feeds[name],
            args.warmup,
            args.iters,
            use_io_binding=args.io_binding,
        )
        results[name] = times
        profiles[name] = profile_path

    print("\nLatency:")
    for name, times in results.items():
        summarize_times(name, times)

    full = results["original_full"].mean()
    dec = results["original_decoder_only"].mean()
    cached = results["cached_kv_decoder_only"].mean()
    print()
    print(f"history_memory effect original_full -> decoder_only: {(full - dec):.3f} ms ({(full - dec) / full * 100:.2f}%)")
    print(f"cross_kv skip decoder_only -> cached_kv:          {(dec - cached):.3f} ms ({(dec - cached) / dec * 100:.2f}%)")
    print(f"ideal cache original_full -> cached_kv:           {(full - cached):.3f} ms ({(full - cached) / full * 100:.2f}%)")
    print(f"ideal cache speedup:                              {full / cached:.3f}x")

    print("\nONNX Runtime profile summaries:")
    summaries = {}
    for name, profile_path in profiles.items():
        print(f"\n=== {name} ===")
        summaries[name] = summarize_profile(profile_path, skip_model_runs=args.warmup + 1)

    summary_path = os.path.join(output_dir, "summary.json")
    summary = {
        "checkpoint": checkpoint_path,
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "provider": args.provider,
        "io_binding": args.io_binding,
        "available_providers": ort.get_available_providers(),
        "onnx_paths": paths,
        "profile_paths": profiles,
        "latency_ms": {
            name: {
                "mean": float(times.mean()),
                "p50": float(np.percentile(times, 50)),
                "p90": float(np.percentile(times, 90)),
                "p99": float(np.percentile(times, 99)),
            }
            for name, times in results.items()
        },
        "ideal_cache_saving_pct": float((full - cached) / full * 100),
        "decoder_kv_saving_pct": float((dec - cached) / dec * 100),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
