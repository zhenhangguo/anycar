#!/usr/bin/env python3
import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F

from profile_decoder_kv_cache_ablation import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    build_action_emb,
    from_heads,
    load_inputs,
    load_model,
    repeat_memory_if_needed,
    resolve_checkpoint,
    split_in_proj,
    sync,
    to_heads,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export detailed Chrome traces for TorchTransformerDecoder internals."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
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
        help="Use one history for the whole action batch, matching sampled-action inference.",
    )
    parser.add_argument(
        "--mode",
        choices=("official_segmented", "split_projection", "both"),
        default="both",
        help=(
            "official_segmented uses PyTorch MultiheadAttention as a black box; "
            "split_projection separates Q/K/V projections for attribution."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        default=os.path.join(REPO_ROOT, "outputs", "profiles"),
    )
    return parser.parse_args()


@torch.no_grad()
def traced_history_memory(model, history, share_history):
    with torch.profiler.record_function("history/select_shared_history"):
        source = history[0:1].contiguous() if share_history else history

    with torch.profiler.record_function("history/split_state_action"):
        state = source[..., : model.state_dim].permute(0, 2, 1).contiguous()
        action = source[..., model.state_dim :].permute(0, 2, 1).contiguous()

    with torch.profiler.record_function("history/state_conv_compressor"):
        state_compressed = model.compressor["state"](state.cuda())

    with torch.profiler.record_function("history/action_conv_compressor"):
        action_compressed = model.compressor["action"](action.cuda())

    with torch.profiler.record_function("history/state_embedding"):
        state_emb = model.embedding["state"](state_compressed.transpose(1, 2))

    with torch.profiler.record_function("history/action_embedding"):
        action_emb = model.embedding["action"](action_compressed.transpose(1, 2))

    with torch.profiler.record_function("history/interleave_tokens"):
        interleaved = torch.stack([state_emb, action_emb], dim=2)
        interleaved = interleaved.view(interleaved.size(0), -1, interleaved.size(-1))
        memory = interleaved[:, :-1, :]

    with torch.profiler.record_function("history/positional_encoding"):
        memory = model.position_encoding["history"](memory)

    with torch.profiler.record_function("history/repeat_shared_memory"):
        memory = repeat_memory_if_needed(memory, history.shape[0], share_history)

    return memory


@torch.no_grad()
def traced_action_embedding(model, action):
    with torch.profiler.record_function("action/linear_embedding"):
        action_emb = model.embedding["action"](action)
    with torch.profiler.record_function("action/positional_encoding"):
        return model.position_encoding["action"](action_emb)


@torch.no_grad()
def traced_feed_forward(layer, x, prefix):
    with torch.profiler.record_function(f"{prefix}/ffn_linear1_256_to_512"):
        x1 = layer.linear1(x)
    with torch.profiler.record_function(f"{prefix}/ffn_relu"):
        x1 = layer.activation(x1)
    with torch.profiler.record_function(f"{prefix}/ffn_dropout"):
        x1 = layer.dropout(x1)
    with torch.profiler.record_function(f"{prefix}/ffn_linear2_512_to_256"):
        return layer.linear2(x1)


@torch.no_grad()
def official_segmented_layer(layer, tgt, memory, tgt_mask, layer_idx):
    prefix = f"layer{layer_idx}"
    if layer.norm_first:
        raise NotImplementedError("This trace script currently expects norm_first=False.")

    with torch.profiler.record_function(f"{prefix}/self_attn_mha_official"):
        sa_out = layer.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=None,
            need_weights=False,
        )[0]
    with torch.profiler.record_function(f"{prefix}/residual_add_norm1"):
        tgt = layer.norm1(tgt + layer.dropout1(sa_out))

    with torch.profiler.record_function(f"{prefix}/cross_attn_mha_official"):
        ca_out = layer.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
        )[0]
    with torch.profiler.record_function(f"{prefix}/residual_add_norm2"):
        tgt = layer.norm2(tgt + layer.dropout2(ca_out))

    ffn = traced_feed_forward(layer, tgt, prefix)
    with torch.profiler.record_function(f"{prefix}/residual_add_norm3"):
        return layer.norm3(tgt + layer.dropout3(ffn))


@torch.no_grad()
def traced_mha_split(mha, query, key, value, attn_mask, prefix):
    parts = split_in_proj(mha)

    with torch.profiler.record_function(f"{prefix}/q_projection"):
        q = F.linear(query, parts["wq"], parts["bq"])
    with torch.profiler.record_function(f"{prefix}/k_projection"):
        k = F.linear(key, parts["wk"], parts["bk"])
    with torch.profiler.record_function(f"{prefix}/v_projection"):
        v = F.linear(value, parts["wv"], parts["bv"])

    with torch.profiler.record_function(f"{prefix}/reshape_to_heads"):
        qh = to_heads(q, mha.num_heads)
        kh = to_heads(k, mha.num_heads)
        vh = to_heads(v, mha.num_heads)

    with torch.profiler.record_function(f"{prefix}/scaled_dot_product_attention"):
        attn_out = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )

    with torch.profiler.record_function(f"{prefix}/merge_heads"):
        attn_out = from_heads(attn_out)

    with torch.profiler.record_function(f"{prefix}/out_projection"):
        return mha.out_proj(attn_out)


@torch.no_grad()
def split_projection_layer(layer, tgt, memory, tgt_mask, layer_idx):
    prefix = f"layer{layer_idx}"
    if layer.norm_first:
        raise NotImplementedError("This trace script currently expects norm_first=False.")

    sa_out = traced_mha_split(
        layer.self_attn,
        tgt,
        tgt,
        tgt,
        tgt_mask,
        f"{prefix}/self_attn",
    )
    with torch.profiler.record_function(f"{prefix}/residual_add_norm1"):
        tgt = layer.norm1(tgt + layer.dropout1(sa_out))

    ca_out = traced_mha_split(
        layer.multihead_attn,
        tgt,
        memory,
        memory,
        None,
        f"{prefix}/cross_attn",
    )
    with torch.profiler.record_function(f"{prefix}/residual_add_norm2"):
        tgt = layer.norm2(tgt + layer.dropout2(ca_out))

    ffn = traced_feed_forward(layer, tgt, prefix)
    with torch.profiler.record_function(f"{prefix}/residual_add_norm3"):
        return layer.norm3(tgt + layer.dropout3(ffn))


@torch.no_grad()
def traced_forward(model, history, action, share_history, mode):
    with torch.profiler.record_function(f"{mode}/history_memory"):
        memory = traced_history_memory(model, history, share_history)

    with torch.profiler.record_function(f"{mode}/action_embedding"):
        tgt = traced_action_embedding(model, action)

    with torch.profiler.record_function(f"{mode}/decoder_layers"):
        for i, layer in enumerate(model.transformer_decoder.layers):
            with torch.profiler.record_function(f"layer{i}/total"):
                if mode == "official_segmented":
                    tgt = official_segmented_layer(layer, tgt, memory, model.tgt_mask, i)
                elif mode == "split_projection":
                    tgt = split_projection_layer(layer, tgt, memory, model.tgt_mask, i)
                else:
                    raise ValueError(mode)

    if model.transformer_decoder.norm is not None:
        with torch.profiler.record_function(f"{mode}/decoder_final_norm"):
            tgt = model.transformer_decoder.norm(tgt)

    with torch.profiler.record_function(f"{mode}/output_projection_256_to_6"):
        return model.embedding["output"](tgt)


def export_trace(model, history, action, share_history, mode, trace_dir):
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(
        trace_dir,
        f"torch_transformer_decoder_{mode}_"
        f"b{action.shape[0]}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json",
    )

    for _ in range(3):
        traced_forward(model, history, action, share_history, mode)
    sync(history.device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        with torch.profiler.record_function(f"{mode}/model_forward"):
            traced_forward(model, history, action, share_history, mode)
        sync(history.device)

    prof.export_chrome_trace(trace_path)
    print(f"\nmode: {mode}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=80))
    print(f"chrome trace: {trace_path}")
    return trace_path


@torch.no_grad()
def check_outputs(model, history, action, share_history):
    official = model(history, action)
    segmented = traced_forward(model, history, action, share_history, "official_segmented")
    split = traced_forward(model, history, action, share_history, "split_projection")
    sync(history.device)

    for name, candidate in (
        ("official_segmented", segmented),
        ("split_projection", split),
    ):
        diff = (official - candidate).abs()
        print(
            f"{name}_check max_abs={diff.max().item():.6e} "
            f"mean_abs={diff.mean().item():.6e}"
        )


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This trace script uses CUDA profiling; run with --device cuda.")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = load_model(args, checkpoint, device)
    history, action, dataset_len, file_count = load_inputs(args, checkpoint, device)

    print(f"checkpoint:    {checkpoint_path}")
    print(f"data_dir:      {args.data_dir}")
    print(f"pkl_files:     {file_count}")
    print(f"dataset_len:   {dataset_len}")
    print(f"batch_size:    {args.batch_size}")
    print(f"history:       {tuple(history.shape)}")
    print(f"action:        {tuple(action.shape)}")
    print(f"share_history: {args.share_history}")
    print()

    check_outputs(model, history, action, args.share_history)

    modes = (
        ("official_segmented", "split_projection")
        if args.mode == "both"
        else (args.mode,)
    )
    for mode in modes:
        export_trace(model, history, action, args.share_history, mode, args.trace_dir)


if __name__ == "__main__":
    main()
