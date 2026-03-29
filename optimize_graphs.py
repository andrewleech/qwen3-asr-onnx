"""
optimize_graphs.py — Apply ORT transformer optimizer to Qwen3-ASR ONNX graphs.

Applies offline graph fusions that ORT's runtime optimizer does not perform:

- Decoders: Fuses decomposed RMSNorm (5-op pattern) into SimplifiedLayerNormalization.
  7% speedup on FP32, 4% on int4 when applied before quantization. (Experiments [111], [113])

- Encoder: Fuses SkipLayerNormalization (skip-connection + LayerNorm) and BiasGelu
  (bias + GELU). ~4% speedup when int4 decoders make the encoder the bottleneck. ([112])

Must be applied to FP32 graphs before int4 quantization so that weights are calibrated
against the fused kernels ([113]).

Usage:
    uv run python optimize_graphs.py --input output/qwen3-asr-0.6b
    uv run python optimize_graphs.py --input output/qwen3-asr-1.7b
"""

import argparse
import json
import os

import onnx
from onnxruntime.transformers.optimizer import optimize_model


def load_config(model_dir: str) -> dict:
    """Read config.json."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)


def fix_external_data_refs(onnx_path: str) -> None:
    """Fix external data location references after optimize_model saves.

    The optimizer saves with a temp filename as the external data location.
    This rewrites all references to point to the actual .onnx.data file.
    """
    basename = os.path.basename(onnx_path)
    expected_data_name = f"{basename}.data"

    model = onnx.load(onnx_path, load_external_data=False)
    fixed = 0
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value != expected_data_name:
                entry.value = expected_data_name
                fixed += 1
    if fixed > 0:
        with open(onnx_path, "wb") as f:
            f.write(model.SerializeToString())
        print(f"  Fixed {fixed} external data references -> {expected_data_name}")


def optimize_graph(onnx_path: str, num_heads: int, hidden_size: int,
                   use_external_data: bool = True) -> None:
    """Run ORT transformer optimizer on a single ONNX file."""
    basename = os.path.basename(onnx_path)
    print(f"\nOptimizing {basename}...")

    # Count nodes before
    model_proto = onnx.load(onnx_path, load_external_data=False)
    nodes_before = len(model_proto.graph.node)
    del model_proto

    # Run optimizer
    model = optimize_model(
        onnx_path,
        model_type="gpt2",
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=1,
    )

    # Report fusions
    fusions = model.get_fused_operator_statistics()
    if fusions:
        print("  Fusions applied:")
        for op_type, count in sorted(fusions.items()):
            if count > 0:
                print(f"    {op_type}: {count}")
    else:
        print("  No fusions applied")

    # Save back (overwrite original)
    model.save_model_to_file(onnx_path, use_external_data_format=use_external_data)

    # Fix external data references if using external data
    if use_external_data:
        fix_external_data_refs(onnx_path)

    # Count nodes after
    model_proto = onnx.load(onnx_path, load_external_data=False)
    nodes_after = len(model_proto.graph.node)
    del model_proto

    reduction = nodes_before - nodes_after
    pct = (reduction / nodes_before * 100) if nodes_before > 0 else 0
    print(f"  Nodes: {nodes_before} -> {nodes_after} ({reduction} removed, {pct:.1f}% reduction)")


def main():
    parser = argparse.ArgumentParser(
        description="Apply ORT transformer optimizer to Qwen3-ASR encoder and decoder graphs"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Model directory containing encoder.onnx, decoder_init.onnx, decoder_step.onnx",
    )
    parser.add_argument(
        "--skip-encoder",
        action="store_true",
        help="Skip encoder optimization",
    )
    parser.add_argument(
        "--skip-decoders",
        action="store_true",
        help="Skip decoder optimization",
    )
    args = parser.parse_args()

    model_dir = args.input
    config = load_config(model_dir)
    decoder_cfg = config["decoder"]
    encoder_cfg = config["encoder"]

    dec_heads = decoder_cfg["num_attention_heads"]
    dec_hidden = decoder_cfg["hidden_size"]
    enc_heads = encoder_cfg["num_heads"]
    enc_hidden = encoder_cfg["hidden_size"]

    print(f"Decoder config: num_heads={dec_heads}, hidden_size={dec_hidden}")
    print(f"Encoder config: num_heads={enc_heads}, hidden_size={enc_hidden}")

    # Optimize encoder (SkipLayerNormalization + BiasGelu fusions)
    if not args.skip_encoder:
        enc_path = os.path.join(model_dir, "encoder.onnx")
        if os.path.exists(enc_path):
            # Encoder weights are inlined (no external data)
            optimize_graph(enc_path, enc_heads, enc_hidden, use_external_data=False)
        else:
            print(f"\nSkipping encoder.onnx (not found)")

    # Optimize decoders (SimplifiedLayerNormalization fusion)
    if not args.skip_decoders:
        for decoder_name in ["decoder_init", "decoder_step"]:
            dec_path = os.path.join(model_dir, f"{decoder_name}.onnx")
            if os.path.exists(dec_path):
                optimize_graph(dec_path, dec_heads, dec_hidden, use_external_data=True)
            else:
                print(f"\nSkipping {decoder_name}.onnx (not found)")

    print("\nDone.")


if __name__ == "__main__":
    main()
