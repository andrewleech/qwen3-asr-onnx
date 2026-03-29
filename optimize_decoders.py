"""
optimize_decoders.py — Apply ORT transformer optimizer to Qwen3-ASR decoder graphs.

Fuses decomposed RMSNorm (5-op pattern: ReduceMean -> Pow -> Add -> Sqrt/Reciprocal -> Mul)
into SimplifiedLayerNormalization. This gives ~4% inference speedup with zero WER impact
when applied to FP32 graphs before int4 quantization (see INVESTIGATION.md experiment [113]).

The encoder does NOT benefit from this optimization (experiment [112]).

Usage:
    uv run python optimize_decoders.py --input output/qwen3-asr-0.6b
"""

import argparse
import json
import os

import onnx
from onnxruntime.transformers.optimizer import optimize_model


def load_decoder_config(model_dir: str) -> tuple[int, int]:
    """Read num_attention_heads and hidden_size from config.json decoder section."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    decoder = config["decoder"]
    return decoder["num_attention_heads"], decoder["hidden_size"]


def fix_external_data_refs(onnx_path: str) -> None:
    """Fix external data location references after optimize_model saves.

    The optimizer saves with a temp filename as the external data location.
    This rewrites all external data references to point to the actual .onnx.data file.
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


def optimize_decoder(onnx_path: str, num_heads: int, hidden_size: int) -> None:
    """Run ORT transformer optimizer on a single decoder file."""
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
    model.save_model_to_file(onnx_path, use_external_data_format=True)

    # Fix external data references (optimizer uses temp filename)
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
        description="Apply ORT transformer optimizer to Qwen3-ASR decoder graphs"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Model directory containing decoder_init.onnx and decoder_step.onnx",
    )
    args = parser.parse_args()

    model_dir = args.input
    num_heads, hidden_size = load_decoder_config(model_dir)
    print(f"Config: num_heads={num_heads}, hidden_size={hidden_size}")

    for decoder_name in ["decoder_init", "decoder_step"]:
        onnx_path = os.path.join(model_dir, f"{decoder_name}.onnx")
        if not os.path.exists(onnx_path):
            print(f"\nSkipping {decoder_name}.onnx (not found)")
            continue
        optimize_decoder(onnx_path, num_heads, hidden_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
