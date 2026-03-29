#!/usr/bin/env python3
"""
Investigate ORT Transformer Optimizer for Qwen3-ASR decoder models.

Tests whether the ORT optimizer's fusion passes (attention, layer norm, etc.)
produce models compatible with the ORT binary and whether they improve performance.

Run in isolation (may OOM on large models):
    nohup uv run python optimize_decoder.py \
        --input output/qwen3-asr-0.6b/decoder_step.onnx \
        --output output/qwen3-asr-0.6b/decoder_step.optimized.onnx \
        > optimize.log 2>&1 &

Or with selective fusions:
    python optimize_decoder.py \
        --input output/qwen3-asr-0.6b/decoder_step.onnx \
        --output output/qwen3-asr-0.6b/decoder_step.optimized.onnx \
        --disable-fusions SimplifiedLayerNormalization
"""

import argparse
import os
import sys
import time
from collections import Counter

import onnx


def count_ops(model):
    """Count operator types in a model."""
    ops = Counter()
    for node in model.graph.node:
        ops[node.op_type] += 1
    return ops


def report_changes(original_ops, optimized_ops):
    """Report operator count changes between original and optimized models."""
    all_ops = sorted(set(original_ops) | set(optimized_ops))
    changes = []
    for op in all_ops:
        orig = original_ops.get(op, 0)
        opt = optimized_ops.get(op, 0)
        if orig != opt:
            changes.append((op, orig, opt))

    if not changes:
        print("  No operator changes")
        return

    print(f"  Operator changes ({len(changes)} types):")
    for op, orig, opt in changes:
        delta = opt - orig
        sign = "+" if delta > 0 else ""
        print(f"    {op}: {orig} -> {opt} ({sign}{delta})")

    orig_total = sum(original_ops.values())
    opt_total = sum(optimized_ops.values())
    print(f"  Total nodes: {orig_total} -> {opt_total}")


def check_contrib_ops(model):
    """Identify contrib/non-standard ops that may not be available in all ORT builds."""
    # Known contrib ops that may cause issues
    contrib_ops = set()
    for node in model.graph.node:
        if node.domain and node.domain != "":
            contrib_ops.add(f"{node.domain}::{node.op_type}")
        # SimplifiedLayerNormalization is a contrib op even without explicit domain
        if node.op_type in (
            "SimplifiedLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "FusedMatMul",
            "Attention",
            "MultiHeadAttention",
            "GroupQueryAttention",
            "RotaryEmbedding",
        ):
            contrib_ops.add(node.op_type)
    return contrib_ops


def optimize_model(input_path, output_path, num_heads=0, hidden_size=0, disable_fusions=None, model_type="bert"):
    """Run ORT Transformer Optimizer on a model."""
    from onnxruntime.transformers import optimizer as ort_optimizer
    from onnxruntime.transformers.fusion_options import FusionOptions

    print(f"Loading model: {input_path}")
    original = onnx.load(input_path, load_external_data=True)
    original_ops = count_ops(original)
    print(f"  Original: {sum(original_ops.values())} nodes, {len(original_ops)} op types")

    # Configure fusion options
    options = FusionOptions(model_type)
    if disable_fusions:
        for fusion in disable_fusions:
            attr = f"enable_{fusion.lower()}"
            # Try common attribute names
            for candidate in [
                attr,
                f"enable_{fusion}",
                fusion.lower(),
            ]:
                if hasattr(options, candidate):
                    setattr(options, candidate, False)
                    print(f"  Disabled fusion: {candidate}")
                    break
            else:
                print(f"  Warning: could not find fusion option for '{fusion}'")

    print(f"Running optimizer (model_type={model_type}, num_heads={num_heads}, hidden_size={hidden_size})...")
    t0 = time.time()

    try:
        optimized = ort_optimizer.optimize_model(
            input_path,
            model_type=model_type,
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=options,
        )
    except Exception as e:
        print(f"  Optimizer failed: {e}")
        return None

    elapsed = time.time() - t0
    print(f"  Optimization took {elapsed:.1f}s")

    opt_model = optimized.model
    optimized_ops = count_ops(opt_model)
    print(f"  Optimized: {sum(optimized_ops.values())} nodes, {len(optimized_ops)} op types")

    report_changes(original_ops, optimized_ops)

    # Check for contrib ops
    contrib = check_contrib_ops(opt_model)
    if contrib:
        print("\n  Contrib/non-standard ops in optimized model:")
        for op in sorted(contrib):
            print(f"    {op}")

    # Save with external data format
    print(f"\n  Saving to {output_path} (external data format)...")
    output_dir = os.path.dirname(output_path)
    data_name = os.path.basename(output_path) + ".data"

    onnx.external_data_helper.convert_model_to_external_data(
        opt_model,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save(opt_model, output_path)

    proto_size = os.path.getsize(output_path)
    data_path = os.path.join(output_dir, data_name)
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    print(f"  Proto: {proto_size / 1e6:.1f} MB, Data: {data_size / 1e9:.2f} GB")

    return opt_model


def verify_with_ort(model_path):
    """Try loading the optimized model in ORT."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    print(f"\n  ORT load test: {os.path.basename(model_path)}")
    try:
        sess = ort.InferenceSession(model_path, opts)
        print("    Loaded successfully")
        print(f"    Inputs:  {[i.name for i in sess.get_inputs()]}")
        print(f"    Outputs: {[o.name for o in sess.get_outputs()]}")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="ORT Transformer Optimizer investigation for Qwen3-ASR")
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output optimized model path")
    parser.add_argument(
        "--num-heads",
        type=int,
        default=0,
        help="Number of attention heads (0 = auto-detect)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=0,
        help="Hidden size (0 = auto-detect)",
    )
    parser.add_argument(
        "--model-type",
        default="bert",
        help="Model type for optimizer (bert, gpt2, etc.)",
    )
    parser.add_argument(
        "--disable-fusions",
        nargs="*",
        default=None,
        help="Fusion types to disable (e.g., SimplifiedLayerNormalization)",
    )
    parser.add_argument(
        "--skip-ort-test",
        action="store_true",
        help="Skip ORT load verification",
    )
    args = parser.parse_args()

    opt_model = optimize_model(
        args.input,
        args.output,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        disable_fusions=args.disable_fusions,
        model_type=args.model_type,
    )

    if opt_model is None:
        print("\nOptimization failed.")
        sys.exit(1)

    if not args.skip_ort_test:
        verify_with_ort(args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
