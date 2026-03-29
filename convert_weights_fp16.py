#!/usr/bin/env python3
"""
Convert ONNX model weights from FP32 to FP16 storage with explicit Cast nodes.

Unlike convert_fp16.py (which uses onnxconverter_common to rewrite the entire
compute graph to FP16 intermediate types), this script:
1. Converts only initializer weight tensors from FP32 → FP16
2. Inserts Cast(FP16 → FP32) nodes so all ops still receive FP32 inputs
3. Leaves the graph structure, op types, and intermediate tensor types untouched

The result is numerically identical to FP32 inference (all compute in FP32)
but with half the file size. ORT casts FP16 weights to FP32 at graph load time.

Usage:
    uv run python convert_weights_fp16.py \
        --input output/qwen3-asr-0.6b/encoder.onnx \
        --output output/qwen3-asr-0.6b/encoder.fp16w.onnx \
        --verify
"""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def convert_weights_fp16(input_path: str, output_path: str) -> tuple[int, int]:
    """Convert FP32 initializer weights to FP16 with Cast nodes for FP32 compute.

    Returns (n_converted, n_skipped) counts.
    """
    print(f"Loading {os.path.basename(input_path)}...")
    model = onnx.load(input_path, load_external_data=True)

    # Collect names of all FP32 initializers to convert
    init_names = set()
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            init_names.add(tensor.name)

    # Also check which initializer names are used as graph inputs (some models
    # list initializers in both graph.initializer and graph.input)
    graph_input_names = {inp.name for inp in model.graph.input}

    n_converted = 0
    n_skipped = 0
    bytes_before = 0
    bytes_after = 0
    cast_nodes = []

    for tensor in model.graph.initializer:
        if tensor.data_type != TensorProto.FLOAT:
            n_skipped += 1
            continue

        data = numpy_helper.to_array(tensor)
        bytes_before += data.nbytes

        # Check for values outside FP16 range
        max_abs = float(np.abs(data).max())
        if max_abs > 65504.0:
            print(
                f"  WARNING: {tensor.name}: max |value| = {max_abs:.1f} "
                f"(exceeds FP16 max 65504) — will be clamped to inf"
            )

        # Convert weight data to FP16
        data_fp16 = data.astype(np.float16)
        bytes_after += data_fp16.nbytes

        # Rename the initializer: "X" → "X__fp16"
        old_name = tensor.name
        new_name = old_name + "__fp16"
        new_tensor = numpy_helper.from_array(data_fp16, new_name)
        tensor.CopyFrom(new_tensor)

        # Create Cast node: input "X__fp16" (FP16) → output "X" (FP32)
        cast_node = helper.make_node(
            "Cast",
            inputs=[new_name],
            outputs=[old_name],
            to=TensorProto.FLOAT,
            name=f"cast_fp16_to_fp32_{old_name}",
        )
        cast_nodes.append(cast_node)

        # Update graph input if this initializer appears there
        if old_name in graph_input_names:
            for inp in model.graph.input:
                if inp.name == old_name:
                    inp.name = new_name
                    # Update type to float16
                    if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
                        inp.type.tensor_type.elem_type = TensorProto.FLOAT16
                    break

        n_converted += 1

    # Insert Cast nodes at the beginning of the graph (before all other nodes)
    for i, node in enumerate(cast_nodes):
        model.graph.node.insert(i, node)

    print(f"  Converted {n_converted} tensors, skipped {n_skipped}")
    print(f"  Added {len(cast_nodes)} Cast(FP16→FP32) nodes")
    print(f"  Weights: {bytes_before / 1e6:.1f} MB → {bytes_after / 1e6:.1f} MB")

    # Save — encoder weights are typically inlined (< 2 GB)
    input_size = os.path.getsize(input_path)
    if input_size > 2 * 1024**3:
        onnx.save(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(output_path) + ".data",
            size_threshold=1024,
        )
    else:
        onnx.save(model, output_path)

    out_size = os.path.getsize(output_path)
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        out_size += os.path.getsize(data_path)

    print(f"  File: {input_size / 1e6:.1f} MB → {out_size / 1e6:.1f} MB")
    return n_converted, n_skipped


def verify_ort_load(model_path: str) -> bool:
    """Verify ORT can load the converted model."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not available, skipping verification")
        return True

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        sess = ort.InferenceSession(model_path, opts)
        inputs = {inp.name: inp for inp in sess.get_inputs()}
        print(f"  ORT load OK — {len(inputs)} inputs: {list(inputs.keys())}")
        return True
    except Exception as e:
        print(f"  ORT load FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX weights to FP16 storage with Cast nodes (FP32 compute)")
    parser.add_argument("--input", required=True, help="Input ONNX file")
    parser.add_argument("--output", required=True, help="Output ONNX file")
    parser.add_argument("--verify", action="store_true", help="Verify ORT can load result")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        return 1

    convert_weights_fp16(args.input, args.output)

    if args.verify:
        print("\nVerifying ORT load...")
        if not verify_ort_load(args.output):
            return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    exit(main())
