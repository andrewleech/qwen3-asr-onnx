#!/usr/bin/env python3
"""
INT8 dynamic quantization for Qwen3-ASR ONNX models.

Quantizes encoder and decoder ONNX files using onnxruntime dynamic quantization.
Embedding table is part of the ONNX graph (Gather op) and is not affected by
dynamic quantization, which only targets MatMul ops.

Usage (in-place, adds .int8 files alongside FP32):
    python quantize.py --input output/qwen3-asr-0.6b --output output/qwen3-asr-0.6b

Usage (separate output dir):
    python quantize.py --input output/qwen3-asr-0.6b --output output/qwen3-asr-0.6b-int8
"""

import argparse
import json
import os
import shutil
import tempfile

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType



ONNX_FILES = ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"]


def _total_size(path: str) -> int:
    """Get total size of an ONNX model including external data."""
    size = os.path.getsize(path)
    data_path = path + ".data"
    if os.path.exists(data_path):
        size += os.path.getsize(data_path)
    return size


def _simplify_if_needed(input_path: str) -> str:
    """Try to simplify the model with onnxsim to fix shape inference issues.

    Returns the path to use for quantization (temp file if simplified, original otherwise).
    """
    try:
        import onnxsim
        model = onnx.load(input_path, load_external_data=True)
        model_sim, ok = onnxsim.simplify(model)
        if ok:
            tmpfile = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            tmpfile.close()  # Close before writing so onnx.save can open it on all platforms
            onnx.save(model_sim, tmpfile.name)
            print(f"    Simplified for shape inference compatibility")
            return tmpfile.name
    except Exception:
        pass
    return input_path


def quantize_onnx_file(
    input_path: str,
    output_path: str,
    op_types_to_quantize=None,
    weight_type=QuantType.QInt8,
    nodes_to_exclude=None,
):
    """Apply dynamic quantization to an ONNX file.

    op_types_to_quantize: if given, restricts quantization to those op types.
    For the encoder, pass ['MatMul'] to skip Conv layers — Conv quantization produces
    ConvInteger ops not supported by ORT <1.24 on Linux.
    weight_type: QuantType.QInt8 (default) or QuantType.QInt16.
    nodes_to_exclude: list of node names to keep in FP32 (e.g. lm_head).
    """
    print(f"  Quantizing {os.path.basename(input_path)}...")
    if nodes_to_exclude:
        print(f"    Excluding nodes: {nodes_to_exclude}")

    quant_kwargs = dict(
        op_types_to_quantize=op_types_to_quantize,
        weight_type=weight_type,
        use_external_data_format=True,
    )
    if nodes_to_exclude:
        quant_kwargs["nodes_to_exclude"] = nodes_to_exclude

    # Try direct quantization first, fall back to simplification
    tmp_path = None
    try:
        try:
            quantize_dynamic(input_path, output_path, **quant_kwargs)
        except Exception as e:
            # Shape inference may fail on dynamo-exported models; simplify first
            print(f"    Direct quantization failed ({e}), retrying with onnxsim")
            tmp_path = _simplify_if_needed(input_path)
            quantize_dynamic(tmp_path, output_path, **quant_kwargs)
    finally:
        if tmp_path and tmp_path != input_path:
            os.unlink(tmp_path)

    in_size = _total_size(input_path)
    out_size = _total_size(output_path)
    ratio = out_size / in_size
    print(
        f"    {in_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB "
        f"({ratio:.1%})"
    )



def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8")
    parser.add_argument("--input", required=True, help="Input directory with FP32 ONNX files")
    parser.add_argument("--output", required=True, help="Output directory for INT8 files")
    parser.add_argument(
        "--no-share-weights",
        action="store_true",
        help="Skip weight sharing for split decoder after quantization",
    )
    parser.add_argument(
        "--nodes-to-exclude",
        type=str,
        default=None,
        help="Comma-separated node names to exclude from quantization (keep in FP32)",
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        default="int8",
        choices=["int8", "int16"],
        help="Quantization weight type (default: int8)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Parse weight type
    wtype_map = {"int8": QuantType.QInt8, "int16": QuantType.QInt16}
    weight_type = wtype_map[args.weight_type]

    # Parse nodes to exclude
    exclude_nodes = None
    if args.nodes_to_exclude:
        exclude_nodes = [n.strip() for n in args.nodes_to_exclude.split(",")]

    # Quantize ONNX files with .int8. suffix for coexistence with FP32.
    # Encoder: MatMul-only — Conv quantization produces ConvInteger ops not supported by ORT <1.24.
    # Decoders: all ops (decoders have no Conv layers).
    suffix = f".{args.weight_type}"  # e.g. ".int8"
    print(f"Quantizing ONNX files (weight_type={args.weight_type})...")
    for filename in ONNX_FILES:
        input_path = os.path.join(args.input, filename)
        if not os.path.exists(input_path):
            print(f"  Skipping {filename} (not found)")
            continue

        # Output with suffix: encoder.int8.onnx, decoder_init.int8.onnx, etc.
        base = filename.removesuffix(".onnx")
        output_filename = f"{base}{suffix}.onnx"
        output_path = os.path.join(args.output, output_filename)
        encoder_only_matmul = ["MatMul"] if filename == "encoder.onnx" else None
        # Only apply node exclusions to decoder files (encoder uses op_type filtering)
        file_exclude = exclude_nodes if filename != "encoder.onnx" else None
        quantize_onnx_file(
            input_path, output_path,
            op_types_to_quantize=encoder_only_matmul,
            weight_type=weight_type,
            nodes_to_exclude=file_exclude,
        )

    # Embed encoder weights into the INT8 .onnx proto (eliminates .data file)
    encoder_out = os.path.join(args.output, f"encoder{suffix}.onnx")
    encoder_data = encoder_out + ".data"
    if os.path.exists(encoder_data):
        print("  Embedding INT8 encoder weights into .onnx proto...")
        enc = onnx.load(encoder_out, load_external_data=True)
        onnx.save(enc, encoder_out)
        os.remove(encoder_data)

    # Inline quantized decoder_step weights into .onnx proto if under 2 GB.
    step_out = os.path.join(args.output, f"decoder_step{suffix}.onnx")
    step_data = step_out + ".data"
    if os.path.exists(step_data):
        step_size = os.path.getsize(step_data)
        if step_size < 1.8e9:  # safe margin under 2 GB protobuf limit
            print(f"\nInlining INT8 decoder_step weights into .onnx proto...")
            step_model = onnx.load(step_out, load_external_data=True)
            onnx.save(step_model, step_out)
            os.remove(step_data)
            print(f"  {os.path.basename(step_out)}: {os.path.getsize(step_out) / 1e6:.1f} MB (self-contained)")

    # Copy non-quantized files (skip if input == output)
    if os.path.realpath(args.input) != os.path.realpath(args.output):
        print("\nCopying non-quantized files...")
        for fname in os.listdir(args.input):
            # Skip quantized variants
            if suffix in fname:
                continue
            src = os.path.join(args.input, fname)
            dst = os.path.join(args.output, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  Copied {fname}")

    # Update config.json quantization metadata
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config_src = os.path.join(args.input, "config.json")
        with open(config_src) as f:
            config = json.load(f)
    quant_info = config.get("quantization", {})
    if isinstance(quant_info, str):
        quant_info = {}
    for name in ["encoder", "decoder_init", "decoder_step"]:
        quant_info[name] = f"{args.weight_type}_dynamic"
    config["quantization"] = quant_info
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("  Updated config.json with quantization metadata")

    # Summary
    print(f"\nQuantization complete. Output: {args.output}")
    total_input = 0
    total_output = 0
    for f in os.listdir(args.output):
        path = os.path.join(args.output, f)
        size = os.path.getsize(path)
        total_output += size
    for f in os.listdir(args.input):
        path = os.path.join(args.input, f)
        size = os.path.getsize(path)
        total_input += size

    print(f"  Total input:  {total_input / 1e6:.1f} MB")
    print(f"  Total output: {total_output / 1e6:.1f} MB")
    print(f"  Compression:  {total_output / total_input:.1%}")


if __name__ == "__main__":
    main()
