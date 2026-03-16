#!/usr/bin/env python3
"""
INT8 dynamic quantization for Qwen3-ASR ONNX models.

Quantizes encoder and decoder ONNX files using onnxruntime dynamic quantization.
Embedding table is part of the ONNX graph (Gather op) and is not affected by
dynamic quantization, which only targets MatMul ops.

Usage:
    python quantize.py --input output/qwen3-asr-0.6b --output output/qwen3-asr-0.6b-int8
"""

import argparse
import json
import os
import shutil
import tempfile

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

from share_weights import share_external_models


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

    # Quantize ONNX files
    # Encoder: MatMul-only — Conv quantization produces ConvInteger ops not supported by ORT <1.24.
    # Decoders: all ops (decoders have no Conv layers).
    print(f"Quantizing ONNX files (weight_type={args.weight_type})...")
    for filename in ONNX_FILES:
        input_path = os.path.join(args.input, filename)
        if not os.path.exists(input_path):
            print(f"  Skipping {filename} (not found)")
            continue

        output_path = os.path.join(args.output, filename)
        encoder_only_matmul = ["MatMul"] if filename == "encoder.onnx" else None
        # Only apply node exclusions to decoder files (encoder uses op_type filtering)
        file_exclude = exclude_nodes if filename != "encoder.onnx" else None
        quantize_onnx_file(
            input_path, output_path,
            op_types_to_quantize=encoder_only_matmul,
            weight_type=weight_type,
            nodes_to_exclude=file_exclude,
        )

    # Embed encoder weights into the INT8 .onnx proto (eliminates encoder.onnx.data)
    encoder_out = os.path.join(args.output, "encoder.onnx")
    encoder_data = encoder_out + ".data"
    if os.path.exists(encoder_data):
        print("  Embedding INT8 encoder weights into .onnx proto...")
        enc = onnx.load(encoder_out, load_external_data=True)
        onnx.save(enc, encoder_out)
        os.remove(encoder_data)

    # Share decoder weights
    if not args.no_share_weights:
        print("\nSharing quantized decoder weights...")
        share_external_models(args.output)

    # Copy non-quantized files
    print("\nCopying config and tokenizer...")
    for filename in ["tokenizer.json", "tokenizer_config.json", "added_tokens.json", "vocab.json"]:
        src = os.path.join(args.input, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, filename))
            print(f"  Copied {filename}")

    # Write updated config with quantization metadata
    config_path = os.path.join(args.input, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["quantization"] = "int8_dynamic"

    output_config = os.path.join(args.output, "config.json")
    with open(output_config, "w") as f:
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
