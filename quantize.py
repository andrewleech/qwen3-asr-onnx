#!/usr/bin/env python3
"""
INT8 dynamic quantization for Qwen3-ASR ONNX models.

Quantizes encoder and decoder ONNX files using onnxruntime dynamic quantization.
Also converts embed_tokens.bin to float16.

Usage:
    python quantize.py --input output/qwen3-asr-0.6b --output output/qwen3-asr-0.6b-int8
"""

import argparse
import json
import os
import shutil

import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType


ONNX_FILES = ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"]


def quantize_onnx_file(input_path: str, output_path: str):
    """Apply INT8 dynamic quantization to an ONNX file."""
    print(f"  Quantizing {os.path.basename(input_path)}...")

    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QInt8,
    )

    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    ratio = output_size / input_size
    print(
        f"    {input_size / 1e6:.1f} MB -> {output_size / 1e6:.1f} MB "
        f"({ratio:.1%})"
    )


def quantize_embeddings(input_dir: str, output_dir: str):
    """Convert embed_tokens.bin from float32 to float16."""
    input_path = os.path.join(input_dir, "embed_tokens.bin")
    output_path = os.path.join(output_dir, "embed_tokens.bin")

    print("  Converting embed_tokens.bin to float16...")

    config_path = os.path.join(input_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    shape = config["embed_tokens_shape"]
    embed = np.fromfile(input_path, dtype=np.float32).reshape(shape)
    embed_fp16 = embed.astype(np.float16)
    embed_fp16.tofile(output_path)

    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    print(f"    {input_size / 1e6:.1f} MB -> {output_size / 1e6:.1f} MB")

    return shape


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8")
    parser.add_argument("--input", required=True, help="Input directory with FP32 ONNX files")
    parser.add_argument("--output", required=True, help="Output directory for INT8 files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Quantize ONNX files
    print("Quantizing ONNX files...")
    for filename in ONNX_FILES:
        input_path = os.path.join(args.input, filename)
        if not os.path.exists(input_path):
            print(f"  Skipping {filename} (not found)")
            continue

        output_path = os.path.join(args.output, filename)
        quantize_onnx_file(input_path, output_path)

    # Quantize embeddings
    print("\nQuantizing embeddings...")
    shape = quantize_embeddings(args.input, args.output)

    # Copy non-quantized files
    print("\nCopying config and tokenizer...")
    for filename in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(args.input, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, filename))
            print(f"  Copied {filename}")

    # Write updated config with fp16 dtype
    config_path = os.path.join(args.input, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["embed_tokens_dtype"] = "float16"
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
