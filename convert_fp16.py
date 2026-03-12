#!/usr/bin/env python3
"""
Convert FP32 ONNX decoder models to FP16 for reduced memory bandwidth.

Uses onnxconverter_common.float16_converter for proper graph-level conversion
(updates both weights AND all intermediate tensor types to float16).

ORT CPU EP will internally cast to FP32 for compute but benefits from reduced
memory bandwidth on weight loads.

Usage:
    uv run python convert_fp16.py \
        --input output/qwen3-asr-0.6b \
        --output output/trial-fp16-dec
"""

import argparse
import json
import os
import shutil

import onnx
from onnxconverter_common import convert_float_to_float16

from share_weights import share_external_models


def convert_model_fp16(input_path: str, output_path: str) -> None:
    """Convert a single ONNX model to FP16 using graph-level conversion."""
    print(f"  Loading {os.path.basename(input_path)}...")
    model = onnx.load(input_path, load_external_data=True)

    print(f"    Converting graph to FP16...")
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,  # Keep inputs/outputs as float32 for compatibility
    )

    # Save with external data
    onnx.save(
        model_fp16,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path) + ".data",
        size_threshold=1024,
    )

    in_size = os.path.getsize(input_path)
    data_in = input_path + ".data"
    if os.path.exists(data_in):
        in_size += os.path.getsize(data_in)

    out_size = os.path.getsize(output_path)
    data_out = output_path + ".data"
    if os.path.exists(data_out):
        out_size += os.path.getsize(data_out)

    print(f"    {in_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB ({out_size / in_size:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX decoder to FP16")
    parser.add_argument("--input", required=True, help="Input directory with FP32 ONNX files")
    parser.add_argument("--output", required=True, help="Output directory for FP16 files")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["decoder_init.onnx", "decoder_step.onnx"],
        help="ONNX files to convert (default: decoder_init.onnx decoder_step.onnx)",
    )
    parser.add_argument(
        "--no-share-weights",
        action="store_true",
        help="Skip weight sharing for split decoder after conversion",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Converting ONNX files to FP16...")
    for filename in args.files:
        input_path = os.path.join(args.input, filename)
        if not os.path.exists(input_path):
            print(f"  Skipping {filename} (not found)")
            continue
        output_path = os.path.join(args.output, filename)
        convert_model_fp16(input_path, output_path)

    # Share decoder weights if both init and step were converted
    if not args.no_share_weights:
        init_out = os.path.join(args.output, "decoder_init.onnx")
        step_out = os.path.join(args.output, "decoder_step.onnx")
        if os.path.exists(init_out) and os.path.exists(step_out):
            print("\nSharing FP16 decoder weights...")
            share_external_models(args.output)

    # Copy non-converted files
    print("\nCopying supporting files...")
    for filename in ["encoder.onnx", "embed_tokens.bin", "config.json",
                     "tokenizer.json", "tokenizer_config.json",
                     "added_tokens.json", "vocab.json"]:
        src = os.path.join(args.input, filename)
        if os.path.exists(src):
            dst = os.path.join(args.output, filename)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Copied {filename}")

    # Update config
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["decoder_dtype"] = "float16"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("  Updated config.json with decoder_dtype=float16")

    print(f"\nConversion complete. Output: {args.output}")


if __name__ == "__main__":
    main()
