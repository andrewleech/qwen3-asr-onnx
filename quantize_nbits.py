"""
quantize_nbits.py — Apply MatMulNBits (int4 or int8) block quantization to Qwen3-ASR decoder.

Uses ORT's built-in MatMulNBitsQuantizer with RTN algorithm (no calibration data needed).
Produces a MatMulNBits ONNX graph — weights are stored as packed uint8 with per-group scales.

Usage:
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b \
        --output output/trial-1.7b-int4-nbits \
        --bits 4 \
        --block-size 64

The encoder and non-decoder files are copied unchanged.
"""

import argparse
import json
import os
import shutil
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.quantization.matmul_nbits_quantizer import RTNWeightOnlyQuantConfig
from onnxruntime.quantization.quant_utils import QuantFormat


def quantize_decoder(input_path: str, output_path: str, bits: int, block_size: int) -> None:
    """Quantize a single decoder ONNX file with MatMulNBits RTN."""
    print(f"  Loading {os.path.basename(input_path)} ...")
    model = onnx.load(input_path)

    algo = RTNWeightOnlyQuantConfig(quant_format=QuantFormat.QOperator)

    quantizer = MatMulNBitsQuantizer(
        model=model,
        bits=bits,
        block_size=block_size,
        is_symmetric=False,
        algo_config=algo,
    )
    quantizer.process()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    quantizer.model.save_model_to_file(output_path, use_external_data_format=True)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input model directory (FP32 ONNX)")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits")
    parser.add_argument("--block-size", type=int, default=64, help="Block size (group size)")
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["decoder_init", "decoder_step"],
        help="Which decoder files to quantize",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Quantize each decoder file
    for name in args.decoders:
        src = os.path.join(args.input, f"{name}.onnx")
        if not os.path.exists(src):
            print(f"  Skipping {name}.onnx (not found)")
            continue
        dst = os.path.join(args.output, f"{name}.onnx")
        quantize_decoder(src, dst, args.bits, args.block_size)

    # Copy everything else unchanged
    skip_exts = {".onnx", ".data"}
    skip_names = {f"{n}.onnx" for n in args.decoders} | {f"{n}.onnx.data" for n in args.decoders}
    for fname in os.listdir(args.input):
        if fname in skip_names:
            continue
        if any(fname.endswith(e) for e in skip_exts) and fname not in ("encoder.onnx",):
            # Copy encoder unchanged
            if fname != "encoder.onnx":
                continue
        src = os.path.join(args.input, fname)
        dst = os.path.join(args.output, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

    # Update config.json quantization field
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["quantization"] = f"int{args.bits}_nbits_block{args.block_size}"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Updated config.json: quantization=int{args.bits}_nbits_block{args.block_size}")

    # Report sizes
    in_mb = sum(
        os.path.getsize(os.path.join(args.input, f))
        for f in os.listdir(args.input)
        if os.path.isfile(os.path.join(args.input, f))
    ) / 1024**2
    out_mb = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if os.path.isfile(os.path.join(args.output, f))
    ) / 1024**2
    print(f"\nDone.")
    print(f"  Input:  {in_mb:.1f} MB")
    print(f"  Output: {out_mb:.1f} MB")
    print(f"  Ratio:  {out_mb / in_mb * 100:.1f}%")


if __name__ == "__main__":
    main()
