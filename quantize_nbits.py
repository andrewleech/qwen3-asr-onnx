"""
quantize_nbits.py — Apply MatMulNBits (int4 or int8) block quantization to Qwen3-ASR decoder.

Uses ORT's MatMulNBitsQuantizer with RTN (default) or GPTQ algorithm.
Produces a MatMulNBits ONNX graph — weights are stored as packed uint8 with per-group scales.

Usage (RTN, default):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b \
        --output output/trial-1.7b-int4-nbits \
        --bits 4 \
        --block-size 64

Usage (GPTQ, calibration-based):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b \
        --output output/trial-1.7b-gptq-int4 \
        --bits 4 \
        --block-size 64 \
        --algo gptq \
        --calib-data calibration_cache/1.7b_gptq_calib.npz

Usage (accuracy_level, for testing AVX-512 VNNI paths):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-0.6b \
        --output output/trial-0.6b-int4-al1 \
        --bits 4 \
        --block-size 64 \
        --accuracy-level 1

The encoder and non-decoder files are copied unchanged.
"""

import argparse
import json
import os
import shutil

import numpy as np
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    GPTQWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
    RTNWeightOnlyQuantConfig,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quant_utils import QuantFormat


class NpzCalibrationReader(CalibrationDataReader):
    """Feeds pre-collected input dicts from a numpy .npz file to GPTQ."""

    def __init__(self, path: str):
        data = np.load(path)
        n = int(data["_n_samples"])
        self._data: list[dict] = []
        for i in range(n):
            prefix = f"{i}_"
            sample_keys = [k[len(prefix):] for k in data.files if k.startswith(prefix)]
            self._data.append({k: data[f"{prefix}{k}"] for k in sample_keys})
        self._idx = 0

    def get_next(self) -> dict | None:
        if self._idx >= len(self._data):
            return None
        item = self._data[self._idx]
        self._idx += 1
        return item

    def __len__(self):
        return len(self._data)


def quantize_decoder(
    input_path: str,
    output_path: str,
    bits: int,
    block_size: int,
    accuracy_level: int | None,
    algo: str,
    calib_data_path: str | None,
) -> None:
    """Quantize a single decoder ONNX file with MatMulNBits."""
    print(f"  Loading {os.path.basename(input_path)} ...")

    if algo == "gptq":
        if calib_data_path is None:
            raise ValueError("--calib-data required for --algo gptq")
        reader = NpzCalibrationReader(calib_data_path)
        algo_config = GPTQWeightOnlyQuantConfig(
            calibration_data_reader=reader,
            block_size=block_size,
            quant_format=QuantFormat.QOperator,
        )
        # GPTQ requires model_path to be set — pass file path, not ModelProto
        model_arg = input_path
    else:
        algo_config = RTNWeightOnlyQuantConfig(quant_format=QuantFormat.QOperator)
        model_arg = onnx.load(input_path)

    quantizer = MatMulNBitsQuantizer(
        model=model_arg,
        bits=bits,
        block_size=block_size,
        is_symmetric=False,
        accuracy_level=accuracy_level,
        algo_config=algo_config,
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
        "--accuracy-level",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        help="MatMulNBits accuracy level (0=default fp32 accum; 1-4 may use VNNI/AVX-512 paths)",
    )
    parser.add_argument(
        "--algo",
        default="rtn",
        choices=["rtn", "gptq"],
        help="Quantization algorithm: rtn (no calibration) or gptq (requires --calib-data)",
    )
    parser.add_argument(
        "--calib-data",
        default=None,
        help="Path to pickle file with calibration inputs (required for --algo gptq)",
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=["decoder_init", "decoder_step"],
        help="Which decoder files to quantize",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Quantize each decoder file
    bits_suffix = f".int{args.bits}"
    for name in args.decoders:
        src = os.path.join(args.input, f"{name}.onnx")
        if not os.path.exists(src):
            print(f"  Skipping {name}.onnx (not found)")
            continue
        dst = os.path.join(args.output, f"{name}{bits_suffix}.onnx")
        quantize_decoder(src, dst, args.bits, args.block_size, args.accuracy_level, args.algo, args.calib_data)

    # Copy everything else unchanged
    skip_exts = {".onnx", ".data"}
    # Skip the FP32 source decoder files (we wrote quantized versions above)
    skip_names = {f"{n}.onnx" for n in args.decoders} | {f"{n}.onnx.data" for n in args.decoders}
    # Also skip any already-quantized variants of the same decoders in the source dir
    skip_names |= {f"{n}{bits_suffix}.onnx" for n in args.decoders} | {f"{n}{bits_suffix}.onnx.data" for n in args.decoders}
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
        al_tag = f"_al{args.accuracy_level}" if args.accuracy_level is not None else ""
        cfg["quantization"] = f"int{args.bits}_nbits_{args.algo}_block{args.block_size}{al_tag}"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Updated config.json: quantization={cfg['quantization']}")

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
