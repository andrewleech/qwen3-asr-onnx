"""
quantize_nbits.py — Apply MatMulNBits (int4 or int8) block quantization to Qwen3-ASR decoder.

Uses ORT's MatMulNBitsQuantizer with RTN (default) or GPTQ algorithm.
Produces a MatMulNBits ONNX graph — weights are stored as packed uint8 with per-group scales.

Usage (RTN, default):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b-v3 \
        --output output/qwen3-asr-1.7b-v3-int4 \
        --bits 4 \
        --block-size 64

Usage (GPTQ decoder_init + RTN decoder_step, split runs):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b-v3 \
        --output output/qwen3-asr-1.7b-v3-gptq-int4 \
        --bits 4 --block-size 64 --accuracy-level 4 \
        --algo gptq --calib-data calibration_cache/1.7b_v3_gptq_init.npz \
        --decoders decoder_init

    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b-v3 \
        --output output/qwen3-asr-1.7b-v3-gptq-int4 \
        --bits 4 --block-size 64 --accuracy-level 4 \
        --algo rtn --decoders decoder_step

Usage (accuracy_level, for testing AVX-512 VNNI paths):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-0.6b \
        --output output/trial-0.6b-int4-al1 \
        --bits 4 \
        --block-size 64 \
        --accuracy-level 1

The encoder and non-decoder files are copied unchanged.
Output decoder files are named {name}.onnx (not {name}.int4.onnx) with external
data in {name}.onnx.data, matching standard ONNX conventions.
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

    # neural_compressor's ONNXModel calls AutoConfig.from_pretrained() on the parent
    # directory when it finds a config.json. Our custom model_type causes it to fail.
    # Temporarily hide config.json during GPTQ quantization.
    model_dir = os.path.dirname(input_path)
    config_path = os.path.join(model_dir, "config.json")
    config_hidden = os.path.join(model_dir, f".config.json.{os.getpid()}.hidden")
    hide_config = algo == "gptq" and os.path.exists(config_path)
    if hide_config:
        os.rename(config_path, config_hidden)

    try:
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

        # ORT >=1.23 removed the `bits` parameter (always 4-bit); older versions require it.
        import inspect
        sig = inspect.signature(MatMulNBitsQuantizer.__init__)
        kwargs = dict(
            model=model_arg,
            block_size=block_size,
            is_symmetric=False,
            accuracy_level=accuracy_level,
            algo_config=algo_config,
        )
        if "bits" in sig.parameters:
            kwargs["bits"] = bits
        elif bits != 4:
            raise ValueError(f"This ORT version only supports 4-bit quantization, got bits={bits}")
        quantizer = MatMulNBitsQuantizer(**kwargs)
        quantizer.process()
    finally:
        if hide_config:
            if os.path.exists(config_hidden):
                os.rename(config_hidden, config_path)
            else:
                print(f"  WARNING: {config_hidden} missing, config.json may not be restored")

    # ORT's quantizer inserts MatMulNBits (com.microsoft domain) nodes but doesn't
    # add the required opset import. Fix this for ONNX spec compliance.
    model = quantizer.model.model
    ms_domains = [o for o in model.opset_import if o.domain == "com.microsoft"]
    has_ms_ops = any(n.op_type == "MatMulNBits" for n in model.graph.node)
    if has_ms_ops and not ms_domains:
        opset = model.opset_import.add()
        opset.domain = "com.microsoft"
        opset.version = 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    quantizer.model.save_model_to_file(output_path, use_external_data_format=True)
    print(f"  Saved to {output_path}")


def _rename_to_standard(output_dir: str, name: str, bits_suffix: str) -> None:
    """Rename quantized output to standard names and fix external data references.

    ORT saves as {name}{bits_suffix}.onnx + {name}{bits_suffix}.onnx.data.
    We rename to {name}.onnx + {name}.onnx.data and patch the external data
    location entries in the protobuf.
    """
    suffixed_onnx = os.path.join(output_dir, f"{name}{bits_suffix}.onnx")
    suffixed_data = suffixed_onnx + ".data"
    std_onnx = os.path.join(output_dir, f"{name}.onnx")
    std_data = std_onnx + ".data"

    if suffixed_onnx == std_onnx:
        return  # no rename needed (e.g. bits_suffix is empty)

    if not os.path.exists(suffixed_onnx):
        return

    # Rename files
    if os.path.exists(std_onnx):
        os.remove(std_onnx)
    os.rename(suffixed_onnx, std_onnx)

    if os.path.exists(suffixed_data):
        if os.path.exists(std_data):
            os.remove(std_data)
        os.rename(suffixed_data, std_data)

    # Fix external data location references in the protobuf.
    # Load without external data — we only need to patch the metadata, not the weights.
    # Use SerializeToString to write just the proto without touching the .data file.
    m = onnx.load(std_onnx, load_external_data=False)
    expected_location = f"{name}.onnx.data"
    changed = False
    for t in m.graph.initializer:
        for entry in t.external_data:
            if entry.key == "location" and entry.value != expected_location:
                entry.value = expected_location
                changed = True
    if changed:
        with open(std_onnx, "wb") as f:
            f.write(m.SerializeToString())
        print(f"  Renamed to {name}.onnx (fixed external data refs)")
    else:
        print(f"  Renamed to {name}.onnx")


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
        help="Path to .npz file with calibration inputs (required for --algo gptq)",
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
    quantized_names = set()
    for name in args.decoders:
        src = os.path.join(args.input, f"{name}.onnx")
        if not os.path.exists(src):
            print(f"  Skipping {name}.onnx (not found)")
            continue
        dst = os.path.join(args.output, f"{name}{bits_suffix}.onnx")
        quantize_decoder(src, dst, args.bits, args.block_size, args.accuracy_level, args.algo, args.calib_data)
        _rename_to_standard(args.output, name, bits_suffix)
        quantized_names.add(name)

    # Copy everything else unchanged from input to output.
    # Protect decoder .onnx/.data files that were produced by this or a prior run.
    all_decoder_names = {"decoder_init", "decoder_step"}
    protected = set()
    for n in all_decoder_names:
        protected |= {f"{n}.onnx", f"{n}.onnx.data",
                      f"{n}{bits_suffix}.onnx", f"{n}{bits_suffix}.onnx.data"}
    for fname in os.listdir(args.input):
        if fname in protected:
            continue
        # Skip non-encoder .onnx/.data files (e.g. leftover quantized variants in source)
        if fname != "encoder.onnx" and any(fname.endswith(e) for e in (".onnx", ".data")):
            continue
        src = os.path.join(args.input, fname)
        dst = os.path.join(args.output, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")

    # Update config.json quantization field (merge with existing for split-decoder runs)
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path) and quantized_names:
        with open(config_path) as f:
            cfg = json.load(f)
        al_tag = f"_al{args.accuracy_level}" if args.accuracy_level is not None else ""
        tag = f"int{args.bits}_{args.algo}_block{args.block_size}{al_tag}"

        quant_info = cfg.get("quantization", {})
        if isinstance(quant_info, str):
            quant_info = {}  # migrate from old scalar format
        for name in quantized_names:
            quant_info[name] = tag
        cfg["quantization"] = quant_info

        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Updated config.json: quantization={json.dumps(quant_info)}")

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
