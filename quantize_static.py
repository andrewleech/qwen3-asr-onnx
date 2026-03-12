#!/usr/bin/env python3
"""
Static INT8 quantization for Qwen3-ASR decoder ONNX models.

Unlike dynamic quantization (quantize.py) which only quantizes weights,
static quantization quantizes both weights AND activations using calibration
data to determine optimal scale/zero_point per tensor.

Calibration pipeline:
  1. Stream audio samples from LibriSpeech train.clean.100
  2. Run encoder (FP32) to get audio features
  3. Run decoder_init (FP32) to get initial KV cache + logits
  4. Run decoder_step (FP32) for N steps, collecting all inputs
  5. Feed collected inputs through CalibrationDataReader

Usage:
    uv run python quantize_static.py \\
        --input output/qwen3-asr-0.6b-smooth \\
        --output output/qwen3-asr-0.6b-smooth-static-int8 \\
        --n-samples 32 --n-steps 16
"""

import argparse
import io
import json
import os
import shutil
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from datasets import Audio, load_dataset
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from share_weights import share_external_models


# ---------------------------------------------------------------------------
# Calibration data collection
# ---------------------------------------------------------------------------

def collect_calibration_data(
    model_dir: str,
    n_samples: int = 32,
    n_steps: int = 16,
) -> tuple[list[dict], list[dict]]:
    """
    Collect calibration inputs for decoder_init and decoder_step.

    Runs FP32 inference on n_samples audio clips, collecting:
    - decoder_init inputs: input_embeds, position_ids
    - decoder_step inputs: input_embeds, position_ids, past_keys, past_values
      (n_steps per sample)

    Returns (init_inputs, step_inputs) as lists of numpy dict.
    """
    from src.mel import log_mel_spectrogram
    from src.prompt import build_prompt_ids, get_audio_pad_range, EOS_TOKEN_IDS

    providers = ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    encoder = ort.InferenceSession(
        os.path.join(model_dir, "encoder.onnx"), opts, providers=providers
    )
    decoder_init = ort.InferenceSession(
        os.path.join(model_dir, "decoder_init.onnx"), opts, providers=providers
    )
    decoder_step = ort.InferenceSession(
        os.path.join(model_dir, "decoder_step.onnx"), opts, providers=providers
    )

    # Load embeddings
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    dtype_str = cfg.get("embed_tokens_dtype", "float32")
    shape = cfg["embed_tokens_shape"]
    dtype = np.float16 if dtype_str == "float16" else np.float32
    embed_tokens = np.fromfile(
        os.path.join(model_dir, "embed_tokens.bin"), dtype=dtype
    ).reshape(shape).astype(np.float32)

    # Stream calibration audio
    ds = load_dataset(
        "openslr/librispeech_asr",
        "all",
        split="train.clean.100",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    init_inputs = []
    step_inputs = []
    sample_count = 0

    print(f"Collecting calibration data from {n_samples} audio samples...")

    for sample in ds:
        if sample_count >= n_samples:
            break

        raw = sample["audio"].get("bytes")
        if raw is None:
            continue
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            continue  # skip non-16k for simplicity
        if len(arr) < 3200:  # < 0.2s
            continue

        try:
            mel = log_mel_spectrogram(arr)
            mel_np = mel.cpu().numpy()
            audio_features = encoder.run(["audio_features"], {"mel": mel_np})[0]
            audio_token_count = audio_features.shape[1]
            if audio_token_count == 0:
                continue

            prompt_ids = build_prompt_ids(audio_token_count)
            input_embeds = embed_tokens[prompt_ids].copy()
            audio_start, audio_end = get_audio_pad_range(prompt_ids)
            input_embeds[audio_start:audio_end] = audio_features[0]
            input_embeds = input_embeds[np.newaxis, :, :]
            position_ids = np.arange(len(prompt_ids), dtype=np.int64)[np.newaxis, :]

            # Collect decoder_init input
            init_inputs.append({
                "input_embeds": input_embeds.copy(),
                "position_ids": position_ids.copy(),
            })

            # Run decoder_init
            logits, keys, values = decoder_init.run(
                ["logits", "present_keys", "present_values"],
                {"input_embeds": input_embeds, "position_ids": position_ids},
            )

            next_token = int(np.argmax(logits[0, -1, :]))
            pos = len(prompt_ids)

            # Collect decoder_step inputs for n_steps
            for step in range(n_steps):
                if next_token in EOS_TOKEN_IDS:
                    break

                step_embed = embed_tokens[next_token][np.newaxis, np.newaxis, :]
                step_pos = np.array([[pos]], dtype=np.int64)

                step_inputs.append({
                    "input_embeds": step_embed.copy(),
                    "position_ids": step_pos.copy(),
                    "past_keys": keys.copy(),
                    "past_values": values.copy(),
                })

                logits, keys, values = decoder_step.run(
                    ["logits", "present_keys", "present_values"],
                    {
                        "input_embeds": step_embed,
                        "position_ids": step_pos,
                        "past_keys": keys,
                        "past_values": values,
                    },
                )
                next_token = int(np.argmax(logits[0, -1, :]))
                pos += 1

            sample_count += 1
            if sample_count % 8 == 0:
                print(f"  {sample_count}/{n_samples} samples "
                      f"({len(init_inputs)} init, {len(step_inputs)} step inputs)")

        except Exception as e:
            print(f"  Warning: sample skipped ({e})")
            continue

    print(f"  Done: {len(init_inputs)} init inputs, {len(step_inputs)} step inputs")
    return init_inputs, step_inputs


class DecoderCalibrationReader(CalibrationDataReader):
    """Feeds pre-collected decoder inputs for static quantization calibration."""

    def __init__(self, inputs: list[dict]):
        self.inputs = inputs
        self.idx = 0

    def get_next(self) -> dict | None:
        if self.idx >= len(self.inputs):
            return None
        data = self.inputs[self.idx]
        self.idx += 1
        return data

    def rewind(self):
        self.idx = 0


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_decoder(
    input_path: str,
    output_path: str,
    calibration_inputs: list[dict],
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    per_channel: bool = False,
    nodes_to_exclude: list[str] | None = None,
    quant_format: QuantFormat = QuantFormat.QOperator,
):
    """Apply static INT8 quantization to a decoder ONNX model."""
    print(f"  Static quantization of {os.path.basename(input_path)} "
          f"({len(calibration_inputs)} calibration samples)...")

    reader = DecoderCalibrationReader(calibration_inputs)

    extra_options = {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
    }

    quantize_static(
        input_path,
        output_path,
        calibration_data_reader=reader,
        quant_format=quant_format,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=calibrate_method,
        nodes_to_exclude=nodes_to_exclude or [],
        use_external_data_format=True,
        extra_options=extra_options,
    )

    # Report sizes
    in_size = os.path.getsize(input_path)
    for ext in [".data", ".onnx.data"]:
        p = input_path + ext
        if os.path.exists(p):
            in_size += os.path.getsize(p)

    out_size = os.path.getsize(output_path)
    for ext in [".data", ".onnx.data"]:
        p = output_path + ext
        if os.path.exists(p):
            out_size += os.path.getsize(p)

    print(f"    {in_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB ({out_size / in_size:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Static INT8 quantization for Qwen3-ASR decoder"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input directory with FP32 ONNX files (smooth or unsmooth)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for statically quantized files",
    )
    parser.add_argument(
        "--n-samples", type=int, default=32,
        help="Number of audio samples for calibration (default: 32)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=16,
        help="Decoder steps per sample for calibration (default: 16)",
    )
    parser.add_argument(
        "--calibrate-method", type=str, default="minmax",
        choices=["minmax", "entropy", "percentile"],
        help="Calibration method (default: minmax)",
    )
    parser.add_argument(
        "--per-channel", action="store_true",
        help="Use per-channel weight quantization",
    )
    parser.add_argument(
        "--nodes-to-exclude", type=str, default=None,
        help="Comma-separated node names to exclude from quantization",
    )
    parser.add_argument(
        "--quant-format", type=str, default="qoperator",
        choices=["qoperator", "qdq"],
        help="Quantization format: qoperator (fused ops) or qdq (Q/DQ nodes). Default: qoperator",
    )
    parser.add_argument(
        "--no-share-weights", action="store_true",
        help="Skip weight sharing for split decoder",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory to cache/load calibration data (.npz)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    cal_method = method_map[args.calibrate_method]

    format_map = {
        "qoperator": QuantFormat.QOperator,
        "qdq": QuantFormat.QDQ,
    }
    quant_fmt = format_map[args.quant_format]

    exclude_nodes = None
    if args.nodes_to_exclude:
        exclude_nodes = [n.strip() for n in args.nodes_to_exclude.split(",")]

    # Collect or load calibration data
    cache_path = None
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        cache_path = os.path.join(
            args.cache_dir,
            f"calib_n{args.n_samples}_s{args.n_steps}.npz"
        )

    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached calibration data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        init_inputs = list(data["init_inputs"])
        step_inputs = list(data["step_inputs"])
        print(f"  Loaded {len(init_inputs)} init, {len(step_inputs)} step inputs")
    else:
        t0 = time.time()
        init_inputs, step_inputs = collect_calibration_data(
            args.input,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
        )
        elapsed = time.time() - t0
        print(f"  Calibration data collected in {elapsed:.1f}s")

        if cache_path:
            np.savez(
                cache_path,
                init_inputs=np.array(init_inputs, dtype=object),
                step_inputs=np.array(step_inputs, dtype=object),
            )
            print(f"  Cached to {cache_path}")

    # Quantize decoder_init
    init_path = os.path.join(args.input, "decoder_init.onnx")
    if os.path.exists(init_path):
        quantize_decoder(
            init_path,
            os.path.join(args.output, "decoder_init.onnx"),
            init_inputs,
            calibrate_method=cal_method,
            per_channel=args.per_channel,
            nodes_to_exclude=exclude_nodes,
            quant_format=quant_fmt,
        )

    # Quantize decoder_step
    step_path = os.path.join(args.input, "decoder_step.onnx")
    if os.path.exists(step_path):
        quantize_decoder(
            step_path,
            os.path.join(args.output, "decoder_step.onnx"),
            step_inputs,
            calibrate_method=cal_method,
            per_channel=args.per_channel,
            nodes_to_exclude=exclude_nodes,
            quant_format=quant_fmt,
        )

    # Share decoder weights
    if not args.no_share_weights:
        init_out = os.path.join(args.output, "decoder_init.onnx")
        step_out = os.path.join(args.output, "decoder_step.onnx")
        if os.path.exists(init_out) and os.path.exists(step_out):
            print("\nSharing quantized decoder weights...")
            share_external_models(args.output)

    # Copy encoder and supporting files from input
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
        config["quantization"] = "int8_static"
        config["calibration_method"] = args.calibrate_method
        config["calibration_samples"] = args.n_samples
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    print(f"\nStatic quantization complete. Output: {args.output}")
    total = 0
    for f in sorted(os.listdir(args.output)):
        path = os.path.join(args.output, f)
        size = os.path.getsize(path)
        total += size
        if size > 1e6:
            print(f"  {f}: {size / 1e6:.1f} MB")
    print(f"  Total: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
