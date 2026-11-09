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
import time
from pathlib import Path

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

from share_weights import share_external_models, shared_data_requires_publication
from src.artifact_metadata import MODEL_METADATA_FILES
from src.artifact_publication import ArtifactStaging, publish_artifacts
from src.calibration_cache import (
    cache_key,
    file_fingerprints,
    load_cache,
    pack_samples,
    recipe,
    save_cache,
    unpack_samples,
)
from src.onnx_artifacts import (
    copy_artifact_entry,
    graph_file_closure,
    isolated_onnx_input,
    preflight_artifact_mutations,
    prepare_onnx_publication,
    write_artifact_json,
)
from src.tokenizer_assets import (
    prepare_tokenizer_assets,
    tokenizer_asset_names,
)

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
    from src.prompt import EOS_TOKEN_IDS, build_prompt_ids, get_audio_pad_range

    providers = ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    encoder = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), opts, providers=providers)
    decoder_init = ort.InferenceSession(os.path.join(model_dir, "decoder_init.onnx"), opts, providers=providers)
    decoder_step = ort.InferenceSession(os.path.join(model_dir, "decoder_step.onnx"), opts, providers=providers)

    from src.embeddings import load_embed_tokens
    from src.prompt import load_prompt_tokenizer

    embed_tokens = load_embed_tokens(model_dir)
    tokenizer = load_prompt_tokenizer(model_dir)

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

            prompt_ids = build_prompt_ids(audio_token_count, tokenizer)
            input_embeds = embed_tokens[prompt_ids].copy()
            audio_start, audio_end = get_audio_pad_range(prompt_ids)
            input_embeds[audio_start:audio_end] = audio_features[0]
            input_embeds = input_embeds[np.newaxis, :, :]
            position_ids = np.arange(len(prompt_ids), dtype=np.int64)[np.newaxis, :]

            # Collect decoder_init input
            init_inputs.append(
                {
                    "input_embeds": input_embeds.copy(),
                    "position_ids": position_ids.copy(),
                }
            )

            # Run decoder_init
            logits, keys, values = decoder_init.run(
                ["logits", "present_keys", "present_values"],
                {"input_embeds": input_embeds, "position_ids": position_ids},
            )

            next_token = int(np.argmax(logits[0, -1, :]))
            pos = len(prompt_ids)

            # Collect decoder_step inputs for n_steps
            for _step in range(n_steps):
                if next_token in EOS_TOKEN_IDS:
                    break

                step_embed = embed_tokens[next_token][np.newaxis, np.newaxis, :]
                step_pos = np.array([[pos]], dtype=np.int64)

                step_inputs.append(
                    {
                        "input_embeds": step_embed.copy(),
                        "position_ids": step_pos.copy(),
                        "past_keys": keys.copy(),
                        "past_values": values.copy(),
                    }
                )

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
                print(f"  {sample_count}/{n_samples} samples ({len(init_inputs)} init, {len(step_inputs)} step inputs)")

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
    print(f"  Static quantization of {os.path.basename(input_path)} ({len(calibration_inputs)} calibration samples)...")

    reader = DecoderCalibrationReader(calibration_inputs)

    extra_options = {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
    }

    with isolated_onnx_input(input_path) as (staged_model, _stage):
        quantize_static(
            str(staged_model),
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


def calibration_identity(model_dir, n_samples, n_steps):
    root = Path(model_dir)
    graphs = {name for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx") if (root / name).is_file()}
    names = graph_file_closure(root, graphs) | tokenizer_asset_names(root)
    names.update(name for name in (*MODEL_METADATA_FILES, "embed_tokens.bin") if (root / name).is_file())
    return recipe(
        "static", file_fingerprints(root, names), {"n_samples": n_samples, "n_steps": n_steps}, Path(__file__)
    )


def main():
    parser = argparse.ArgumentParser(description="Static INT8 quantization for Qwen3-ASR decoder")
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory with FP32 ONNX files (smooth or unsmooth)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for statically quantized files",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=32,
        help="Number of audio samples for calibration (default: 32)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=16,
        help="Decoder steps per sample for calibration (default: 16)",
    )
    parser.add_argument(
        "--calibrate-method",
        type=str,
        default="minmax",
        choices=["minmax", "entropy", "percentile"],
        help="Calibration method (default: minmax)",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel weight quantization",
    )
    parser.add_argument(
        "--nodes-to-exclude",
        type=str,
        default=None,
        help="Comma-separated node names to exclude from quantization",
    )
    parser.add_argument(
        "--quant-format",
        type=str,
        default="qoperator",
        choices=["qoperator", "qdq"],
        help="Quantization format: qoperator (fused ops) or qdq (Q/DQ nodes). Default: qoperator",
    )
    parser.add_argument(
        "--no-share-weights",
        action="store_true",
        help="Skip weight sharing for split decoder",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for provenance-checked numeric calibration caches (.npz)",
    )
    args = parser.parse_args()
    with prepare_tokenizer_assets(args.input, args.output) as tokenizer:
        _run_conversion(args, tokenizer)


def _run_conversion(args, tokenizer):
    converted = {
        f"{stem}.onnx"
        for stem in ("decoder_init", "decoder_step")
        if os.path.isfile(os.path.join(args.input, f"{stem}.onnx"))
    }
    generated = converted
    share_pair = {"decoder_init.onnx", "decoder_step.onnx"} <= converted and not args.no_share_weights
    files = {"embed_tokens.bin", *MODEL_METADATA_FILES}
    if os.path.isfile(os.path.join(args.input, "encoder.onnx")):
        files.add("encoder.onnx")
    tokenizer_files = tokenizer.names if tokenizer else frozenset()
    files.difference_update(tokenizer_files)
    original_reads = files | (tokenizer.read_files if tokenizer else set())
    files = preflight_artifact_mutations(
        args.input,
        args.output,
        files,
        generated=generated,
        rewritten={"config.json"},
        read_files=original_reads,
        copied_assets=tokenizer_files,
        asset_source=tokenizer.root if tokenizer else None,
        removed=tokenizer.removed if tokenizer else (),
        reads=converted,
    )
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

    cache_path = None
    identity = None
    cached = None
    if args.cache_dir:
        identity = calibration_identity(args.input, args.n_samples, args.n_steps)
        cache_path = Path(args.cache_dir) / f"calib_{cache_key(identity)}.npz"
        cached = load_cache(cache_path, identity)
    if cached is not None:
        groups = unpack_samples(cached, ("init", "step"))
        init_inputs, step_inputs = groups["init"], groups["step"]
        print(f"Loaded {len(init_inputs)} init and {len(step_inputs)} step samples from {cache_path}")
    else:
        t0 = time.time()
        init_inputs, step_inputs = collect_calibration_data(args.input, n_samples=args.n_samples, n_steps=args.n_steps)
        print(f"Calibration data collected in {time.time() - t0:.1f}s")
        if cache_path is not None:
            save_cache(cache_path, identity, pack_samples({"init": init_inputs, "step": step_inputs}))

    with ArtifactStaging(prefix=".static-", dir=args.output) as stage:
        init_path = os.path.join(args.input, "decoder_init.onnx")
        if os.path.exists(init_path):
            quantize_decoder(
                init_path,
                str(stage.path / "decoder_init.onnx"),
                init_inputs,
                calibrate_method=cal_method,
                per_channel=args.per_channel,
                nodes_to_exclude=exclude_nodes,
                quant_format=quant_fmt,
            )

        step_path = os.path.join(args.input, "decoder_step.onnx")
        if os.path.exists(step_path):
            quantize_decoder(
                step_path,
                str(stage.path / "decoder_step.onnx"),
                step_inputs,
                calibrate_method=cal_method,
                per_channel=args.per_channel,
                nodes_to_exclude=exclude_nodes,
                quant_format=quant_fmt,
            )

        if share_pair:
            print("\nSharing quantized decoder weights...")
            share_external_models(stage.path)

        replacements, generated = prepare_onnx_publication(
            stage.path, args.output, converted, shared_data={"decoder_weights.data"} if share_pair else ()
        )
        canonical = Path(args.output) / "decoder_weights.data"
        if (
            share_pair
            and canonical.name in generated
            and not shared_data_requires_publication(stage.path / canonical.name, canonical)
        ):
            generated.remove(canonical.name)
            replacements = [(source, target) for source, target in replacements if target != canonical]
        files = preflight_artifact_mutations(
            args.input,
            args.output,
            files,
            generated=generated,
            rewritten={"config.json"},
            read_files=original_reads,
            copied_assets=tokenizer_files,
            asset_source=tokenizer.root if tokenizer else None,
            removed=tokenizer.removed if tokenizer else (),
            reads=converted,
        )
        if replacements:
            publish_artifacts(replacements)
        stage.mark_complete()

    # Copy encoder and supporting files from input
    print("\nCopying supporting files...")
    for filename in sorted(files):
        src = os.path.join(args.input, filename)
        if os.path.exists(src):
            copy_artifact_entry(args.input, args.output, filename)
            print(f"  Copied {filename}")

    if tokenizer:
        tokenizer.publish(written_files=files | generated)

    # Update config
    config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["quantization"] = "int8_static"
        config["calibration_method"] = args.calibrate_method
        config["calibration_samples"] = args.n_samples
        write_artifact_json(args.output, "config.json", config)

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
