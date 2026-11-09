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
from pathlib import Path

import onnx
from onnxconverter_common import convert_float_to_float16

from share_weights import share_external_models, shared_data_requires_publication
from src.artifact_metadata import MODEL_METADATA_FILES
from src.artifact_publication import ArtifactStaging, publish_artifacts
from src.onnx_artifacts import (
    copy_artifact_entry,
    preflight_artifact_mutations,
    prepare_onnx_publication,
    write_artifact_json,
)
from src.tokenizer_assets import (
    prepare_tokenizer_assets,
)


def convert_model_fp16(input_path: str, output_path: str) -> None:
    """Convert a single ONNX model to FP16 using graph-level conversion."""
    print(f"  Loading {os.path.basename(input_path)}...")
    model = onnx.load(input_path, load_external_data=True)

    print("    Converting graph to FP16...")
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
    with prepare_tokenizer_assets(args.input, args.output) as tokenizer:
        _run_conversion(args, tokenizer)


def _run_conversion(args, tokenizer):
    converted = {name for name in args.files if os.path.isfile(os.path.join(args.input, name))}
    generated = converted
    share_pair = {"decoder_init.onnx", "decoder_step.onnx"} <= converted and not args.no_share_weights
    files = {"embed_tokens.bin", *MODEL_METADATA_FILES}
    if os.path.isfile(os.path.join(args.input, "encoder.onnx")) and "encoder.onnx" not in converted:
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

    print("Converting ONNX files to FP16...")
    with ArtifactStaging(prefix=".fp16-", dir=args.output) as stage:
        for filename in args.files:
            input_path = os.path.join(args.input, filename)
            if not os.path.exists(input_path):
                print(f"  Skipping {filename} (not found)")
                continue
            convert_model_fp16(input_path, str(stage.path / filename))
        if share_pair:
            print("\nSharing FP16 decoder weights...")
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

    # Copy non-converted files
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
        config["decoder_dtype"] = "float16"
        write_artifact_json(args.output, "config.json", config)
        print("  Updated config.json with decoder_dtype=float16")

    print(f"\nConversion complete. Output: {args.output}")


if __name__ == "__main__":
    main()
