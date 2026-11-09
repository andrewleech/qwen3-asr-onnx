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
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.artifact_publication import ArtifactStaging, publish_artifacts
from src.onnx_artifacts import (
    copy_artifact_entry,
    external_data_files,
    isolated_onnx_input,
    preflight_artifact_mutations,
    prepare_onnx_publication,
    retained_quantization,
    write_artifact_json,
)
from src.tokenizer_assets import (
    TOKENIZER_FILES,
    prepare_tokenizer_assets,
)

ONNX_FILES = ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"]


def _total_size(path: str) -> int:
    """Get total size of an ONNX model including external data."""
    size = os.path.getsize(path)
    data_path = path + ".data"
    if os.path.exists(data_path):
        size += os.path.getsize(data_path)
    return size


def _simplify_if_needed(input_path: str, directory: Path) -> str:
    """Try to simplify the model with onnxsim to fix shape inference issues.

    Returns the path to use for quantization (temp file if simplified, original otherwise).
    """
    try:
        import onnxsim

        model = onnx.load(input_path, load_external_data=True)
        model_sim, ok = onnxsim.simplify(model)
        if ok:
            simplified = directory / "simplified.onnx"
            onnx.save(model_sim, simplified)
            print("    Simplified for shape inference compatibility")
            return str(simplified)
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

    with isolated_onnx_input(input_path) as (staged_model, stage):
        try:
            quantize_dynamic(str(staged_model), output_path, **quant_kwargs)
        except Exception as error:
            print(f"    Direct quantization failed ({error}), retrying with onnxsim")
            simplified = _simplify_if_needed(str(staged_model), stage)
            quantize_dynamic(simplified, output_path, **quant_kwargs)

    in_size = _total_size(input_path)
    out_size = _total_size(output_path)
    ratio = out_size / in_size
    print(f"    {in_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB ({ratio:.1%})")


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
    with prepare_tokenizer_assets(args.input, args.output) as tokenizer:
        _run_conversion(args, tokenizer)


def _run_conversion(args, tokenizer):
    suffix = f".{args.weight_type}"
    files = {
        fname
        for fname in os.listdir(args.input)
        if fname not in TOKENIZER_FILES and suffix not in fname and os.path.isfile(os.path.join(args.input, fname))
    }
    generated_graphs = {
        f"{name.removesuffix('.onnx')}{suffix}.onnx"
        for name in ONNX_FILES
        if os.path.isfile(os.path.join(args.input, name))
    }
    generated = generated_graphs
    retained_graphs, previous_quantization = retained_quantization(args.output, suffix, generated_graphs)
    files.discard("config.json")
    original_reads = files | {"config.json"} | (tokenizer.read_files if tokenizer else set())
    tokenizer_files = tokenizer.names if tokenizer else frozenset()
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
        reads=ONNX_FILES,
    )
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
    with ArtifactStaging(prefix=".quantize-", dir=args.output) as stage:
        for filename in ONNX_FILES:
            input_path = os.path.join(args.input, filename)
            if not os.path.exists(input_path):
                print(f"  Skipping {filename} (not found)")
                continue

            # Output with suffix: encoder.int8.onnx, decoder_init.int8.onnx, etc.
            base = filename.removesuffix(".onnx")
            output_filename = f"{base}{suffix}.onnx"
            encoder_only_matmul = ["MatMul"] if filename == "encoder.onnx" else None
            # Only apply node exclusions to decoder files (encoder uses op_type filtering)
            file_exclude = exclude_nodes if filename != "encoder.onnx" else None
            output_path = stage.path / output_filename
            quantize_onnx_file(
                input_path,
                str(output_path),
                op_types_to_quantize=encoder_only_matmul,
                weight_type=weight_type,
                nodes_to_exclude=file_exclude,
            )
            data = external_data_files(output_path)
            if data and (
                filename == "encoder.onnx"
                or (filename == "decoder_step.onnx" and sum(path.stat().st_size for path in data) < 1.8e9)
            ):
                print(f"  Inlining {output_filename} weights into .onnx proto...")
                model = onnx.load(output_path, load_external_data=True)
                onnx.save(model, output_path)
        replacements, generated = prepare_onnx_publication(stage.path, args.output, generated_graphs)
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
            reads=ONNX_FILES,
        )
        if replacements:
            publish_artifacts(replacements)
        stage.mark_complete()

    # Copy non-quantized files (skip if input == output)
    if os.path.realpath(args.input) != os.path.realpath(args.output):
        print("\nCopying non-quantized files...")
        for fname in sorted(files):
            copy_artifact_entry(args.input, args.output, fname)
            print(f"  Copied {fname}")

    if tokenizer:
        tokenizer.publish(written_files=files | generated)

    config_path = Path(args.input) / "config.json"
    if not config_path.is_file():
        config_path = Path(args.output) / "config.json"
    config = json.loads(config_path.read_text())
    quant_info = config.get("quantization", {})
    if isinstance(quant_info, str):
        quant_info = {}
    for graph in retained_graphs:
        quant_info.pop(graph.removesuffix(f"{suffix}.onnx"), None)
    quant_info.update(previous_quantization)
    for graph in generated_graphs:
        quant_info[graph.removesuffix(f"{suffix}.onnx")] = f"{args.weight_type}_dynamic"
    config["quantization"] = quant_info
    write_artifact_json(args.output, "config.json", config)
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
