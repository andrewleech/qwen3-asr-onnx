"""
quantize_nbits.py — Apply 4-bit MatMulNBits block quantization to Qwen3-ASR decoder.

Uses ORT's MatMulNBitsQuantizer with RTN (default) or GPTQ algorithm.
Produces a MatMulNBits ONNX graph — weights are stored as packed uint8 with per-group scales.

Usage (RTN, in-place — adds int4 files alongside FP32):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b \
        --output output/qwen3-asr-1.7b \
        --bits 4 --block-size 64 --accuracy-level 4

Usage (GPTQ decoder_init + RTN decoder_step, split runs):
    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b \
        --output output/qwen3-asr-1.7b \
        --bits 4 --block-size 64 --accuracy-level 4 \
        --algo gptq --calib-data calibration_cache/1.7b_gptq_init.npz \
        --decoders decoder_init

    uv run python quantize_nbits.py \
        --input output/qwen3-asr-1.7b \
        --output output/qwen3-asr-1.7b \
        --bits 4 --block-size 64 --accuracy-level 4 \
        --algo rtn --decoders decoder_step

Output files use suffixed names (e.g. decoder_init.int4.onnx) so multiple
quantization variants can coexist in one directory alongside the FP32 source.
When --output differs from --input, all non-quantized files (encoder, tokenizer,
config, FP32 decoders) are copied to the output directory.
"""

import argparse
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.matmul_nbits_quantizer import (
    GPTQWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
    RTNWeightOnlyQuantConfig,
)
from onnxruntime.quantization.quant_utils import QuantFormat

from src.artifact_publication import ArtifactStaging, publish_artifacts
from src.onnx_artifacts import (
    copy_artifact_entry,
    external_tensor_range,
    inventory_graphs,
    isolated_onnx_input,
    model_tensors,
    preflight_artifact_mutations,
    prepare_onnx_publication,
    retained_quantization,
    validate_generated_destinations,
    validate_quantization_storage,
    write_artifact_json,
)
from src.tokenizer_assets import (
    TOKENIZER_FILES,
    prepare_tokenizer_assets,
)


class NpzCalibrationReader(CalibrationDataReader):
    """Feeds pre-collected input dicts from a numpy .npz file to GPTQ."""

    def __init__(self, path: str):
        data = np.load(path)
        n = int(data["_n_samples"])
        self._data: list[dict] = []
        for i in range(n):
            prefix = f"{i}_"
            sample_keys = [k[len(prefix) :] for k in data.files if k.startswith(prefix)]
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


def validate_quantization_request(bits, block_size, algo, calib_data_path, nodes_to_exclude=None):
    if algo not in {"rtn", "gptq"}:
        raise ValueError(f"Unsupported quantization algorithm: {algo}")
    if bits != 4:
        raise ValueError(f"{algo.upper()} supports only 4-bit MatMulNBits in the supported ORT environment; got {bits}")
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError("MatMulNBits block size must be a power of two and at least 16")
    if algo == "gptq":
        if calib_data_path is None:
            raise ValueError("--calib-data required for --algo gptq")
        if nodes_to_exclude:
            raise ValueError("GPTQ node exclusions are unsupported by the supported ORT algorithm; use RTN")


def _quantized_nodes(graph, functions, visited):
    for node in graph.node:
        key = (node.domain, node.op_type, node.overload)
        if key in functions:
            if key not in visited:
                visited.add(key)
                yield from _quantized_nodes(functions[key], functions, visited)
            continue
        if node.domain == "com.microsoft" and node.op_type == "MatMulNBits":
            yield node
        for attribute in node.attribute:
            if attribute.ref_attr_name and attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError("Precision validation of function attribute indirection is unsupported")
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _quantized_nodes(attribute.g, functions, visited)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _quantized_nodes(subgraph, functions, visited)


def verify_quantized_precision(model, bits, block_size):
    functions = {(function.domain, function.name, function.overload): function for function in model.functions}
    nodes = list(_quantized_nodes(model.graph, functions, set()))
    if not nodes:
        raise ValueError("No MatMulNBits operators were produced; refusing to label an unquantized model")
    for node in nodes:
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        actual_bits = attributes.get("bits")
        actual_block_size = attributes.get("block_size")
        if actual_bits != bits or actual_block_size != block_size:
            raise ValueError(
                f"Quantized operator {node.name!r} has bits={actual_bits}, block_size={actual_block_size}; "
                f"requested bits={bits}, block_size={block_size}. No output was written."
            )


@contextmanager
def _quantization_input(input_path: str, algo: str) -> Iterator[tuple[onnx.ModelProto | str, str | None]]:
    if algo != "gptq":
        model = onnx.load(input_path, load_external_data=False)
        validate_quantization_storage(model)
        onnx.external_data_helper.load_external_data_for_model(model, str(Path(input_path).absolute().parent))
        yield model, None
        return
    with isolated_onnx_input(input_path, prefix="qwen-gptq-") as (staged_model, stage):
        model = onnx.load(staged_model, load_external_data=False)
        external_bytes = sum(
            external_tensor_range(stage, tensor)[2]
            for tensor in model_tensors(model)
            if tensor.data_location == onnx.TensorProto.EXTERNAL
        )
        if model.ByteSize() + external_bytes < onnx.checker.MAXIMUM_PROTOBUF:
            # GPTQ runs smaller graphs from bytes, which cannot resolve adjacent data files.
            onnx.external_data_helper.load_external_data_for_model(model, str(stage))
            staged_model.write_bytes(model.SerializeToString())
        yield str(staged_model), str(stage)


def quantize_decoder(
    input_path: str,
    output_path: str,
    bits: int,
    block_size: int,
    accuracy_level: int | None,
    algo: str,
    calib_data_path: str | None,
    nodes_to_exclude: list[str] | None = None,
) -> None:
    validate_quantization_request(bits, block_size, algo, calib_data_path, nodes_to_exclude)
    print(f"  Loading {os.path.basename(input_path)} ...")
    if algo == "gptq":
        assert calib_data_path is not None
        algo_config = GPTQWeightOnlyQuantConfig(
            calibration_data_reader=NpzCalibrationReader(calib_data_path),
            block_size=block_size,
            quant_format=QuantFormat.QOperator,
        )
    else:
        algo_config = RTNWeightOnlyQuantConfig(quant_format=QuantFormat.QOperator)

    with _quantization_input(input_path, algo) as (model_arg, external_root):
        quantizer = MatMulNBitsQuantizer(
            model=model_arg,
            bits=bits,
            block_size=block_size,
            is_symmetric=False,
            accuracy_level=accuracy_level,
            algo_config=algo_config,
            nodes_to_exclude=nodes_to_exclude or None,
        )
        quantizer.process()
        model = quantizer.model.model
        verify_quantized_precision(model, bits, block_size)

        if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
            opset = model.opset_import.add()
            opset.domain = "com.microsoft"
            opset.version = 1

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if external_root is not None:
            onnx.external_data_helper.load_external_data_for_model(model, external_root)
        quantizer.model.save_model_to_file(output_path, use_external_data_format=True)
        print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input model directory (FP32 ONNX)")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Requested bits (RTN/GPTQ support 4; unsupported requests fail before output)",
    )
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
        help="Model file stems to quantize (default: decoder_init decoder_step)",
    )
    parser.add_argument(
        "--nodes-to-exclude",
        nargs="+",
        default=None,
        help="MatMul node names to exclude from quantization (keep at FP32)",
    )
    args = parser.parse_args()

    validate_quantization_request(args.bits, args.block_size, args.algo, args.calib_data, args.nodes_to_exclude)

    for name in args.decoders:
        if not name or name in {".", ".."} or any(character in name for character in "/\\:"):
            raise ValueError(f"Unsafe model name: {name}")
        path = os.path.join(args.input, f"{name}.onnx")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required decoder not found: {path}")

    with prepare_tokenizer_assets(args.input, args.output) as tokenizer:
        _run_conversion(args, tokenizer)


def _run_conversion(args, tokenizer):
    tokenizer_files = tokenizer.names if tokenizer else frozenset()
    bits_suffix = f".int{args.bits}"
    generated_graphs = {f"{name}{bits_suffix}.onnx" for name in args.decoders}
    validate_generated_destinations(args.output, generated_graphs)
    existing_graphs = inventory_graphs(args.output, generated_graphs)
    retained_graphs, previous_quantization = retained_quantization(
        args.output, bits_suffix, generated_graphs, existing=existing_graphs
    )
    excluded_graphs = (
        generated_graphs | retained_graphs | {f"{name}{bits_suffix}.onnx" for name in ("decoder_init", "decoder_step")}
    )
    excluded_files = excluded_graphs | {f"{name}.data" for name in excluded_graphs}
    files = {
        fname
        for fname in os.listdir(args.input)
        if fname not in TOKENIZER_FILES
        and fname not in excluded_files
        and os.path.isfile(os.path.join(args.input, fname))
    }
    generated = generated_graphs
    files.discard("config.json")
    original_reads = files | {"config.json"} | (tokenizer.read_files if tokenizer else set())
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
        reads=[f"{name}.onnx" for name in args.decoders],
        existing=existing_graphs,
    )
    os.makedirs(args.output, exist_ok=True)

    quantized_names = set()
    with ArtifactStaging(prefix=".nbits-", dir=args.output) as stage:
        for name in args.decoders:
            src = os.path.join(args.input, f"{name}.onnx")
            dst = str(stage.path / f"{name}{bits_suffix}.onnx")
            quantize_decoder(
                src,
                dst,
                args.bits,
                args.block_size,
                args.accuracy_level,
                args.algo,
                args.calib_data,
                args.nodes_to_exclude,
            )
            quantized_names.add(name)
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
            reads=[f"{name}.onnx" for name in args.decoders],
            existing=existing_graphs,
        )
        if replacements:
            publish_artifacts(replacements)
        stage.mark_complete()

    # Copy non-quantized files from input to output (encoder, tokenizer, config, etc.)
    # Skip if input == output (in-place quantization).
    if os.path.realpath(args.input) != os.path.realpath(args.output):
        for fname in sorted(files):
            copy_artifact_entry(args.input, args.output, fname)
            print(f"  Copied {fname}")

    if tokenizer:
        tokenizer.publish(written_files=files | generated)

    # Update config.json quantization field (merge with existing for split-decoder runs)
    config_path = os.path.join(args.input, "config.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(args.output, "config.json")
    if os.path.exists(config_path) and quantized_names:
        with open(config_path) as f:
            cfg = json.load(f)
        al_tag = f"_al{args.accuracy_level}" if args.accuracy_level is not None else ""
        tag = f"int{args.bits}_{args.algo}_block{args.block_size}{al_tag}"

        quant_info = cfg.get("quantization", {})
        if isinstance(quant_info, str):
            quant_info = {}  # migrate from old scalar format
        for graph in retained_graphs:
            quant_info.pop(graph.removesuffix(f"{bits_suffix}.onnx"), None)
        quant_info.update(previous_quantization)
        for name in quantized_names:
            quant_info[name] = tag
        cfg["quantization"] = quant_info

        write_artifact_json(args.output, "config.json", cfg)
        print(f"  Updated config.json: quantization={json.dumps(quant_info)}")

    # Report sizes
    in_mb = (
        sum(
            os.path.getsize(os.path.join(args.input, f))
            for f in os.listdir(args.input)
            if os.path.isfile(os.path.join(args.input, f))
        )
        / 1024**2
    )
    out_mb = (
        sum(
            os.path.getsize(os.path.join(args.output, f))
            for f in os.listdir(args.output)
            if os.path.isfile(os.path.join(args.output, f))
        )
        / 1024**2
    )
    print("\nDone.")
    print(f"  Input:  {in_mb:.1f} MB")
    print(f"  Output: {out_mb:.1f} MB")
    print(f"  Ratio:  {out_mb / in_mb * 100:.1f}%")


if __name__ == "__main__":
    main()
