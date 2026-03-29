#!/usr/bin/env python3
"""
Export encoder in native FP16 via autocast tracing.

Loads the model in FP32, converts weights to FP16, then traces with
torch.amp.autocast so PyTorch captures native FP16 ops where safe and
keeps FP32 for precision-sensitive ops (LayerNorm, softmax). The attention
mask uses torch.finfo(float16).min = -65504 (valid FP16) instead of the
FP32 -3.4e38 that breaks post-hoc FP16 conversion.

The output file has FP32 I/O (mel input, audio_features output) with 2 Cast
nodes at the boundary. All internal ops are native FP16. This is a drop-in
replacement for encoder.onnx / encoder.int4.onnx with half the file size
and zero measured WER impact.

Usage:
    # 0.6B — produces encoder.int4.onnx (376 MB vs 717 MB FP32)
    uv run python export_encoder_native_fp16.py \
        --model Qwen/Qwen3-ASR-0.6B \
        --output output/qwen3-asr-0.6b/encoder.int4.onnx

    # 1.7B — produces encoder.int4.onnx (639 MB vs 1.2 GB FP32)
    uv run python export_encoder_native_fp16.py \
        --model Qwen/Qwen3-ASR-1.7B \
        --output output/qwen3-asr-1.7b/encoder.int4.onnx
"""

import argparse
import os

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from src.encoder_wrapper import EncoderWrapper, _get_feat_extract_output_lengths


def _patch_io_to_fp32(model_path, output_dim):
    """Patch FP16 I/O to FP32 with Cast nodes at the boundary.

    Adds:
    - Cast(FP32 → FP16) for mel input
    - Cast(FP16 → FP32) for audio_features output

    So the encoder is a drop-in replacement for the FP32 version.
    """
    model = onnx.load(model_path, load_external_data=True)

    # Rename FP16 mel input → mel_fp16, update all references
    for inp in model.graph.input:
        if inp.name == "mel":
            inp.name = "mel_fp16"
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == "mel":
                node.input[i] = "mel_fp16"

    # Add FP32 mel input + Cast node
    mel_fp32 = helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, 128, "time"])
    model.graph.input.insert(0, mel_fp32)
    for inp in list(model.graph.input):
        if inp.name == "mel_fp16":
            model.graph.input.remove(inp)
    cast_in = helper.make_node(
        "Cast",
        ["mel"],
        ["mel_fp16"],
        to=TensorProto.FLOAT16,
        name="cast_mel_fp32_to_fp16",
    )
    model.graph.node.insert(0, cast_in)

    # Rename FP16 audio_features output → audio_features_fp16
    for out in model.graph.output:
        if out.name == "audio_features":
            out.name = "audio_features_fp16"
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name == "audio_features":
                node.output[i] = "audio_features_fp16"

    # Add FP32 output + Cast node
    af_fp32 = helper.make_tensor_value_info(
        "audio_features",
        TensorProto.FLOAT,
        [1, "enc_time", output_dim],
    )
    model.graph.output.insert(0, af_fp32)
    for out in list(model.graph.output):
        if out.name == "audio_features_fp16":
            model.graph.output.remove(out)
    cast_out = helper.make_node(
        "Cast",
        ["audio_features_fp16"],
        ["audio_features"],
        to=TensorProto.FLOAT,
        name="cast_features_fp16_to_fp32",
    )
    model.graph.node.append(cast_out)

    onnx.save(model, model_path)


def main():
    parser = argparse.ArgumentParser(description="Export encoder in native FP16 (autocast, FP32 I/O)")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--output", required=True, help="Output path (e.g. output/qwen3-asr-0.6b/encoder.int4.onnx)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--verify", action="store_true", help="Verify ORT loads the result and produces correct output shape"
    )
    args = parser.parse_args()

    print(f"Loading {args.model} in float32 (will autocast to FP16 during export)...")
    from export import load_model

    model = load_model(args.model, dtype=torch.float32)

    audio_tower = model.thinker.audio_tower
    output_dim = audio_tower.config.output_dim
    wrapper = EncoderWrapper(audio_tower).eval().half()

    # Verify forward pass
    dummy_mel = torch.randn(1, 128, 997, dtype=torch.float32)
    with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.float16):
        test_output = wrapper(dummy_mel.half())
        expected_tokens = _get_feat_extract_output_lengths(997)
        assert test_output.shape == (1, expected_tokens, output_dim), (
            f"Shape {test_output.shape} != expected (1, {expected_tokens}, {output_dim})"
        )
        print(f"  Forward pass OK: {test_output.shape}, dtype={test_output.dtype}")

    # Export ONNX with autocast
    print(f"\nExporting to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.float16):
        torch.onnx.export(
            wrapper,
            (dummy_mel.half(),),
            args.output,
            input_names=["mel"],
            output_names=["audio_features"],
            dynamic_axes={"mel": {2: "time"}, "audio_features": {1: "enc_time"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )

    # Fix Reshape allowzero for DirectML compatibility
    from src.onnx_fixup import fix_reshape_allowzero

    n = fix_reshape_allowzero(args.output)
    print(f"  Fixed {n} Reshape allowzero attrs")

    # Embed weights into proto (encoder is < 2 GB)
    data_path = args.output + ".data"
    if os.path.exists(data_path):
        print("  Embedding weights into .onnx proto...")
        enc = onnx.load(args.output, load_external_data=True)
        onnx.save(enc, args.output)
        os.remove(data_path)

    # Patch I/O from FP16 to FP32 (drop-in compatible with FP32 encoder)
    print("  Patching I/O to FP32 (2 Cast nodes at boundary)...")
    _patch_io_to_fp32(args.output, output_dim)

    size = os.path.getsize(args.output)
    print(f"  File size: {size / 1e6:.1f} MB")

    # Verify I/O types
    enc = onnx.load(args.output, load_external_data=False)
    for inp in enc.graph.input:
        t = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"  Input '{inp.name}': {t}")
    for out in enc.graph.output:
        t = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"  Output '{out.name}': {t}")

    if args.verify:
        print("\nVerifying ORT load...")
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(args.output, opts)

        mel = np.random.randn(1, 128, 997).astype(np.float32)
        result = sess.run(["audio_features"], {"mel": mel})[0]
        print(f"  ORT OK: shape={result.shape}, dtype={result.dtype}")

    print("\nDone.")


if __name__ == "__main__":
    main()
