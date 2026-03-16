#!/usr/bin/env python3
"""
Export Qwen3-ASR to ONNX format.

Produces:
    encoder.onnx         - Audio encoder (mel -> features), weights embedded in proto
    decoder_init.onnx    - Decoder prefill (input_ids + audio -> logits + KV cache)
    decoder_step.onnx    - Decoder step (token ID + KV cache -> logits + KV cache)
    decoder_weights.data - Shared external weights for both decoder models (includes embedding table)
    tokenizer.json       - HuggingFace tokenizer
    config.json          - Architecture config + special tokens + mel params

Usage:
    python export.py --model Qwen/Qwen3-ASR-0.6B
    python export.py --model Qwen/Qwen3-ASR-1.7B
"""

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModel, AutoTokenizer

from src.encoder_wrapper import export_encoder
import onnx

from src.decoder_wrapper import export_decoder_init, export_decoder_step
from share_weights import share_external_models
from src.prompt import (
    ENDOFTEXT_TOKEN_ID,
    IM_START_TOKEN_ID,
    IM_END_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_PAD_TOKEN_ID,
    ASR_TEXT_TOKEN_ID,
    EOS_TOKEN_IDS,
)


def load_model(model_id: str, device: str = "cpu", dtype=torch.float32):
    """
    Load Qwen3-ASR model.

    The model uses a custom architecture registered by the qwen-asr package.
    If qwen-asr is not installed, we try loading with trust_remote_code=True
    to use the modeling code bundled in the HF repo (if available).
    """
    print(f"Loading model {model_id}...")

    # Try loading with qwen-asr package first, fall back to trust_remote_code
    try:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"AutoModel.from_pretrained failed: {e}")
        print("Trying with explicit qwen_asr import...")
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        )

    model.eval()
    print(f"Model loaded. Device: {device}, dtype: {dtype}")
    return model



def copy_tokenizer(model_id: str, output_dir: str):
    """Copy tokenizer.json from the HuggingFace model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Save tokenizer files
    tokenizer.save_pretrained(output_dir)

    # Remove unwanted files from save_pretrained, keep only tokenizer files
    keep_files = {"tokenizer.json", "tokenizer_config.json"}
    remove_extensions = {".py", ".txt"}
    for f in os.listdir(output_dir):
        if f in keep_files:
            continue
        path = os.path.join(output_dir, f)
        _, ext = os.path.splitext(f)
        if ext in remove_extensions or f == "special_tokens_map.json":
            os.remove(path)

    print(f"Tokenizer saved to {output_dir}")


def verify_special_tokens(model_id: str):
    """
    Verify special token IDs match our constants.

    Loads the tokenizer and checks that the token IDs we hardcoded
    in prompt.py match the actual tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    checks = {
        "<|audio_start|>": AUDIO_START_TOKEN_ID,
        "<|audio_end|>": AUDIO_END_TOKEN_ID,
        "<|audio_pad|>": AUDIO_PAD_TOKEN_ID,
        "<|im_start|>": IM_START_TOKEN_ID,
        "<|im_end|>": IM_END_TOKEN_ID,
        "<|endoftext|>": ENDOFTEXT_TOKEN_ID,
    }

    all_ok = True
    for token_str, expected_id in checks.items():
        actual_id = tokenizer.convert_tokens_to_ids(token_str)
        if actual_id != expected_id:
            print(f"  MISMATCH: {token_str} expected={expected_id} actual={actual_id}")
            all_ok = False
        else:
            print(f"  OK: {token_str} = {actual_id}")

    if not all_ok:
        raise ValueError(
            "Special token IDs do not match. Update src/prompt.py with correct values."
        )
    print("All special token IDs verified.")


def write_preprocessor_config(output_dir: str):
    """
    Write preprocessor_config.json with mel spectrogram parameters.

    Uses HF WhisperFeatureExtractor field names for compatibility with
    tools that recognise this format (faster-whisper, transformers, etc.).
    Parameters are identical to Whisper and fixed for all Qwen3-ASR model sizes.
    """
    chunk_length = 30
    sample_rate = 16000
    hop_length = 160
    n_fft = 400
    n_mels = 128

    config = {
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": n_mels,
        "sampling_rate": sample_rate,
        "hop_length": hop_length,
        "n_fft": n_fft,
        "chunk_length": chunk_length,
        "n_samples": chunk_length * sample_rate,
        "nb_max_frames": chunk_length * sample_rate // hop_length,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": False,
    }

    output_path = os.path.join(output_dir, "preprocessor_config.json")
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Preprocessor config saved to {output_path}")


def write_config(model, output_dir: str):
    """
    Write config.json with architecture parameters, special tokens, and mel params.

    These values are extracted from the model's config at export time,
    not hardcoded.
    """
    thinker_config = model.config.thinker_config
    audio_config = thinker_config.audio_config
    text_config = thinker_config.text_config

    config = {
        "model_type": "qwen3_asr",
        "encoder": {
            "num_layers": audio_config.encoder_layers,
            "hidden_size": audio_config.d_model,
            "num_heads": audio_config.encoder_attention_heads,
            "ffn_dim": audio_config.encoder_ffn_dim,
            "conv_channels": audio_config.downsample_hidden_size,
            "output_dim": audio_config.output_dim,
            "downsample_factor": 8,
            "num_mel_bins": audio_config.num_mel_bins,
        },
        "decoder": {
            "num_layers": text_config.num_hidden_layers,
            "hidden_size": text_config.hidden_size,
            "num_attention_heads": text_config.num_attention_heads,
            "num_key_value_heads": text_config.num_key_value_heads,
            "head_dim": text_config.head_dim,
            "intermediate_size": text_config.intermediate_size,
            "vocab_size": text_config.vocab_size,
            "rope_theta": text_config.rope_theta,
            "rms_norm_eps": text_config.rms_norm_eps,
            "tie_word_embeddings": text_config.tie_word_embeddings,
            "rope_scaling": {
                "mrope_section": text_config.rope_scaling.get("mrope_section", [24, 20, 20]),
                "interleaved": text_config.rope_scaling.get("mrope_interleaved", True),
            },
        },
        "mel": {
            "sample_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 128,
            "fmin": 0,
            "fmax": 8000,
        },
        "special_tokens": {
            "eos_token_ids": EOS_TOKEN_IDS,
            "pad_token_id": ENDOFTEXT_TOKEN_ID,
            "im_start_token_id": IM_START_TOKEN_ID,
            "im_end_token_id": IM_END_TOKEN_ID,
            "audio_start_token_id": AUDIO_START_TOKEN_ID,
            "audio_end_token_id": AUDIO_END_TOKEN_ID,
            "audio_pad_token_id": AUDIO_PAD_TOKEN_ID,
            "asr_text_token_id": ASR_TEXT_TOKEN_ID,
        },
    }

    output_path = os.path.join(output_dir, "config.json")
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_path}")



def _convert_to_fp16(output_dir: str, filenames: list[str]):
    """Convert FP32 ONNX files to FP16 using ORT's graph optimizer.

    Uses onnxruntime.transformers.optimizer which correctly inserts Cast nodes
    for mixed-precision ops (RMSNorm, softmax, etc.) and converts weights to
    FP16. I/O types are kept as FP32 (keep_io_types=True) so callers don't
    need to change tensor types.
    """
    from onnxruntime.transformers.optimizer import optimize_model

    for filename in filenames:
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            print(f"  Skipping {filename} (not found)")
            continue

        size_before = os.path.getsize(path)
        data_path = path + ".data"
        if os.path.exists(data_path):
            size_before += os.path.getsize(data_path)

        print(f"  Converting {filename} to FP16...")
        m = optimize_model(path, model_type='gpt2', opt_level=0)
        m.convert_float_to_float16(keep_io_types=True)

        # Remove old external data before saving
        if os.path.exists(data_path):
            os.remove(data_path)
        m.save_model_to_file(path)

        size_after = os.path.getsize(path)
        if os.path.exists(data_path):
            size_after += os.path.getsize(data_path)
        print(f"    {size_before / 1e6:.1f} MB -> {size_after / 1e6:.1f} MB")


def _default_output_dir(model_id: str) -> str:
    """Derive default output directory from model ID.

    "Qwen/Qwen3-ASR-0.6B" -> "output/qwen3-asr-0.6b"
    "Qwen/Qwen3-ASR-1.7B" -> "output/qwen3-asr-1.7b"
    "/local/path/model"    -> "output/model"
    """
    name = model_id.rstrip("/").rsplit("/", 1)[-1]
    return os.path.join("output", name.lower())


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-ASR-0.6B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for ONNX files (default: derived from model name)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export tracing (cpu or cuda)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--skip-encoder",
        action="store_true",
        help="Skip encoder export",
    )
    parser.add_argument(
        "--skip-decoder",
        action="store_true",
        help="Skip decoder export",
    )
    parser.add_argument(
        "--no-share-weights",
        action="store_true",
        help="Skip weight sharing (keep separate .data files for each decoder model)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Weight storage dtype (fp16 halves model size, ORT upcasts internally)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = _default_output_dir(args.model)

    os.makedirs(args.output, exist_ok=True)

    # Always load in FP32 initially — encoder needs FP32 for mel spectrogram input.
    model = load_model(args.model, device=args.device, dtype=torch.float32)

    # Verify special token IDs
    print("\nVerifying special token IDs...")
    verify_special_tokens(args.model)

    # Always load and export in FP32 for clean ONNX graphs. FP16 conversion
    # is applied post-export by ORT's optimizer which correctly inserts Cast
    # nodes and handles mixed-precision ops (RMSNorm, softmax).

    # Export encoder
    if not args.skip_encoder:
        print("\n=== Exporting encoder ===")
        encoder_path = os.path.join(args.output, "encoder.onnx")
        export_encoder(
            model,
            encoder_path,
            opset_version=args.opset,
            device=args.device,
        )
        # Embed encoder weights into the .onnx proto (eliminates encoder.onnx.data).
        # Safe for all model sizes: 0.6B=751MB, 1.7B=1.28GB — well under protobuf 2GB limit.
        encoder_data = encoder_path + ".data"
        if os.path.exists(encoder_data):
            print("  Embedding encoder weights into .onnx proto...")
            enc = onnx.load(encoder_path, load_external_data=True)
            onnx.save(enc, encoder_path)
            os.remove(encoder_data)

        if args.dtype == "fp16":
            print("\n=== Converting encoder to FP16 ===")
            _convert_to_fp16(args.output, ["encoder.onnx"])

    # Export decoder (split architecture: decoder_init + decoder_step)
    if not args.skip_decoder:
        print("\n=== Exporting decoder (init) ===")
        export_decoder_init(
            model,
            os.path.join(args.output, "decoder_init.onnx"),
            opset_version=args.opset,
            device=args.device,
        )
        print("\n=== Exporting decoder (step) ===")
        export_decoder_step(
            model,
            os.path.join(args.output, "decoder_step.onnx"),
            opset_version=args.opset,
            device=args.device,
        )
        # FP16 conversion must happen BEFORE inlining decoder_step (the FP32
        # inlined proto exceeds the 2 GB protobuf limit that the ORT optimizer
        # needs to parse). After FP16 conversion, weights are ~half size.
        if args.dtype == "fp16":
            print("\n=== Converting decoders to FP16 ===")
            _convert_to_fp16(args.output, ["decoder_init.onnx", "decoder_step.onnx"])

        # Inline decoder_step weights into its .onnx proto. decoder_step has no
        # embedding table, so it's well under the 2 GB protobuf limit (~596 MB
        # INT8 / ~1.2 GB FP16 / ~2.4 GB FP32). FP32 exceeds the limit and stays
        # as external data.
        step_path = os.path.join(args.output, "decoder_step.onnx")
        step_data = step_path + ".data"
        if os.path.exists(step_data):
            step_model = onnx.load(step_path, load_external_data=True)
            proto_size = sum(len(t.raw_data) for t in step_model.graph.initializer)
            if proto_size < 1_800_000_000:  # ~1.8 GB safety margin for protobuf
                print("  Inlining decoder_step weights into .onnx proto...")
                onnx.save(step_model, step_path)
                os.remove(step_data)
                print(f"  decoder_step.onnx: {os.path.getsize(step_path) / 1e6:.1f} MB (self-contained)")
            else:
                print(f"  decoder_step too large to inline ({proto_size / 1e6:.0f} MB), keeping external data")
                del step_model

        # decoder_init keeps external data (exceeds 2 GB with embedding table).
        # Rename its .data file for clarity.
        init_path = os.path.join(args.output, "decoder_init.onnx")
        init_data = init_path + ".data"
        if os.path.exists(init_data):
            final_data = os.path.join(args.output, "decoder_init.onnx.data")
            if init_data != final_data:
                os.rename(init_data, final_data)

    # Save embedding cache for fast Rust-side lookup in the decode step loop.
    if not args.skip_decoder:
        print("\n=== Saving embedding cache ===")
        embed_weight = model.thinker.model.embed_tokens.weight.data
        import numpy as np
        embed_np = embed_weight.cpu().float().numpy()
        embed_path = os.path.join(args.output, "embed_tokens.bin")
        embed_np.tofile(embed_path)
        print(f"  {embed_np.shape} ({embed_np.nbytes / 1e6:.1f} MB)")

    # Copy tokenizer
    print("\n=== Copying tokenizer ===")
    copy_tokenizer(args.model, args.output)

    # Write config
    print("\n=== Writing config ===")
    write_config(model, args.output)
    write_preprocessor_config(args.output)

    print(f"\nExport complete. Output directory: {args.output}")
    print("Files:")
    for f in sorted(os.listdir(args.output)):
        path = os.path.join(args.output, f)
        size = os.path.getsize(path)
        if size > 1e6:
            print(f"  {f}: {size / 1e6:.1f} MB")
        else:
            print(f"  {f}: {size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
