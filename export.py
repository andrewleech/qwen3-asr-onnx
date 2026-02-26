#!/usr/bin/env python3
"""
Export Qwen3-ASR to ONNX format.

Produces:
    encoder.onnx         - Audio encoder (mel -> features)
    decoder_init.onnx    - Decoder prefill (embeddings -> logits + KV cache)
    decoder_step.onnx    - Decoder step (token + KV cache -> logits + KV cache)
    embed_tokens.bin     - Token embedding matrix [vocab_size, hidden_size] as raw float32
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

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.encoder_wrapper import export_encoder
from src.decoder_wrapper import export_decoder_init, export_decoder_step
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


def extract_embed_tokens(model, output_dir: str):
    """
    Save the token embedding matrix as raw float32 bytes.

    Shape: [vocab_size, hidden_size] (e.g. [151936, 1024] for 0.6B, [151936, 2048] for 1.7B)
    Format: row-major float32, suitable for memory-mapped loading in Rust.
    """
    embed_weight = model.thinker.model.embed_tokens.weight.data
    print(f"Embedding matrix shape: {embed_weight.shape}, dtype: {embed_weight.dtype}")

    # Convert to float32 if needed (model may be loaded in bfloat16)
    embed_np = embed_weight.cpu().float().numpy()

    output_path = os.path.join(output_dir, "embed_tokens.bin")
    embed_np.tofile(output_path)
    print(f"Embedding matrix saved to {output_path} ({embed_np.nbytes / 1e6:.1f} MB)")

    return embed_np.shape


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


def write_config(model, output_dir: str, embed_shape: tuple):
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
        "embed_tokens_shape": list(embed_shape),
        "embed_tokens_dtype": "float32",
    }

    output_path = os.path.join(output_dir, "config.json")
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_path}")


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
    args = parser.parse_args()

    if args.output is None:
        args.output = _default_output_dir(args.model)

    os.makedirs(args.output, exist_ok=True)

    # Load model in float32 for export (ONNX tracing works best with fp32)
    model = load_model(args.model, device=args.device, dtype=torch.float32)

    # Verify special token IDs
    print("\nVerifying special token IDs...")
    verify_special_tokens(args.model)

    # Export encoder
    if not args.skip_encoder:
        print("\n=== Exporting encoder ===")
        export_encoder(
            model,
            os.path.join(args.output, "encoder.onnx"),
            opset_version=args.opset,
            device=args.device,
        )

    # Export decoder
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

    # Extract embedding matrix
    print("\n=== Extracting embedding matrix ===")
    embed_shape = extract_embed_tokens(model, args.output)

    # Copy tokenizer
    print("\n=== Copying tokenizer ===")
    copy_tokenizer(args.model, args.output)

    # Write config
    print("\n=== Writing config ===")
    write_config(model, args.output, embed_shape)

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
