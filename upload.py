#!/usr/bin/env python3
"""
Upload ONNX model files to HuggingFace Hub.

Uploads a combined FP32 + quantized release directory, replacing all existing
repo contents. Generates a model card based on which variants are present.

Usage:
    # 0.6B (FP32 + int4)
    python upload.py --input release/qwen3-asr-0.6b --repo andrewleech/qwen3-asr-0.6b-onnx --model Qwen/Qwen3-ASR-0.6B

    # 1.7B (FP32 + int4)
    python upload.py --input release/qwen3-asr-1.7b --repo andrewleech/qwen3-asr-1.7b-onnx --model Qwen/Qwen3-ASR-1.7B
"""

import argparse
import json
import os

from huggingface_hub import HfApi


MODEL_CARD_HEADER = """\
---
license: apache-2.0
base_model: {base_model}
pipeline_tag: automatic-speech-recognition
tags:
  - automatic-speech-recognition
  - onnx
  - asr
  - speech-recognition
  - qwen3
language:
  - multilingual
library_name: onnxruntime
---

# {model_name} ONNX

ONNX export of [{base_model}](https://huggingface.co/{base_model}) for use with ONNX Runtime.

{variants_intro}

## Files

### FP32

| File | Description |
|---|---|
| `encoder.onnx` | Audio encoder — mel spectrogram to features (weights inlined) |
| `decoder_init.onnx` | Decoder prefill — accepts `input_ids`, outputs logits + KV cache |
| `decoder_init.onnx.data` | External weights for FP32 decoder_init |
| `decoder_step.onnx` | Decoder autoregressive step — accepts `input_embeds` + KV cache |
| `decoder_step.onnx.data` | External weights for FP32 decoder_step |
"""

INT4_SECTION = """\

### int4 ({int4_method})

| File | Description |
|---|---|
| `encoder.int4.onnx` | Encoder (native FP16 via autocast export, FP32 I/O, half the size of FP32) |
| `decoder_init.int4.onnx` | int4 decoder prefill |
| `decoder_init.int4.onnx.data` | Weights for int4 decoder_init |
| `decoder_step.int4.onnx` | int4 decoder step |
| `decoder_step.int4.onnx.data` | Weights for int4 decoder_step |
"""

MODEL_CARD_FOOTER = """\

### Shared

| File | Description |
|---|---|
| `embed_tokens.bin` | Token embedding matrix [{vocab_size}, {hidden_size}], float32 |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config, special tokens, mel params |
| `preprocessor_config.json` | Mel spectrogram parameters (WhisperFeatureExtractor format) |

## Architecture

- **Encoder**: mel → windowed Conv2D + windowed attention → audio features
- **Decoder**: Qwen3 (28 layers, GQA, SwiGLU, MRoPE) in split init/step format

The two decoder models use different input strategies:
- `decoder_init` (prefill) accepts `input_ids` and has the embedding table in its graph — handles audio feature scatter internally
- `decoder_step` (autoregressive) accepts pre-looked-up `input_embeds` — keeps the embedding table out of its graph

The consumer loads `embed_tokens.bin` once at startup and performs the embedding lookup per token before calling `decoder_step`.

## Mel Spectrogram Parameters

Identical to Whisper: 16kHz, 128 bins, n_fft=400, hop_length=160, Hann window, Slaney mel scale, 0-8kHz.
Also documented in `preprocessor_config.json` (WhisperFeatureExtractor format).

## Inference Pipeline

1. Compute log-mel spectrogram from 16kHz audio
2. Run `encoder.onnx` (or `encoder.int4.onnx`): mel → audio_features
3. Build prompt token IDs: `<|im_start|>system<|im_end|><|im_start|>user<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|><|im_start|>assistant`
4. Run `decoder_init.onnx` (or `.int4`): `input_ids` + `audio_features` + `audio_offset` → logits + KV cache
5. Greedy decode with `decoder_step.onnx` (or `.int4`): look up `embed_tokens.bin` → `input_embeds` + KV cache → logits, until EOS

## Special Token IDs

| Token | ID |
|---|---|
| `<\\|audio_start\\|>` | 151669 |
| `<\\|audio_end\\|>` | 151670 |
| `<\\|audio_pad\\|>` | 151676 |
| `<\\|im_end\\|>` (EOS) | 151645 |
| `<\\|endoftext\\|>` (EOS) | 151643 |

## Export Tool

Exported with [qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx).
"""


def build_model_card(input_dir, base_model, model_name, vocab_size, hidden_size, config):
    """Build model card content based on which variants are present in input_dir."""
    files = set(os.listdir(input_dir))
    has_int4 = any(f.endswith(".int4.onnx") for f in files)

    # Determine int4 quantization method from config
    quant_info = config.get("quantization", {})
    init_method = quant_info.get("decoder_init", "")
    if "gptq" in init_method.lower():
        int4_method = "GPTQ decoder_init + RTN decoder_step, accuracy_level=4"
    else:
        int4_method = "RTN, accuracy_level=4"

    variants = []
    if has_int4:
        variants.append("int4 MatMulNBits")

    if variants:
        variants_intro = (
            f"Includes FP32 and {' and '.join(variants)} variants. "
            "Quantized files use `.int4.` naming and coexist with FP32 in the same directory. "
            "The [transcribe-rs](https://github.com/andrewleech/transcribe-rs) engine "
            "auto-detects quantized models by file presence."
        )
    else:
        variants_intro = (
            "FP32 export for use with ONNX Runtime. "
            "The [transcribe-rs](https://github.com/andrewleech/transcribe-rs) engine "
            "loads these files directly."
        )

    card = MODEL_CARD_HEADER.format(
        base_model=base_model,
        model_name=model_name,
        variants_intro=variants_intro,
    )
    if has_int4:
        card += INT4_SECTION.format(int4_method=int4_method)

    card += MODEL_CARD_FOOTER.format(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )

    variants_str = "+".join(["FP32"] + (["int4"] if has_int4 else []))
    return card, variants_str


def main():
    parser = argparse.ArgumentParser(description="Upload ONNX model to HuggingFace Hub")
    parser.add_argument("--input", required=True, help="Release directory with ONNX files")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g. user/model-onnx)")
    parser.add_argument("--model", default=None, help="Base model ID (e.g. Qwen/Qwen3-ASR-0.6B) for model card")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--dry-run", action="store_true", help="Generate model card only, don't upload")
    parser.add_argument("--force", action="store_true", help="Skip upload confirmation prompt")
    args = parser.parse_args()

    # Read config for model card fields
    config_path = os.path.join(args.input, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Derive embedding shape from decoder config
    decoder = config["decoder"]
    vocab_size = decoder["vocab_size"]
    hidden_size = decoder["hidden_size"]

    # Derive base_model from --model arg or fall back to repo name
    base_model = args.model or f"Qwen/{args.repo.rsplit('/', 1)[-1].replace('-onnx', '').upper()}"
    model_name = base_model.rsplit("/", 1)[-1]

    # Build and write model card
    readme_content, variants_str = build_model_card(
        args.input, base_model, model_name, vocab_size, hidden_size, config
    )
    readme_path = os.path.join(args.input, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Wrote model card: {readme_path} ({variants_str})")

    if args.dry_run:
        print("Dry run — skipping upload")
        print(readme_content)
        return

    api = HfApi()

    # Create repo
    print(f"Creating repo {args.repo}...")
    api.create_repo(args.repo, exist_ok=True, private=args.private)

    # Confirm before destructive upload
    if not args.force:
        print(f"\nTarget repo: https://huggingface.co/{args.repo}")
        print(f"This will replace ALL existing contents of that repo.")
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    # Upload, replacing all existing contents
    print(f"Uploading {args.input} to {args.repo}...")
    api.upload_folder(
        folder_path=args.input,
        repo_id=args.repo,
        commit_message=f"Upload {model_name} ONNX ({variants_str})",
        delete_patterns=["*"],
    )

    print(f"Upload complete: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
