#!/usr/bin/env python3
"""
Upload ONNX model files to HuggingFace Hub.

Uploads a combined FP32 + quantized release directory, replacing all existing
repo contents. Generates a model card based on which variants are present.

Usage:
    # 0.6B (FP32 + INT8)
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
| `encoder.onnx` | Audio encoder (mel → features) |
| `decoder_init.onnx` | Decoder prefill (embeddings → logits + KV cache) |
| `decoder_step.onnx` | Decoder step (token + KV cache → logits + KV cache) |
| `decoder_weights.data` | Shared external weights for both FP32 decoder models |
"""

INT8_SECTION = """\

### INT8 (AWQ-smoothed, α=0.2)

| File | Description |
|---|---|
| `encoder.int8.onnx` | INT8 quantized audio encoder |
| `decoder_init.int8.onnx` | INT8 decoder prefill |
| `decoder_step.int8.onnx` | INT8 decoder step |
| `decoder_weights.int8.data` | Shared external weights for INT8 decoders |
"""

INT4_SECTION = """\

### int4 (MatMulNBits, GPTQ decoder_init + RTN al4 decoder_step)

| File | Description |
|---|---|
| `decoder_init.int4.onnx` | int4 decoder prefill (GPTQ calibrated) |
| `decoder_init.int4.onnx.data` | Weights for int4 decoder_init |
| `decoder_step.int4.onnx` | int4 decoder step (RTN accuracy_level=4) |
| `decoder_step.int4.onnx.data` | Weights for int4 decoder_step |

The encoder is not quantized for int4; use `encoder.onnx` (FP32).
"""

MODEL_CARD_FOOTER = """\

### Shared

| File | Description |
|---|---|
| `embed_tokens.bin` | Token embedding matrix [{vocab_size}, {hidden_size}], {embed_dtype} |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config, special tokens, mel params |
| `preprocessor_config.json` | Mel spectrogram parameters (WhisperFeatureExtractor format) |
| `test_wavs/0.wav` | Short test audio clip (LibriSpeech) |
| `{tar_name}` | Quantized model + shared files in a single archive |

## Weight Sharing

FP32 and INT8 decoders use a split architecture where `decoder_init` and `decoder_step`
share a single external weights file (`decoder_weights.data` / `decoder_weights.int8.data`),
eliminating duplicate weight storage. int4 decoders have separate data files per model.

## Mel Spectrogram Parameters

Identical to Whisper: 16kHz, 128 bins, n_fft=400, hop_length=160, Hann window, Slaney mel scale, 0-8kHz.
Also documented in `preprocessor_config.json` (WhisperFeatureExtractor format).

## Inference Pipeline

1. Compute log-mel spectrogram from 16kHz audio
2. Run `encoder.onnx`: mel → audio_features
3. Build prompt token IDs with `<|audio_pad|>` placeholders
4. Look up token embeddings from `embed_tokens.bin`
5. Replace `<|audio_pad|>` embeddings with audio_features
6. Run `decoder_init.onnx`: combined embeddings → logits + KV cache
7. Greedy decode with `decoder_step.onnx` until EOS

Substitute `*.int8.onnx` or `*.int4.onnx` files for quantized inference.

## Prompt Template

```
<|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n<|audio_start|><|audio_pad|>...<|audio_pad|><|audio_end|><|im_end|>\\n<|im_start|>assistant\\n
```

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


def build_model_card(input_dir, base_model, model_name, embed_dtype, vocab_size, hidden_size, tar_name):
    """Build model card content based on which variants are present in input_dir."""
    files = set(os.listdir(input_dir))
    has_int8 = any(f.endswith(".int8.onnx") for f in files)
    has_int4 = any(f.endswith(".int4.onnx") for f in files)

    variants = []
    if has_int8:
        variants.append("INT8 (AWQ-smoothed dynamic quantization)")
    if has_int4:
        variants.append("int4 MatMulNBits")

    if variants:
        variants_intro = (
            f"Includes FP32 and {' and '.join(variants)} variants. "
            "Quantized files use `.int8.`/`.int4.` naming and coexist with FP32 in the same directory. "
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
    if has_int8:
        card += INT8_SECTION
    if has_int4:
        card += INT4_SECTION
    card += MODEL_CARD_FOOTER.format(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        embed_dtype=embed_dtype,
        tar_name=tar_name,
    )

    variants_str = "+".join(["FP32"] + (["INT8"] if has_int8 else []) + (["int4"] if has_int4 else []))
    return card, variants_str


def main():
    parser = argparse.ArgumentParser(description="Upload ONNX model to HuggingFace Hub")
    parser.add_argument("--input", required=True, help="Release directory with ONNX files")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g. user/model-onnx)")
    parser.add_argument("--model", default=None, help="Base model ID (e.g. Qwen/Qwen3-ASR-0.6B) for model card")
    parser.add_argument("--tar", action="store_true", help="Include the .tar.gz in the upload")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--dry-run", action="store_true", help="Generate model card only, don't upload")
    args = parser.parse_args()

    # Read config for model card fields
    config_path = os.path.join(args.input, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    embed_dtype = config.get("embed_tokens_dtype", "float32")
    vocab_size, hidden_size = config["embed_tokens_shape"]

    # Derive base_model from --model arg or fall back to repo name
    base_model = args.model or f"Qwen/{args.repo.rsplit('/', 1)[-1].replace('-onnx', '').upper()}"
    model_name = base_model.rsplit("/", 1)[-1]
    tar_name = os.path.basename(args.input.rstrip("/")) + ".tar.gz"

    # Build and write model card
    readme_content, variants_str = build_model_card(
        args.input, base_model, model_name, embed_dtype, vocab_size, hidden_size, tar_name
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

    # If --tar, copy the tar.gz into the upload directory
    tar_path = args.input.rstrip("/") + ".tar.gz"
    if args.tar:
        if os.path.exists(tar_path):
            import shutil
            dest = os.path.join(args.input, tar_name)
            if not os.path.exists(dest):
                print(f"Copying {tar_path} into upload directory...")
                shutil.copy2(tar_path, dest)
        else:
            print(f"WARNING: --tar specified but {tar_path} not found")

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
