#!/usr/bin/env python3
"""
Upload ONNX model files to HuggingFace Hub.

Uploads a combined FP32 + INT8 release directory, replacing all existing
repo contents. Generates a model card documenting both variants.

Usage:
    python upload.py --input release/qwen3-asr-0.6b --repo andrewleech/qwen3-asr-0.6b-onnx --model Qwen/Qwen3-ASR-0.6B
    python upload.py --input release/qwen3-asr-0.6b --repo andrewleech/qwen3-asr-0.6b-onnx --model Qwen/Qwen3-ASR-0.6B --tar
"""

import argparse
import json
import os

from huggingface_hub import HfApi


MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
base_model: {base_model}
tags:
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

Includes both FP32 and INT8 (dynamic quantization) variants. The INT8 files use `.int8.` naming
and can coexist in the same directory as FP32. The [transcribe-rs](https://github.com/andrewleech/transcribe-rs)
engine auto-detects INT8 models when present.

## Files

### FP32

| File | Description |
|---|---|
| `encoder.onnx` | Audio encoder (mel → features) |
| `decoder_init.onnx` | Decoder prefill (embeddings → logits + KV cache) |
| `decoder_step.onnx` | Decoder step (token + KV cache → logits + KV cache) |
| `decoder_weights.data` | Shared external weights for both decoder models |

### INT8

| File | Description |
|---|---|
| `encoder.int8.onnx` | INT8 quantized audio encoder |
| `decoder_init.int8.onnx` | INT8 decoder prefill |
| `decoder_step.int8.onnx` | INT8 decoder step |
| `decoder_weights.int8.data` | Shared external weights for INT8 decoders |

### Shared

| File | Description |
|---|---|
| `embed_tokens.bin` | Token embedding matrix [{vocab_size}, {hidden_size}], {embed_dtype} |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config |
| `{tar_name}` | All files in a single archive for download |

## Weight Sharing

Each decoder variant (FP32 and INT8) uses a split decoder architecture with shared weights:
`decoder_init` and `decoder_step` reference the same external data file (`decoder_weights.data`
or `decoder_weights.int8.data`), eliminating duplicate weight storage.

## Mel Spectrogram Parameters

Identical to Whisper: 16kHz, 128 bins, n_fft=400, hop_length=160, Hann window, Slaney mel scale, 0-8kHz.

## Inference Pipeline

1. Compute log-mel spectrogram from 16kHz audio
2. Run `encoder.onnx`: mel → audio_features
3. Build prompt token IDs with `<|audio_pad|>` placeholders
4. Look up token embeddings from `embed_tokens.bin`
5. Replace `<|audio_pad|>` embeddings with audio_features
6. Run `decoder_init.onnx`: combined embeddings → logits + KV cache
7. Greedy decode with `decoder_step.onnx` until EOS

For INT8 inference, substitute `*.int8.onnx` for the encoder and decoder models.

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

    # Write model card
    readme_content = MODEL_CARD_TEMPLATE.format(
        base_model=base_model,
        model_name=model_name,
        embed_dtype=embed_dtype,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        tar_name=tar_name,
    )
    readme_path = os.path.join(args.input, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"Wrote model card: {readme_path}")

    if args.dry_run:
        print("Dry run — skipping upload")
        return

    api = HfApi()

    # Create repo
    print(f"Creating repo {args.repo}...")
    api.create_repo(args.repo, exist_ok=True, private=args.private)

    # Build upload path: either the directory alone, or a temp dir including the tar
    upload_dir = args.input

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
    print(f"Uploading {upload_dir} to {args.repo}...")
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=args.repo,
        commit_message=f"Upload {model_name} ONNX (FP32 + INT8, shared weights)",
        delete_patterns=["*"],
    )

    print(f"Upload complete: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
