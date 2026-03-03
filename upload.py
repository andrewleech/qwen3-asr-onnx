#!/usr/bin/env python3
"""
Upload ONNX model files to HuggingFace Hub.

Usage:
    python upload.py --input output/qwen3-asr-0.6b --repo andrewleech/qwen3-asr-0.6b-onnx
"""

import argparse
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

## Files

| File | Description |
|---|---|
| `encoder.onnx` | Audio encoder: mel [1, 128, T] -> features [1, T/8, {output_dim}] |
| `decoder_init.onnx` | Decoder prefill: embeddings -> logits + KV cache |
| `decoder_step.onnx` | Decoder step: token + KV cache -> logits + KV cache |
| `embed_tokens.bin` | Token embedding matrix [{vocab_size}, {hidden_size}], {embed_dtype} |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config |

## Mel Spectrogram Parameters

Identical to Whisper: 16kHz, 128 bins, n_fft=400, hop_length=160, Hann window, Slaney mel scale, 0-8kHz.

## Inference Pipeline

1. Compute log-mel spectrogram from 16kHz audio
2. Run `encoder.onnx`: mel -> audio_features
3. Build prompt token IDs with `<|audio_pad|>` placeholders
4. Look up token embeddings from `embed_tokens.bin`
5. Replace `<|audio_pad|>` embeddings with audio_features
6. Run `decoder_init.onnx`: combined embeddings -> logits + KV cache
7. Greedy decode with `decoder_step.onnx` until EOS

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
    parser.add_argument("--input", required=True, help="Directory with ONNX files")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g. user/model-onnx)")
    parser.add_argument("--model", default=None, help="Base model ID (e.g. Qwen/Qwen3-ASR-0.6B) for model card")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    args = parser.parse_args()

    api = HfApi()

    # Create repo
    print(f"Creating repo {args.repo}...")
    api.create_repo(args.repo, exist_ok=True, private=args.private)

    # Read config for model card fields
    import json
    config_path = os.path.join(args.input, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    embed_dtype = config.get("embed_tokens_dtype", "float32")
    vocab_size, hidden_size = config["embed_tokens_shape"]
    output_dim = config["encoder"]["output_dim"]

    # Derive base_model from --model arg or fall back to repo name
    base_model = args.model or f"Qwen/{args.repo.rsplit('/', 1)[-1].replace('-onnx', '').upper()}"
    model_name = base_model.rsplit("/", 1)[-1]

    # Write model card
    readme_content = MODEL_CARD_TEMPLATE.format(
        base_model=base_model,
        model_name=model_name,
        embed_dtype=embed_dtype,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        output_dim=output_dim,
    )
    readme_path = os.path.join(args.input, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    # Upload
    print(f"Uploading {args.input} to {args.repo}...")
    api.upload_folder(
        folder_path=args.input,
        repo_id=args.repo,
        commit_message=f"Upload {model_name} ONNX export",
    )

    print(f"Upload complete: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
