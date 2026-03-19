"""
convert_embed_fp16.py — Convert embed_tokens.bin from float32 to float16.

Reads embed_tokens.bin (raw float32) from the model directory, derives the
shape from config.json (decoder.vocab_size × decoder.hidden_size), and writes
embed_tokens.fp16.bin (raw float16) to the same directory.

Usage:
    uv run python convert_embed_fp16.py --model-dir output/qwen3-asr-0.6b
"""

import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Convert embed_tokens.bin from float32 to float16"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Model directory containing embed_tokens.bin and config.json",
    )
    args = parser.parse_args()

    model_dir = args.model_dir

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    vocab_size = cfg["decoder"]["vocab_size"]
    hidden_size = cfg["decoder"]["hidden_size"]
    shape = (vocab_size, hidden_size)

    src_path = os.path.join(model_dir, "embed_tokens.bin")
    dst_path = os.path.join(model_dir, "embed_tokens.fp16.bin")

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"embed_tokens.bin not found in {model_dir}")

    src_size = os.path.getsize(src_path)
    expected_bytes = vocab_size * hidden_size * 4  # float32
    if src_size != expected_bytes:
        raise ValueError(
            f"embed_tokens.bin size {src_size} bytes does not match expected "
            f"{expected_bytes} bytes for shape {shape} float32"
        )

    print(f"Reading {src_path}")
    print(f"  Shape:  {shape[0]} × {shape[1]} = {shape[0] * shape[1]:,} elements")
    print(f"  Input:  {src_size / 1024**2:.1f} MB (float32)")

    embed_fp32 = np.fromfile(src_path, dtype=np.float32).reshape(shape)
    embed_fp16 = embed_fp32.astype(np.float16)
    embed_fp16.tofile(dst_path)

    dst_size = os.path.getsize(dst_path)
    print(f"  Output: {dst_size / 1024**2:.1f} MB (float16) → {dst_path}")
    print(f"  Ratio:  {dst_size / src_size * 100:.1f}%")


if __name__ == "__main__":
    main()
