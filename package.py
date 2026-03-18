#!/usr/bin/env python3
"""
Package FP32 + int4 model variants from a consolidated output directory
into a release directory ready for HuggingFace upload.

Reads from a single directory containing all variants (FP32 + quantized
with suffixed filenames) and copies only the publishable files — FP32 and
int4 variants plus shared metadata. Intermediate files (INT8, FP16, BF16)
are skipped.

Usage:
    # 0.6B
    python package.py --model 0.6b --output release/qwen3-asr-0.6b

    # 1.7B
    python package.py --model 1.7b --output release/qwen3-asr-1.7b

    # Explicit input directory
    python package.py --input output/qwen3-asr-0.6b --output release/qwen3-asr-0.6b
"""

import argparse
import os
import shutil


# FP32 model files (encoder weights inlined, decoder weights external)
FP32_FILES = [
    "encoder.onnx",
    "decoder_init.onnx",
    "decoder_init.onnx.data",
    "decoder_step.onnx",
    "decoder_step.onnx.data",
]

# int4 model files (already suffixed by quantize_nbits.py)
INT4_FILES = [
    "encoder.int4.onnx",
    "decoder_init.int4.onnx",
    "decoder_init.int4.onnx.data",
    "decoder_step.int4.onnx",
    "decoder_step.int4.onnx.data",
]

# Shared across all variants
SHARED_FILES = [
    "embed_tokens.bin",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "vocab.json",
]


def format_size(size_bytes):
    """Format byte count as human-readable string."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    else:
        return f"{size_bytes / 1e3:.1f} KB"


def copy_file(src, dst, hardlink=False):
    """Copy a file (or hardlink it), printing size."""
    size = os.path.getsize(src)
    method = ""
    if hardlink:
        try:
            os.link(src, dst)
            method = " [hardlink]"
        except OSError:
            # Cross-device or unsupported — fall back to copy
            shutil.copy2(src, dst)
            method = " [copy, hardlink failed]"
    else:
        shutil.copy2(src, dst)
    print(f"    {os.path.basename(dst)} ({format_size(size)}){method}")
    return size


def copy_file_set(input_dir, output_dir, file_list, label, hardlink=False):
    """Copy a set of files from input to output, returning total bytes copied."""
    print(f"\n{label}:")
    total = 0
    missing = []
    for name in file_list:
        src = os.path.join(input_dir, name)
        if os.path.exists(src):
            total += copy_file(src, os.path.join(output_dir, name), hardlink=hardlink)
        else:
            missing.append(name)
    if missing:
        print(f"    Missing: {', '.join(missing)}")
    return total


def verify_ort_load(output_dir):
    """Load all ONNX models in ORT as a sanity check."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not available, skipping verification")
        return True

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    ok = True
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(".onnx"):
            continue
        path = os.path.join(output_dir, name)
        try:
            ort.InferenceSession(path, opts)
            print(f"    {name}: OK")
        except Exception as e:
            print(f"    {name}: FAILED — {e}")
            ok = False
    return ok


def package(input_dir, output_dir, test_wavs_src=None, hardlink=False):
    """Package FP32 + int4 variants from consolidated input directory."""

    if os.path.exists(output_dir):
        assert os.path.realpath(output_dir) != os.path.realpath(input_dir), (
            f"output must differ from input"
        )
        print(f"Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if hardlink:
        print("Using hardlinks (no additional disk space)")

    # Detect which variants are present
    has_int4 = os.path.exists(os.path.join(input_dir, "decoder_init.int4.onnx"))

    # Copy file sets
    fp32_total = copy_file_set(input_dir, output_dir, FP32_FILES, "FP32 files", hardlink=hardlink)
    int4_total = 0
    if has_int4:
        int4_total = copy_file_set(input_dir, output_dir, INT4_FILES, "int4 files", hardlink=hardlink)
    else:
        print("\n  No int4 files found, skipping")
    shared_total = copy_file_set(input_dir, output_dir, SHARED_FILES, "Shared metadata", hardlink=hardlink)

    # Test WAVs
    if test_wavs_src is not None:
        test_wavs_dir = os.path.join(output_dir, "test_wavs")
        os.makedirs(test_wavs_dir, exist_ok=True)
        dst = os.path.join(test_wavs_dir, "0.wav")
        print(f"\nTest audio:")
        wav_size = copy_file(test_wavs_src, dst)
        shared_total += wav_size

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Release directory: {output_dir}")
    print(f"  FP32 variant:  {format_size(fp32_total + shared_total)}")
    if has_int4:
        print(f"  int4 variant:  {format_size(int4_total + shared_total)}")
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(output_dir) for f in fns)
    print(f"  Total on disk: {format_size(total)}")

    # ORT verification
    print(f"\nVerifying ORT load:")
    if not verify_ort_load(output_dir):
        print("ERROR: ORT verification failed")
        return False

    print("\nDone.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Package FP32 + int4 models from consolidated output directory"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", choices=["0.6b", "1.7b"],
        help="Model size (resolves to output/qwen3-asr-{size}/)"
    )
    group.add_argument(
        "--input", help="Explicit input directory"
    )
    parser.add_argument("--output", required=True, help="Output release directory")
    parser.add_argument(
        "--test-wavs", default=None,
        help="WAV file to include as test_wavs/0.wav",
    )
    parser.add_argument(
        "--hardlink", action="store_true",
        help="Use hardlinks instead of copies (saves disk space, same filesystem only)",
    )
    args = parser.parse_args()

    input_dir = args.input or f"output/qwen3-asr-{args.model}"

    if not os.path.isdir(input_dir):
        print(f"ERROR: input directory not found: {input_dir}")
        return 1
    if args.test_wavs and not os.path.isfile(args.test_wavs):
        print(f"ERROR: test WAV file not found: {args.test_wavs}")
        return 1

    ok = package(input_dir, args.output, test_wavs_src=args.test_wavs, hardlink=args.hardlink)
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
