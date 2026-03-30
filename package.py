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
import json
import os
import shutil
import subprocess

import numpy as np

# FP32 model files (encoder weights inlined, decoder weights in shared file)
FP32_FILES = [
    "encoder.onnx",
    "decoder_init.onnx",
    "decoder_step.onnx",
    "decoder_weights.data",          # shared external weights (from share_weights.py)
    "decoder_init.onnx.data",        # fallback: per-decoder data (pre-sharing format)
    "decoder_step.onnx.data",        # fallback: per-decoder data (pre-sharing format)
]

# int4 model files
# encoder.int4.onnx is a copy of encoder.onnx (FP32) — INT4/INT8 encoders degrade WER.
INT4_FILES = [
    "encoder.int4.onnx",
    "decoder_init.int4.onnx",
    "decoder_step.int4.onnx",
    "decoder_weights.int4.data",     # shared external weights (from share_weights.py --suffix int4)
    "decoder_init.int4.onnx.data",   # fallback: per-decoder data (pre-sharing format)
    "decoder_step.int4.onnx.data",   # fallback: per-decoder data (pre-sharing format)
]

# Shared across all variants
SHARED_FILES = [
    # embed_tokens.bin handled separately (FP16 conversion)
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
        assert os.path.realpath(output_dir) != os.path.realpath(input_dir), "output must differ from input"
        print(f"Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if hardlink:
        print("Using hardlinks (no additional disk space)")

    # Detect which variants are present
    has_int4 = os.path.exists(os.path.join(input_dir, "decoder_init.int4.onnx"))

    # Copy file sets.
    # FP32 decoder files are never hardlinked — share_weights.py modifies them in-place.
    fp32_total = copy_file_set(input_dir, output_dir, FP32_FILES, "FP32 files", hardlink=False)
    int4_total = 0
    if has_int4:
        int4_total = copy_file_set(input_dir, output_dir, INT4_FILES, "int4 files", hardlink=False)
    else:
        print("\n  No int4 files found, skipping")
    shared_total = copy_file_set(input_dir, output_dir, SHARED_FILES, "Shared metadata", hardlink=hardlink)

    # Share FP32 decoder weights — eliminates duplicate data between init and step.
    # Both protos reference a single decoder_weights.data file.
    from share_weights import share_external_models

    init_data = os.path.join(output_dir, "decoder_init.onnx.data")
    step_data = os.path.join(output_dir, "decoder_step.onnx.data")
    if os.path.exists(init_data) and os.path.exists(step_data):
        print("\nSharing FP32 decoder weights:")
        saved = os.path.getsize(step_data)
        share_external_models(output_dir)
        fp32_total -= saved
        print(f"    Eliminated {format_size(saved)} duplicate")

    # Share int4 decoder weights — same deduplication as FP32.
    init_int4_data = os.path.join(output_dir, "decoder_init.int4.onnx.data")
    step_int4_data = os.path.join(output_dir, "decoder_step.int4.onnx.data")
    if os.path.exists(init_int4_data) and os.path.exists(step_int4_data):
        print("\nSharing int4 decoder weights:")
        saved = os.path.getsize(step_int4_data)
        share_external_models(output_dir, suffix="int4")
        int4_total -= saved
        print(f"    Eliminated {format_size(saved)} duplicate")

    # Embed tokens — ship as FP16 to halve file size (zero WER impact per experiment [104]).
    # Prefer embed_tokens.fp16.bin if present, otherwise convert embed_tokens.bin on the fly.
    embed_total = 0
    fp16_src = os.path.join(input_dir, "embed_tokens.fp16.bin")
    fp32_src = os.path.join(input_dir, "embed_tokens.bin")
    embed_dst = os.path.join(output_dir, "embed_tokens.bin")
    print("\nEmbed tokens:")
    if os.path.exists(fp16_src):
        embed_total += copy_file(fp16_src, embed_dst, hardlink=False)  # never hardlink (different name)
        print("    (copied FP16 as embed_tokens.bin)")
    elif os.path.exists(fp32_src):
        print("    Converting FP32 → FP16...")
        data = np.fromfile(fp32_src, dtype=np.float32)
        data.astype(np.float16).tofile(embed_dst)
        embed_total += os.path.getsize(embed_dst)
        print(f"    embed_tokens.bin ({format_size(embed_total)}) — converted from FP32")
    else:
        print(f"    WARNING: no embed_tokens found in {input_dir}")

    # Ensure config.json has embed_tokens_dtype: float16.
    # Break any hardlink first (config.json may be hardlinked to the source dir).
    config_dst = os.path.join(output_dir, "config.json")
    if os.path.exists(config_dst) and embed_total > 0:
        with open(config_dst) as f:
            cfg = json.load(f)
        if cfg.get("embed_tokens_dtype") != "float16":
            cfg["embed_tokens_dtype"] = "float16"
            # Write to a temp file then replace to break hardlinks
            tmp = config_dst + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, config_dst)
            print("    Set embed_tokens_dtype=float16 in config.json")

    shared_total += embed_total

    # Test WAVs
    if test_wavs_src is not None:
        test_wavs_dir = os.path.join(output_dir, "test_wavs")
        os.makedirs(test_wavs_dir, exist_ok=True)
        dst = os.path.join(test_wavs_dir, "0.wav")
        print("\nTest audio:")
        wav_size = copy_file(test_wavs_src, dst)
        shared_total += wav_size

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Release directory: {output_dir}")
    print(f"  FP32 variant:  {format_size(fp32_total + shared_total)}")
    if has_int4:
        print(f"  int4 variant:  {format_size(int4_total + shared_total)}")
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(output_dir) for f in fns)
    print(f"  Total on disk: {format_size(total)}")

    # ORT verification
    print("\nVerifying ORT load:")
    if not verify_ort_load(output_dir):
        print("ERROR: ORT verification failed")
        return False

    # Create per-variant tarballs (FP32 and int4) in the output directory's parent.
    # Each tarball contains shared files + variant-specific ONNX files.
    tar_dir = os.path.dirname(os.path.abspath(output_dir))
    base_name = os.path.basename(output_dir.rstrip("/"))
    shared_names = [f for f in SHARED_FILES if os.path.exists(os.path.join(output_dir, f))]
    shared_names.append("embed_tokens.bin")

    # FP32 variant: encoder.onnx + decoder*.onnx + decoder_weights.data (shared)
    fp32_variant_files = ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx", "decoder_weights.data"]
    # int4 variant: encoder.int4.onnx + decoder*.int4.onnx + decoder_weights.int4.data (shared)
    int4_variant_files = [
        "encoder.int4.onnx",
        "decoder_init.int4.onnx",
        "decoder_step.int4.onnx",
        "decoder_weights.int4.data",
    ]

    for variant, file_list in [("", fp32_variant_files), ("-int4", int4_variant_files)]:
        if variant == "-int4" and not has_int4:
            continue
        archive_name = f"{base_name}{variant}"
        tar_name = f"{archive_name}.tar.gz"
        tar_path = os.path.join(tar_dir, tar_name)
        # Filter to files that exist in the release dir
        members = [f for f in file_list + shared_names if os.path.exists(os.path.join(output_dir, f))]
        tar_members = [f"{base_name}/{f}" for f in members]

        print(f"\nCreating {tar_name}...")
        cmd = ["tar", "cf", "-", "-C", tar_dir]
        # Rename directory inside tar to match archive name (e.g. qwen3-asr-0.6b-int4/)
        if base_name != archive_name:
            cmd += [f"--transform=s,^{base_name},{archive_name},"]
        cmd += tar_members
        with open(tar_path, "wb") as f:
            tar_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            gz_proc = subprocess.Popen(["gzip", "-1"], stdin=tar_proc.stdout, stdout=f)
            tar_proc.stdout.close()
            gz_proc.wait()
            tar_proc.wait()

        if tar_proc.returncode == 0 and gz_proc.returncode == 0:
            size = os.path.getsize(tar_path)
            print(f"  {tar_name}: {format_size(size)}")
        else:
            print(f"  ERROR creating {tar_name}")

    print("\nDone.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Package FP32 + int4 models from consolidated output directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=["0.6b", "1.7b"], help="Model size (resolves to output/qwen3-asr-{size}/)")
    group.add_argument("--input", help="Explicit input directory")
    parser.add_argument("--output", required=True, help="Output release directory")
    parser.add_argument(
        "--test-wavs",
        default=None,
        help="WAV file to include as test_wavs/0.wav",
    )
    parser.add_argument(
        "--hardlink",
        action="store_true",
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
