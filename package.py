#!/usr/bin/env python3
"""
Package combined FP32 + INT8 + int4 models into a single release directory.

Merges FP32, INT8, and optionally int4 output directories. INT8 files are
renamed with the .int8. convention; int4 files use .int4. convention.
ONNX external_data references are rewritten to match renamed data files.

Usage:
    # 0.6B: FP32 + INT8
    python package.py \
        --fp32 output/qwen3-asr-0.6b \
        --int8 output/qwen3-asr-0.6b-int8 \
        --output release/qwen3-asr-0.6b \
        --test-wavs tests/fixtures/librispeech_0.wav \
        --tar

    # 1.7B: FP32 + int4
    python package.py \
        --fp32 output/qwen3-asr-1.7b \
        --int4 output/qwen3-asr-1.7b-int4 \
        --output release/qwen3-asr-1.7b \
        --test-wavs tests/fixtures/librispeech_0.wav \
        --tar
"""

import argparse
import os
import shutil
import tarfile

import onnx


# Mapping of INT8 source filenames to their renamed counterparts.
# Only ONNX proto files need rewriting; data files are just renamed.
INT8_RENAME = {
    "encoder.onnx": "encoder.int8.onnx",
    "decoder_init.onnx": "decoder_init.int8.onnx",
    "decoder_step.onnx": "decoder_step.int8.onnx",
}

# External data file renames for INT8
INT8_DATA_RENAME = {
    "decoder_weights.data": "decoder_weights.int8.data",
}

# int4 files use .int4. naming convention (output directly by quantize_nbits.py).
# Encoder is not quantized; only decoder files carry the suffix.
INT4_FILES = [
    "decoder_init.int4.onnx",
    "decoder_step.int4.onnx",
]

# int4 decoder data files (one per decoder; not shared like FP32/INT8)
INT4_DATA_PATTERN = ".int4.onnx.data"

# Metadata/tokenizer files to copy from FP32 dir
METADATA_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "vocab.json",
]


def rewrite_external_data(proto_path, data_renames):
    """Rewrite external_data location references in an ONNX proto.

    Args:
        proto_path: Path to the ONNX file to rewrite (modified in-place).
        data_renames: Dict mapping old filenames to new filenames.
    """
    model = onnx.load(proto_path, load_external_data=False)
    changed = 0
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location" and entry.value in data_renames:
                old = entry.value
                entry.value = data_renames[old]
                changed += 1
                break
    if changed:
        print(f"    Rewrote {changed} external_data references in {os.path.basename(proto_path)}")
        with open(proto_path, "wb") as f:
            f.write(model.SerializeToString())
    return changed


def copy_test_wavs(src_wav: str, output_dir: str):
    """Copy a test WAV file into test_wavs/0.wav in the output directory."""
    test_wavs_dir = os.path.join(output_dir, "test_wavs")
    os.makedirs(test_wavs_dir, exist_ok=True)
    dst = os.path.join(test_wavs_dir, "0.wav")
    shutil.copy2(src_wav, dst)
    size = os.path.getsize(dst)
    print(f"    test_wavs/0.wav ({size / 1e3:.1f} KB)")


def verify_ort_load(output_dir):
    """Load all ONNX models in ORT as a sanity check."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not available, skipping ORT verification")
        return True

    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    models = [
        "encoder.onnx",
        "decoder_init.onnx",
        "decoder_step.onnx",
        "encoder.int8.onnx",
        "decoder_init.int8.onnx",
        "decoder_step.int8.onnx",
        "decoder_init.int4.onnx",
        "decoder_step.int4.onnx",
    ]

    ok = True
    for name in models:
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            continue
        try:
            ort.InferenceSession(path, opts)
            print(f"    {name}: OK")
        except Exception as e:
            print(f"    {name}: FAILED — {e}")
            ok = False
    return ok


def copy_file(src, dst):
    """Copy a file, printing size info."""
    size = os.path.getsize(src)
    if size > 1e9:
        size_str = f"{size / 1e9:.2f} GB"
    elif size > 1e6:
        size_str = f"{size / 1e6:.1f} MB"
    else:
        size_str = f"{size / 1e3:.1f} KB"
    print(f"    {os.path.basename(dst)} ({size_str})")
    shutil.copy2(src, dst)


def package(fp32_dir, int8_dir, output_dir, create_tar, int4_dir=None, test_wavs_src=None):
    """Package FP32 + INT8 into a combined release directory."""

    if os.path.exists(output_dir):
        print(f"Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- FP32 files ---
    print(f"\nCopying FP32 files from {fp32_dir}:")
    fp32_files = [
        "encoder.onnx",
        "decoder_init.onnx",
        "decoder_step.onnx",
        "decoder_weights.data",
        "embed_tokens.bin",
    ]
    for name in fp32_files:
        src = os.path.join(fp32_dir, name)
        if not os.path.exists(src):
            print(f"    WARNING: {name} not found in FP32 dir, skipping")
            continue
        copy_file(src, os.path.join(output_dir, name))

    # --- INT8 ONNX protos and data files (optional) ---
    int8_needs_rewrite = set()
    int8_data_needs_rewrite = {}
    if int8_dir is not None:
        print(f"\nCopying INT8 files from {int8_dir}:")
        for old_name, new_name in INT8_RENAME.items():
            # Prefer already-renamed file (e.g. encoder.int8.onnx) over renaming encoder.onnx
            src_int8 = os.path.join(int8_dir, new_name)
            src_orig = os.path.join(int8_dir, old_name)
            if os.path.exists(src_int8):
                copy_file(src_int8, os.path.join(output_dir, new_name))
            elif os.path.exists(src_orig):
                copy_file(src_orig, os.path.join(output_dir, new_name))
                int8_needs_rewrite.add(new_name)
            else:
                print(f"    WARNING: neither {new_name} nor {old_name} found in INT8 dir, skipping")

        for old_name, new_name in INT8_DATA_RENAME.items():
            src_int8 = os.path.join(int8_dir, new_name)
            src_orig = os.path.join(int8_dir, old_name)
            if os.path.exists(src_int8):
                copy_file(src_int8, os.path.join(output_dir, new_name))
            elif os.path.exists(src_orig):
                copy_file(src_orig, os.path.join(output_dir, new_name))
                int8_data_needs_rewrite[old_name] = new_name
            else:
                print(f"    WARNING: neither {new_name} nor {old_name} found in INT8 dir, skipping")

    # --- Rewrite external_data references in renamed INT8 protos ---
    if int8_data_needs_rewrite:
        print(f"\nRewriting INT8 external_data references:")
        for old_name, new_name in INT8_RENAME.items():
            proto_path = os.path.join(output_dir, new_name)
            if not os.path.exists(proto_path):
                continue
            rewrite_external_data(proto_path, int8_data_needs_rewrite)

    # --- int4 files (already named with .int4. suffix by quantize_nbits.py) ---
    if int4_dir is not None:
        print(f"\nCopying int4 files from {int4_dir}:")
        for name in INT4_FILES:
            src = os.path.join(int4_dir, name)
            if not os.path.exists(src):
                print(f"    WARNING: {name} not found in int4 dir, skipping")
                continue
            copy_file(src, os.path.join(output_dir, name))
            # Copy accompanying data file if present
            data_name = name + ".data"
            data_src = os.path.join(int4_dir, data_name)
            if os.path.exists(data_src):
                copy_file(data_src, os.path.join(output_dir, data_name))

    # --- Metadata files from FP32 ---
    print(f"\nCopying metadata:")
    for name in METADATA_FILES:
        src = os.path.join(fp32_dir, name)
        if os.path.exists(src):
            copy_file(src, os.path.join(output_dir, name))

    # --- test_wavs/ ---
    if test_wavs_src is not None:
        print(f"\nCopying test audio:")
        copy_test_wavs(test_wavs_src, output_dir)

    # --- Summary ---
    print(f"\nRelease directory: {output_dir}")
    total = 0
    for name in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            total += os.path.getsize(path)
    print(f"  Total: {total / 1e9:.2f} GB")

    # --- ORT verification ---
    print(f"\nVerifying ORT load:")
    if not verify_ort_load(output_dir):
        print("ERROR: ORT verification failed")
        return False

    # --- tar.gz (quantized only — excludes FP32 ONNX for smaller download) ---
    if create_tar:
        model_name = os.path.basename(output_dir.rstrip("/"))
        tar_path = output_dir.rstrip("/") + ".tar.gz"
        # Include quantized models (INT8, int4) + shared metadata/embeddings.
        # Exclude FP32 ONNX files and FP32 decoder_weights.data.
        # The Rust engine auto-detects .int8./.int4. files by file-presence probing.
        tar_entries = []
        for name in sorted(os.listdir(output_dir)):
            filepath = os.path.join(output_dir, name)
            is_fp32_onnx = name.endswith(".onnx") and ".int8." not in name and ".int4." not in name
            is_fp32_data = name == "decoder_weights.data"
            if is_fp32_onnx or is_fp32_data:
                continue
            tar_entries.append((name, filepath))

        print(f"\nCreating {tar_path} (quantized + shared files)...")
        tar_total = 0
        for name, filepath in tar_entries:
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                tar_total += size
                if size > 1e9:
                    print(f"    {name} ({size / 1e9:.2f} GB)")
                elif size > 1e6:
                    print(f"    {name} ({size / 1e6:.1f} MB)")
            else:
                print(f"    {name}/ (directory)")
        print(f"  Uncompressed: {tar_total / 1e9:.2f} GB")

        with tarfile.open(tar_path, "w:gz") as tar:
            for name, filepath in tar_entries:
                arcname = f"{model_name}/{name}"
                tar.add(filepath, arcname=arcname, recursive=True)
        tar_size = os.path.getsize(tar_path)
        print(f"  tar.gz: {tar_size / 1e9:.2f} GB")

    print("\nDone.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Package combined FP32 + INT8 + int4 models for release"
    )
    parser.add_argument("--fp32", required=True, help="FP32 model directory")
    parser.add_argument("--int8", default=None, help="INT8 model directory (optional)")
    parser.add_argument("--int4", default=None, help="int4 model directory (optional)")
    parser.add_argument("--output", required=True, help="Output release directory")
    parser.add_argument("--tar", action="store_true", help="Create .tar.gz archive")
    parser.add_argument(
        "--test-wavs",
        default=None,
        help="WAV file to include as test_wavs/0.wav in the release",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.fp32):
        print(f"ERROR: FP32 directory not found: {args.fp32}")
        return 1
    if args.int8 and not os.path.isdir(args.int8):
        print(f"ERROR: INT8 directory not found: {args.int8}")
        return 1
    if args.int4 and not os.path.isdir(args.int4):
        print(f"ERROR: int4 directory not found: {args.int4}")
        return 1
    if args.test_wavs and not os.path.isfile(args.test_wavs):
        print(f"ERROR: test WAV file not found: {args.test_wavs}")
        return 1

    ok = package(args.fp32, args.int8, args.output, args.tar,
                 int4_dir=args.int4, test_wavs_src=args.test_wavs)
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
