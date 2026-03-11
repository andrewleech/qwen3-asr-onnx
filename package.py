#!/usr/bin/env python3
"""
Package combined FP32 + INT8 models into a single release directory.

Merges FP32 and INT8 output directories, renaming INT8 files with .int8.
convention and rewriting ONNX external_data references to match.

Usage:
    python package.py \
        --fp32 output/qwen3-asr-0.6b \
        --int8 output/qwen3-asr-0.6b-int8 \
        --output release/qwen3-asr-0.6b \
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

# External data file renames
INT8_DATA_RENAME = {
    "encoder.onnx.data": "encoder.int8.onnx.data",
    "decoder_weights.data": "decoder_weights.int8.data",
}

# Metadata/tokenizer files to copy from FP32 dir
METADATA_FILES = [
    "config.json",
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


def package(fp32_dir, int8_dir, output_dir, create_tar):
    """Package FP32 + INT8 into a combined release directory."""

    if os.path.exists(output_dir):
        print(f"Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- FP32 files ---
    print(f"\nCopying FP32 files from {fp32_dir}:")
    fp32_files = [
        "encoder.onnx",
        "encoder.onnx.data",
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

    # --- INT8 ONNX protos (prefer .int8. naming, fall back to renaming) ---
    print(f"\nCopying INT8 files from {int8_dir}:")
    int8_needs_rewrite = set()
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

    # --- INT8 data files (prefer .int8. naming, fall back to renaming) ---
    int8_data_needs_rewrite = {}
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

    # --- Metadata files from FP32 ---
    print(f"\nCopying metadata:")
    for name in METADATA_FILES:
        src = os.path.join(fp32_dir, name)
        if os.path.exists(src):
            copy_file(src, os.path.join(output_dir, name))

    # --- Summary ---
    print(f"\nRelease directory: {output_dir}")
    total = 0
    for name in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, name))
        total += size
    print(f"  Total: {total / 1e9:.2f} GB ({len(os.listdir(output_dir))} files)")

    # --- ORT verification ---
    print(f"\nVerifying ORT load:")
    if not verify_ort_load(output_dir):
        print("ERROR: ORT verification failed")
        return False

    # --- tar.gz (INT8 only — smaller download for Handy) ---
    if create_tar:
        model_name = os.path.basename(output_dir.rstrip("/"))
        tar_path = output_dir.rstrip("/") + ".tar.gz"
        # Include INT8 models + shared metadata/embeddings (no FP32 ONNX files).
        # The Rust engine auto-detects .int8. files when called with Quantization::default().
        int8_tar_files = set()
        for name in sorted(os.listdir(output_dir)):
            is_fp32_onnx = (
                name.endswith(".onnx") and ".int8." not in name
            )
            is_fp32_data = name in ("encoder.onnx.data", "decoder_weights.data")
            if is_fp32_onnx or is_fp32_data:
                continue
            int8_tar_files.add(name)

        print(f"\nCreating {tar_path} (INT8 only)...")
        tar_total = 0
        for name in sorted(int8_tar_files):
            filepath = os.path.join(output_dir, name)
            size = os.path.getsize(filepath)
            tar_total += size
            if size > 1e9:
                print(f"    {name} ({size / 1e9:.2f} GB)")
            elif size > 1e6:
                print(f"    {name} ({size / 1e6:.1f} MB)")
        print(f"  Uncompressed: {tar_total / 1e9:.2f} GB")

        with tarfile.open(tar_path, "w:gz") as tar:
            for name in sorted(int8_tar_files):
                filepath = os.path.join(output_dir, name)
                arcname = f"{model_name}/{name}"
                tar.add(filepath, arcname=arcname)
        tar_size = os.path.getsize(tar_path)
        print(f"  tar.gz: {tar_size / 1e9:.2f} GB")

    print("\nDone.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Package combined FP32 + INT8 models for release"
    )
    parser.add_argument("--fp32", required=True, help="FP32 model directory")
    parser.add_argument("--int8", required=True, help="INT8 model directory")
    parser.add_argument("--output", required=True, help="Output release directory")
    parser.add_argument("--tar", action="store_true", help="Create .tar.gz archive")
    args = parser.parse_args()

    if not os.path.isdir(args.fp32):
        print(f"ERROR: FP32 directory not found: {args.fp32}")
        return 1
    if not os.path.isdir(args.int8):
        print(f"ERROR: INT8 directory not found: {args.int8}")
        return 1

    ok = package(args.fp32, args.int8, args.output, args.tar)
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
