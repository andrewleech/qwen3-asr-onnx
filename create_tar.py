#!/usr/bin/env python3
"""
Create tar.gz archives for individual model variants and upload to HuggingFace.

Creates one archive at a time to manage disk space, uploading and deleting
before creating the next.

Usage:
    python create_tar.py --dry-run   # Create tars, show sizes, don't upload
    python create_tar.py             # Create and upload all 4 archives
"""

import argparse
import os
import subprocess
import sys

from huggingface_hub import HfApi


# Shared metadata files included in every archive
METADATA = ["embed_tokens.bin", "config.json", "tokenizer.json"]

# Archives are built from release/ dirs (produced by package.py), which contain
# the correct FP16 embed_tokens.bin and patched config.json.
ARCHIVES = [
    {
        "name": "qwen3-asr-0.6b",
        "source_dir": "release/qwen3-asr-0.6b",
        "repo": "andrewleech/qwen3-asr-0.6b-onnx",
        "files": [
            "encoder.onnx",
            "decoder_init.onnx",
            "decoder_step.onnx",
            "decoder_weights.data",
        ],
    },
    {
        "name": "qwen3-asr-0.6b-int4",
        "source_dir": "release/qwen3-asr-0.6b",
        "repo": "andrewleech/qwen3-asr-0.6b-onnx",
        "files": [
            "encoder.int4.onnx",
            "decoder_init.int4.onnx",
            "decoder_init.int4.onnx.data",
            "decoder_step.int4.onnx",
            "decoder_step.int4.onnx.data",
        ],
    },
    {
        "name": "qwen3-asr-1.7b",
        "source_dir": "release/qwen3-asr-1.7b",
        "repo": "andrewleech/qwen3-asr-1.7b-onnx",
        "files": [
            "encoder.onnx",
            "decoder_init.onnx",
            "decoder_step.onnx",
            "decoder_weights.data",
        ],
    },
    {
        "name": "qwen3-asr-1.7b-int4",
        "source_dir": "release/qwen3-asr-1.7b",
        "repo": "andrewleech/qwen3-asr-1.7b-onnx",
        "files": [
            "encoder.int4.onnx",
            "decoder_init.int4.onnx",
            "decoder_init.int4.onnx.data",
            "decoder_step.int4.onnx",
            "decoder_step.int4.onnx.data",
        ],
    },
]


def create_tar(archive, output_dir):
    """Create a tar.gz archive with the correct directory structure."""
    name = archive["name"]
    source_dir = archive["source_dir"]
    all_files = archive["files"] + METADATA
    tar_path = os.path.join(output_dir, f"{name}.tar.gz")

    # Verify all source files exist
    for f in all_files:
        src = os.path.join(source_dir, f)
        if not os.path.exists(src):
            print(f"  ERROR: {src} not found")
            return None

    # Get source directory basename for --transform
    source_basename = os.path.basename(source_dir.rstrip("/"))

    # Build tar file list (relative to parent of source_dir)
    parent_dir = os.path.dirname(os.path.abspath(source_dir))
    tar_members = [f"{source_basename}/{f}" for f in all_files]

    # Use --transform to rename the top-level directory if needed
    cmd = ["tar", "cf", "-", "-C", parent_dir]
    if source_basename != name:
        cmd += [f"--transform=s,^{source_basename},{name},"]
    cmd += tar_members

    # Pipe through gzip -1 (fast, weights are incompressible)
    print(f"  Creating {tar_path}...")
    with open(tar_path, "wb") as f:
        tar_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        gz_proc = subprocess.Popen(
            ["gzip", "-1"], stdin=tar_proc.stdout, stdout=f
        )
        tar_proc.stdout.close()
        gz_proc.wait()
        tar_proc.wait()

    if tar_proc.returncode != 0 or gz_proc.returncode != 0:
        print(f"  ERROR: tar/gzip failed")
        return None

    size = os.path.getsize(tar_path)
    print(f"  Size: {size / 1e6:.1f} MB ({size / 1e9:.2f} GB)")
    return tar_path


def upload_tar(tar_path, repo, filename):
    """Upload a single file to an existing HuggingFace repo."""
    api = HfApi()
    print(f"  Uploading {filename} to {repo}...")
    api.upload_file(
        path_or_fileobj=tar_path,
        path_in_repo=filename,
        repo_id=repo,
        commit_message=f"Add {filename}",
    )
    print(f"  Upload complete")


def main():
    parser = argparse.ArgumentParser(
        description="Create and upload tar.gz archives for Handy"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Create archives and show sizes, don't upload or delete"
    )
    parser.add_argument(
        "--output-dir", default="/tmp",
        help="Directory for temporary tar.gz files (default: /tmp)"
    )
    parser.add_argument(
        "--only", default=None,
        help="Only process this archive name (e.g. qwen3-asr-0.6b-int4)"
    )
    args = parser.parse_args()

    results = []

    for archive in ARCHIVES:
        name = archive["name"]
        if args.only and name != args.only:
            continue

        filename = f"{name}.tar.gz"
        print(f"\n{'='*60}")
        print(f"Archive: {filename}")

        tar_path = create_tar(archive, args.output_dir)
        if tar_path is None:
            print("  FAILED")
            continue

        size_mb = os.path.getsize(tar_path) / 1e6
        results.append((filename, size_mb, archive["repo"]))

        if not args.dry_run:
            upload_tar(tar_path, archive["repo"], filename)
            os.remove(tar_path)
            print(f"  Deleted local tar")
        else:
            print(f"  Dry run — skipping upload")

    print(f"\n{'='*60}")
    print("Summary:")
    for filename, size_mb, repo in results:
        print(f"  {filename}: {size_mb:.1f} MB → {repo}")


if __name__ == "__main__":
    exit(main())
