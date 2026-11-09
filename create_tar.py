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
import tarfile
from pathlib import Path

from huggingface_hub import HfApi

from src.artifact_publication import ArtifactStaging
from src.artifact_sources import validate_publication_sources
from src.onnx_artifacts import inventory_graphs, validate_generated_destinations
from src.release_validation import validate_release
from src.tokenizer_assets import prepare_tokenizer_assets

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
        ],
    },
    {
        "name": "qwen3-asr-0.6b-int4",
        "source_dir": "release/qwen3-asr-0.6b",
        "repo": "andrewleech/qwen3-asr-0.6b-onnx",
        "files": [
            "encoder.int4.onnx",
            "decoder_init.int4.onnx",
            "decoder_step.int4.onnx",
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
        ],
    },
    {
        "name": "qwen3-asr-1.7b-int4",
        "source_dir": "release/qwen3-asr-1.7b",
        "repo": "andrewleech/qwen3-asr-1.7b-onnx",
        "files": [
            "encoder.int4.onnx",
            "decoder_init.int4.onnx",
            "decoder_step.int4.onnx",
        ],
    },
]


def _archive_destination(source, output, name, members):
    if not isinstance(name, str) or not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        raise ValueError("Archive name must be a single safe path component")
    source, output = Path(source).resolve(), Path(output).resolve()
    filename = f"{name}.tar.gz"
    validate_generated_destinations(output, [filename])
    target = output / filename
    graphs = inventory_graphs(source)
    protected_names = set(members) | graphs.keys() | {name for names in graphs.values() for name in names}
    protected = {source / name for name in protected_names}
    if target in protected | {path.resolve() for path in protected} or (
        target.exists() and any(target.samefile(path) for path in protected)
    ):
        raise ValueError(f"Archive destination would replace a source artifact: {target}")
    return target


def create_tar(archive, output_dir):
    try:
        with (
            ArtifactStaging(prefix="qwen-archive-family-", dir=None) as family,
            prepare_tokenizer_assets(archive["source_dir"], family.path) as tokenizer,
        ):
            if tokenizer is None:
                raise ValueError("Required tokenizer assets missing")
            result = _create_prepared_tar(archive, output_dir, tokenizer)
            if result is not None:
                family.mark_complete()
            return result
    except Exception as error:
        print(f"  ERROR: Invalid release: {error}")
        return None


def _create_prepared_tar(archive, output_dir, tokenizer):
    name = archive["name"]
    source_dir = Path(archive["source_dir"]).resolve()
    graphs = {filename for filename in archive["files"] if filename.endswith(".onnx")}
    try:
        all_files = validate_release(source_dir, graphs, tokenizer_root=tokenizer.root)
        source_files = {filename for filename in all_files if (source_dir / filename).is_file()}
        tar_path = _archive_destination(source_dir, output_dir, name, source_files)
        original_reads = {Path(archive["source_dir"]) / filename for filename in source_files} | set(tokenizer.reads)
        validate_publication_sources(original_reads, files=[tar_path])
    except Exception as error:
        print(f"  ERROR: Invalid release: {error}")
        return None
    print(f"  Creating {tar_path}...")
    try:
        with ArtifactStaging(prefix=".archive-", dir=tar_path.parent) as staging:
            staged_tar = staging.path / tar_path.name
            with tarfile.open(staged_tar, "w:gz", compresslevel=1, dereference=True) as target:
                for filename in sorted(all_files):
                    member_root = tokenizer.root if filename in tokenizer.names else source_dir
                    target.add(member_root / filename, arcname=f"{name}/{filename}", recursive=False)
            os.replace(staged_tar, tar_path)
            staging.mark_complete()
    except (OSError, tarfile.TarError) as error:
        print(f"  ERROR: Archive creation failed: {error}")
        return None

    size = os.path.getsize(tar_path)
    print(f"  Size: {size / 1e6:.1f} MB ({size / 1e9:.2f} GB)")
    return str(tar_path)


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
    print("  Upload complete")


def main():
    parser = argparse.ArgumentParser(description="Create and upload tar.gz archives for Handy")
    parser.add_argument("--dry-run", action="store_true", help="Create archives and show sizes, don't upload or delete")
    parser.add_argument("--output-dir", default="/tmp", help="Directory for temporary tar.gz files (default: /tmp)")
    parser.add_argument("--only", default=None, help="Only process this archive name (e.g. qwen3-asr-0.6b-int4)")
    args = parser.parse_args()

    results = []
    failed = False

    for archive in ARCHIVES:
        name = archive["name"]
        if args.only and name != args.only:
            continue

        filename = f"{name}.tar.gz"
        print(f"\n{'=' * 60}")
        print(f"Archive: {filename}")

        tar_path = create_tar(archive, args.output_dir)
        if tar_path is None:
            failed = True
            print("  FAILED")
            continue

        size_mb = os.path.getsize(tar_path) / 1e6
        results.append((filename, size_mb, archive["repo"]))

        if not args.dry_run:
            upload_tar(tar_path, archive["repo"], filename)
            os.remove(tar_path)
            print("  Deleted local tar")
        else:
            print("  Dry run — skipping upload")

    print(f"\n{'=' * 60}")
    print("Summary:")
    for filename, size_mb, repo in results:
        print(f"  {filename}: {size_mb:.1f} MB → {repo}")

    return 1 if failed else 0


if __name__ == "__main__":
    exit(main())
