#!/usr/bin/env python3
"""Validate complete FP32/int4 releases before replacing published artifacts."""

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from share_weights import share_external_models
from src.artifact_metadata import MODEL_METADATA_FILES
from src.artifact_publication import ArtifactStaging, publish_artifacts
from src.artifact_sources import paths_overlap, validate_publication_sources
from src.embeddings import load_embed_tokens
from src.onnx_artifacts import graph_file_closure, transport_graph_files
from src.release_validation import REQUIRED_METADATA, validate_release, validate_release_metadata
from src.release_validation import graph_names as _graph_names
from src.tokenizer_assets import prepare_tokenizer_assets, tokenizer_asset_changes, tokenizer_asset_names


def copy_file(src, dst, hardlink=False):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink and not src.is_symlink():
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def _create_archive(root, target, names, archive_root):
    with tarfile.open(target, "w:gz", compresslevel=1, dereference=True) as archive:
        for name in sorted(names):
            archive.add(root / name, arcname=f"{archive_root}/{name}", recursive=False)


def _validate_archive_target(target, archive_root, variant):
    if not target.exists() and not target.is_symlink():
        return
    if target.is_symlink() or not target.is_file() or target.stat().st_nlink > 1:
        raise ValueError(f"Archive destination is not an owned regular archive: {target}")
    required = set(REQUIRED_METADATA) | {"embed_tokens.bin", *_graph_names(variant)}
    seen = set()
    try:
        with tarfile.open(target, "r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.name
                    or not path.parts
                    or path.parts[0] != archive_root
                    or len(path.parts) < 2
                ):
                    raise ValueError("Unexpected archive member")
                name = path.relative_to(archive_root).as_posix()
                if name in seen:
                    raise ValueError("Duplicate archive member")
                seen.add(name)
                if name in MODEL_METADATA_FILES:
                    if member.size > 1024 * 1024:
                        raise ValueError("Archive metadata exceeds 1 MiB")
                    stream = archive.extractfile(member)
                    if stream is None or not isinstance(json.loads(stream.read()), dict):
                        raise ValueError("Archive metadata must be a JSON object")
        if not required <= seen:
            raise ValueError("Archive lacks required model artifacts")
    except (OSError, tarfile.TarError, ValueError) as error:
        raise ValueError(f"Unrecognized archive destination {target}; preserve it or choose another output") from error


def _package(input_dir, output_dir, test_wavs_src=None, hardlink=False):
    output = Path(output_dir).absolute()
    if output.is_symlink():
        raise ValueError("Output cannot be a symlink")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output must be a directory: {output}")
    with prepare_tokenizer_assets(input_dir, output) as tokenizer:
        if tokenizer is None:
            raise ValueError("Required tokenizer assets missing")
        return _package_prepared(input_dir, output_dir, test_wavs_src, hardlink, tokenizer)


def _package_prepared(input_dir, output_dir, test_wavs_src, hardlink, tokenizer):
    source = Path(input_dir).resolve()
    output = Path(output_dir).absolute()
    archives = {
        variant: output.parent / f"{output.name}{'-int4' if variant else ''}.tar.gz" for variant in ("", "int4")
    }
    for target in [output, *archives.values()]:
        if paths_overlap(source, target):
            raise ValueError(f"Input directory overlaps publication destination: {target}")
    if output.is_symlink():
        raise ValueError("Output cannot be a symlink")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output must be a directory: {output}")
    validate_release_metadata(source)
    variants = [variant for variant in ("", "int4") if any((source / name).exists() for name in _graph_names(variant))]
    if not variants:
        raise ValueError("No model variants found")
    for variant in variants:
        for name in _graph_names(variant):
            if not (source / name).is_file():
                raise ValueError(f"Incomplete {variant or 'FP32'} variant: {name}")
    for variant, target in archives.items():
        if variant not in variants:
            if target.exists() or target.is_symlink():
                raise ValueError(
                    f"Inactive variant archive already exists: {target}; preserve it and choose a clean output "
                    "or move it aside explicitly before retrying"
                )
        else:
            archive_root = output.name + ("-int4" if variant else "")
            _validate_archive_target(target, archive_root, variant)
    embed = load_embed_tokens(source, output_dtype="float16")
    sidecar = source / "embed_tokens.fp16.bin"
    if sidecar.exists():
        candidate = load_embed_tokens(source, filename=sidecar.name, dtype="float16", output_dtype="float16")
        if not (candidate == embed).all():
            raise ValueError("FP16 embedding sidecar disagrees with canonical data")
        embed = candidate
    rewritten = {"config.json", "embed_tokens.bin"}
    rewritten.update(name for variant in variants for name in _graph_names(variant)[1:])
    if test_wavs_src is not None:
        rewritten.add("test_wavs/0.wav")
    rewritten.update(tokenizer_asset_changes(source, tokenizer.root))
    graph_files = transport_graph_files(
        source,
        (name for variant in variants for name in _graph_names(variant)),
        rewritten=rewritten,
    )
    for variant in variants:
        canonical = f"decoder_weights.{variant}.data" if variant else "decoder_weights.data"
        unaffected_graphs = ["encoder" + (f".{variant}" if variant else "") + ".onnx"]
        unaffected_graphs.extend(name for other in variants if other != variant for name in _graph_names(other))
        if canonical in graph_file_closure(source, unaffected_graphs):
            raise ValueError(f"ONNX transport destination conflict: {canonical}")
    metadata = set(MODEL_METADATA_FILES) | tokenizer.names
    read_names = set(MODEL_METADATA_FILES) | tokenizer_asset_names(source) | graph_files | {"embed_tokens.bin"}
    if sidecar.exists():
        read_names.add(sidecar.name)
    read_roots = {Path(input_dir).absolute(), source}
    source_reads = {root / name for root in read_roots for name in read_names} | read_roots | set(tokenizer.reads)
    if test_wavs_src is not None:
        source_reads.add(Path(test_wavs_src))
    validate_publication_sources(source_reads, files=archives.values(), directories=[output])
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactStaging(prefix=".package-", dir=output.parent) as staging:
        stage = staging.path
        release = stage / output.name
        release.mkdir()
        names = set(metadata)
        names.update(graph_files)
        for name in sorted(names):
            asset_root = tokenizer.root if name in tokenizer.names else source
            copy_file(asset_root / name, release / name, hardlink=hardlink and name in metadata)
        embed.tofile(release / "embed_tokens.bin")
        config = json.loads((release / "config.json").read_text())
        config["embed_tokens_dtype"] = "float16"
        config["embed_tokens_shape"] = list(embed.shape)
        config_path = release / "config.json"
        config_path.unlink()
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        for variant in variants:
            share_external_models(release, suffix=variant or None)
        shared = metadata | {"embed_tokens.bin"}
        if test_wavs_src is not None:
            copy_file(test_wavs_src, release / "test_wavs" / "0.wav")
            shared.add("test_wavs/0.wav")
        variant_files = {variant: graph_file_closure(release, _graph_names(variant)) for variant in variants}
        complete = shared | set().union(*variant_files.values())
        for path in release.rglob("*"):
            if path.is_file() and path.relative_to(release).as_posix() not in complete:
                path.unlink()
        validate_release(release, (name for variant in variants for name in _graph_names(variant)))
        replacements = [(release, output)]
        for variant in variants:
            base = output.name + ("-int4" if variant else "")
            archive_path = stage / f"{base}.tar.gz"
            members = shared | variant_files[variant]
            _create_archive(release, archive_path, members, base)
            replacements.append((archive_path, archives[variant]))
        publish_artifacts(replacements)
        staging.mark_complete()
    print(f"Release directory: {output}")
    return True


def package(input_dir, output_dir, test_wavs_src=None, hardlink=False):
    try:
        return _package(input_dir, output_dir, test_wavs_src=test_wavs_src, hardlink=hardlink)
    except Exception as error:
        print(f"ERROR: {error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Package complete FP32 + int4 model releases")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=["0.6b", "1.7b"])
    group.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-wavs", default=None)
    parser.add_argument("--hardlink", action="store_true")
    args = parser.parse_args()
    input_dir = args.input or f"output/qwen3-asr-{args.model}"
    return 0 if package(input_dir, args.output, test_wavs_src=args.test_wavs, hardlink=args.hardlink) else 1


if __name__ == "__main__":
    raise SystemExit(main())
