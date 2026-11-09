import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import onnx
import pytest

import package as pack
from tests.artifact_fixtures import _ort_result, _release_input, _write_external_identity, _write_model
from tests.converter_fixtures import bytes_under
from tests.tokenizer_fixtures import save_tokenizer_assets


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_packaging_preserves_embedding_values_and_can_repeat(tmp_path, dtype):
    source = _release_input(tmp_path / "source", dtype)
    output = tmp_path / "release"
    assert pack.package(str(source), str(output))
    values = np.fromfile(output / "embed_tokens.bin", dtype=np.float16).reshape(8, 4)
    np.testing.assert_array_equal(values, np.arange(32).reshape(8, 4))
    assert pack.package(str(output), str(tmp_path / "repeated"))
    assert (tmp_path / "repeated" / "embed_tokens.bin").read_bytes() == (output / "embed_tokens.bin").read_bytes()


@pytest.mark.parametrize("missing", ["decoder_step.onnx", "encoder.onnx", "tokenizer.json", "embed_tokens.bin"])
def test_incomplete_release_preserves_existing_output(tmp_path, missing):
    source = _release_input(tmp_path / "source")
    (source / missing).unlink()
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_bytes(b"previous release")
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_bytes() == b"previous release"


def test_partial_int4_variant_is_rejected(tmp_path):
    source = _release_input(tmp_path / "source")
    _write_model(source, "decoder_init.int4.onnx")
    assert not pack.package(str(source), str(tmp_path / "release"))


@pytest.mark.parametrize("layout", ["inline", "per-decoder", "shared", "custom"])
def test_archives_include_actual_graph_dependencies(tmp_path, layout):
    source = _release_input(tmp_path / "source", external=layout == "per-decoder")
    if layout == "shared":
        from share_weights import share_external_models

        share_external_models(source)
    if layout == "custom":
        for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, external=True, data_name=f"custom-{name}.bin")
    for name in ("encoder.int4.onnx", "decoder_init.int4.onnx", "decoder_step.int4.onnx"):
        _write_model(source, name, external=True)
    output = tmp_path / "release"
    assert pack.package(str(source), str(output))
    for suffix in ("", "-int4"):
        archive_path = tmp_path / f"release{suffix}.tar.gz"
        extracted = tmp_path / f"extracted{suffix}"
        with tarfile.open(archive_path) as archive:
            archive.extractall(extracted, filter="data")
        for model in extracted.rglob("*.onnx"):
            np.testing.assert_array_equal(_ort_result(model), _ort_result(source / model.name))
        assert (extracted / f"release{suffix}" / "embed_tokens.bin").stat().st_size == 64
    assert not list(output.glob("decoder_*.onnx.data"))
    assert not list(output.glob("custom-decoder*"))


def test_truncated_external_dependency_preserves_output(tmp_path):
    source = _release_input(tmp_path / "source", external=True)
    (source / "encoder.onnx.data").write_bytes(b"short")
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("old")
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_text() == "old"


def test_archive_failure_preserves_previous_release(tmp_path, monkeypatch):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "release"
    assert pack.package(source, output)
    archive = tmp_path / "release.tar.gz"
    archive_bytes = archive.read_bytes()
    previous = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    reached = []

    def fail(root, target, names, archive_root):
        reached.append(archive_root)
        raise OSError("injected archive failure")

    monkeypatch.setattr(pack, "_create_archive", fail)
    assert not pack.package(str(source), str(output))
    assert reached == ["release"]
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == previous
    assert archive.read_bytes() == archive_bytes


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_converter_cli_honors_declared_storage_dtype(tmp_path, dtype):
    source = _release_input(tmp_path / "source", dtype)
    result = subprocess.run(
        [sys.executable, str(Path(pack.__file__).with_name("convert_embed_fp16.py")), "--model-dir", str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    values = np.fromfile(source / "embed_tokens.fp16.bin", dtype=np.float16)
    np.testing.assert_array_equal(values, np.arange(32, dtype=np.float16))


@pytest.mark.parametrize("problem", ["shape", "dtype", "truncated", "nan", "overflow"])
def test_bad_embeddings_are_rejected_without_replacing_output(tmp_path, problem):
    source = _release_input(tmp_path / "source")
    config = json.loads((source / "config.json").read_text())
    if problem == "shape":
        config["embed_tokens_shape"] = [4, 8]
    elif problem == "dtype":
        config["embed_tokens_dtype"] = "bfloat16"
    elif problem == "truncated":
        (source / "embed_tokens.bin").write_bytes(b"short")
    else:
        np.full((8, 4), np.nan if problem == "nan" else 1e20, dtype=np.float32).tofile(source / "embed_tokens.bin")
    (source / "config.json").write_text(json.dumps(config))
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("old")
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_text() == "old"


def test_legacy_metadata_requires_unambiguous_fp32_bytes(tmp_path):
    from src.embeddings import load_embed_tokens

    source = _release_input(tmp_path / "source")
    (source / "config.json").write_text(json.dumps({"decoder": {"vocab_size": 8, "hidden_size": 4}}))
    np.testing.assert_array_equal(load_embed_tokens(source), np.arange(32).reshape(8, 4))
    np.arange(32, dtype=np.float16).tofile(source / "embed_tokens.bin")
    with pytest.raises(ValueError, match="byte count"):
        load_embed_tokens(source)


def test_publication_failure_restores_release_and_archives(tmp_path, monkeypatch):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "release"
    assert pack.package(source, output)
    previous_archive = tmp_path / "release.tar.gz"
    archive_bytes = previous_archive.read_bytes()
    previous = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    original_replace = pack.os.replace
    reached = []

    def injected(source, target):
        if Path(target) == previous_archive and ".package-" in str(source):
            reached.append(target)
            raise OSError("injected archive publication failure")
        original_replace(source, target)

    monkeypatch.setattr(pack.os, "replace", injected)
    assert not pack.package(str(source), str(output))
    assert reached == [previous_archive]
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == previous
    assert previous_archive.read_bytes() == archive_bytes


def test_converter_rejects_overflow_and_preserves_existing_sidecar(tmp_path):
    source = _release_input(tmp_path / "source")
    np.full((8, 4), 1e20, dtype=np.float32).tofile(source / "embed_tokens.bin")
    destination = source / "embed_tokens.fp16.bin"
    destination.write_bytes(b"previous validated conversion")
    result = subprocess.run(
        [sys.executable, str(Path(pack.__file__).with_name("convert_embed_fp16.py")), "--model-dir", str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert destination.read_bytes() == b"previous validated conversion"


def test_hardlinked_metadata_does_not_mutate_source(tmp_path):
    source = _release_input(tmp_path / "source")
    before = (source / "config.json").read_bytes()
    assert pack.package(str(source), str(tmp_path / "release"), hardlink=True)
    assert (source / "config.json").read_bytes() == before
    assert json.loads((tmp_path / "release" / "config.json").read_text())["embed_tokens_dtype"] == "float16"


def test_complete_int4_only_release_can_be_repackaged(tmp_path):
    source = _release_input(tmp_path / "source", "float16")
    for role in ("encoder", "decoder_init", "decoder_step"):
        (source / f"{role}.onnx").unlink()
        _write_model(source, f"{role}.int4.onnx", external=True)
    output = tmp_path / "release"
    assert pack.package(str(source), str(output))
    assert not (tmp_path / "release.tar.gz").exists()
    with tarfile.open(tmp_path / "release-int4.tar.gz") as archive:
        archive.extractall(tmp_path / "extracted", filter="data")
    extracted = tmp_path / "extracted" / "release-int4"
    assert pack.package(str(extracted), str(tmp_path / "repeated"))
    for model in (tmp_path / "repeated").glob("*.onnx"):
        np.testing.assert_array_equal(_ort_result(model), _ort_result(source / model.name))
    assert (tmp_path / "repeated" / "embed_tokens.bin").read_bytes() == (source / "embed_tokens.bin").read_bytes()


@pytest.mark.parametrize("name", ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"])
@pytest.mark.parametrize("contents", ['{"truncated":', "[]"])
def test_required_metadata_must_be_valid_json_objects(tmp_path, name, contents):
    source = _release_input(tmp_path / "source")
    (source / name).write_text(contents)
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("previous release")
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_text() == "previous release"
    assert not (tmp_path / "release.tar.gz").exists()


@pytest.mark.parametrize("template_file", ["chat_template.jinja", "chat_template.json"])
def test_packaged_chat_template_reloads_offline(tmp_path, template_file, monkeypatch):
    from src.prompt import load_prompt_tokenizer

    source = _release_input(tmp_path / "source")
    template_format = "json" if template_file.endswith(".json") else "jinja"
    tokenizer = save_tokenizer_assets(source, template_format=template_format)
    template_bytes = (source / template_file).read_bytes()
    messages = [
        {"role": "system", "content": "Kepler"},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    expected = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    output = tmp_path / "release"
    assert pack.package(str(source), str(output))
    with tarfile.open(tmp_path / "release.tar.gz") as archive:
        archive.extractall(tmp_path / "extracted", filter="data")
    packaged = tmp_path / "extracted" / "release"
    assert (packaged / template_file).read_bytes() == template_bytes
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    offline = load_prompt_tokenizer(str(packaged))
    assert offline.chat_template == tokenizer.chat_template
    assert offline.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) == expected


@pytest.mark.parametrize("source_name", ["release.tar.gz", "release-int4.tar.gz", "release.tar.gz/input"])
def test_publication_destinations_cannot_overlap_source(tmp_path, source_name):
    source = tmp_path / source_name
    source.parent.mkdir(parents=True, exist_ok=True)
    _release_input(source)
    original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("old release")
    assert not pack.package(str(source), str(output))
    assert source.is_dir()
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == original
    assert (output / "sentinel").read_text() == "old release"


@pytest.mark.parametrize("problem", ["missing-template", "invalid-template", "invalid-tokenizer"])
def test_unusable_tokenizer_contract_preserves_previous_release(tmp_path, problem, monkeypatch):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "release"
    assert pack.package(source, output)
    archive = tmp_path / "release.tar.gz"
    archive_bytes = archive.read_bytes()
    previous = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    if problem == "missing-template":
        for name in ("chat_template.jinja", "chat_template.json"):
            (source / name).unlink(missing_ok=True)
        config = json.loads((source / "tokenizer_config.json").read_text())
        config.pop("chat_template", None)
        (source / "tokenizer_config.json").write_text(json.dumps(config))
    elif problem == "invalid-template":
        (source / "chat_template.jinja").write_text("{{ messages[0].content }}")
    else:
        (source / "tokenizer.json").write_text("{}")
    from src import tokenizer_assets

    original_source = bytes_under(source)
    validate = tokenizer_assets.validate_tokenizer_assets
    reached = []

    def validate_and_record(root):
        reached.append(Path(root).name)
        return validate(root)

    monkeypatch.setattr(tokenizer_assets, "validate_tokenizer_assets", validate_and_record)
    assert not pack.package(str(source), str(output))
    assert reached == ["source"]
    assert bytes_under(source) == original_source
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == previous
    assert archive.read_bytes() == archive_bytes


def _symlink_metadata_to_cache(source, cache):
    cache.mkdir()
    names = {"config.json", "preprocessor_config.json"} | pack.tokenizer_asset_names(source)
    for name in names:
        path = source / name
        target = cache / name
        path.rename(target)
        path.symlink_to(os.path.relpath(target, path.parent))
    return names


@pytest.mark.parametrize("hardlink", [False, True])
def test_packaged_cache_symlinks_are_portable_offline(tmp_path, hardlink):
    from src.prompt import build_prompt_ids, load_prompt_tokenizer

    source = _release_input(tmp_path / "source")
    cache = tmp_path / "hf-cache"
    names = _symlink_metadata_to_cache(source, cache)
    expected = build_prompt_ids(3, load_prompt_tokenizer(str(source), local_files_only=True), context="Kepler")
    output = tmp_path / "release"
    assert pack.package(source, output, hardlink=hardlink)
    assert all(not (output / name).is_symlink() for name in names)
    shutil.rmtree(source)
    shutil.rmtree(cache)
    with tarfile.open(tmp_path / "release.tar.gz") as archive:
        assert all(member.isfile() for member in archive.getmembers())
        archive.extractall(tmp_path / "extracted", filter="data")
    for path in (output, tmp_path / "extracted" / "release"):
        tokenizer = load_prompt_tokenizer(str(path), local_files_only=True)
        assert build_prompt_ids(3, tokenizer, context="Kepler") == expected


def test_archive_cache_symlinks_are_materialized(tmp_path):
    from src.prompt import build_prompt_ids, load_prompt_tokenizer

    source = _release_input(tmp_path / "source")
    cache = tmp_path / "hf-cache"
    names = _symlink_metadata_to_cache(source, cache)
    expected = build_prompt_ids(3, load_prompt_tokenizer(str(source), local_files_only=True), context="Kepler")
    archive_path = tmp_path / "portable.tar.gz"
    pack._create_archive(source, archive_path, names | {"embed_tokens.bin"}, "portable")
    shutil.rmtree(source)
    shutil.rmtree(cache)
    with tarfile.open(archive_path) as archive:
        assert all(member.isfile() for member in archive.getmembers())
        archive.extractall(tmp_path / "extracted", filter="data")
    tokenizer = load_prompt_tokenizer(str(tmp_path / "extracted" / "portable"), local_files_only=True)
    assert build_prompt_ids(3, tokenizer, context="Kepler") == expected


def test_standalone_archive_includes_external_encoder_closure(tmp_path):
    from create_tar import create_tar

    source = _release_input(tmp_path / "source", external=True)
    assert pack.package(str(source), str(tmp_path / "release"))
    release = tmp_path / "release"
    archive = {
        "name": "bundle",
        "source_dir": str(release),
        "files": ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx", "decoder_weights.data"],
    }
    path = create_tar(archive, str(tmp_path))
    assert path is not None
    with tarfile.open(path) as tar:
        assert "bundle/encoder.onnx.data" in tar.getnames()
        tar.extractall(tmp_path / "extracted", filter="data")
    onnx.load(tmp_path / "extracted" / "bundle" / "encoder.onnx")


def test_standalone_archive_failure_preserves_prior_bytes(tmp_path, monkeypatch):
    from create_tar import create_tar

    source = _release_input(tmp_path / "source", external=True)
    assert pack.package(str(source), str(tmp_path / "release"))
    archive = {
        "name": "bundle",
        "source_dir": str(tmp_path / "release"),
        "files": ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx", "decoder_weights.data"],
    }
    destination = tmp_path / "bundle.tar.gz"
    destination.write_bytes(b"retained archive")
    original_add = tarfile.TarFile.add
    count = 0

    def interrupt(self, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected archive write failure")
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", interrupt)
    assert create_tar(archive, str(tmp_path)) is None
    assert destination.read_bytes() == b"retained archive"


def test_package_rejects_graph_dependency_on_mutated_metadata(tmp_path):
    source = _release_input(tmp_path / "source")
    _write_model(source, "encoder.onnx", weight=np.ones((1, 1), dtype=np.float32))
    model = onnx.load(source / "encoder.onnx")
    tensor = model.graph.initializer[0]
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", "config.json"), ("offset", "0"), ("length", "4")):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    onnx.save_model(model, source / "encoder.onnx")
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_bytes(b"prior release")
    assert not pack.package(str(source), str(output))
    assert (output / "sentinel").read_bytes() == b"prior release"


def test_package_rejects_encoder_dependency_on_decoder_canonical_data(tmp_path, capsys):
    source = _release_input(tmp_path / "source")
    _write_model(source, "encoder.onnx", external=True, data_name="decoder_weights.data")
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_text("previous release")
    assert not pack.package(str(source), str(output))
    assert "destination conflict: decoder_weights.data" in capsys.readouterr().out
    assert (output / "sentinel").read_text() == "previous release"
    assert not (tmp_path / "release.tar.gz").exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_absent_archive_is_preserved(tmp_path, kind):
    source = _release_input(tmp_path / "source")
    target = tmp_path / "release-int4.tar.gz"
    if kind == "directory":
        target.mkdir()
        sentinel = target / "foreign"
    else:
        sentinel = target
    sentinel.write_bytes(b"foreign bytes")
    assert not pack.package(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()
    assert sentinel.read_bytes() == b"foreign bytes"


def test_emitted_archive_directory_is_rejected_before_publication(tmp_path):
    source = _release_input(tmp_path / "source")
    target = tmp_path / "release.tar.gz"
    target.mkdir()
    (target / "foreign").write_bytes(b"foreign bytes")
    assert not pack.package(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()
    assert (target / "foreign").read_bytes() == b"foreign bytes"


@pytest.mark.parametrize("kind", ["plain-file", "symlink", "foreign-tar", "tar-link", "large-metadata"])
def test_unowned_archive_is_rejected_before_publication(tmp_path, kind):
    import io

    source = _release_input(tmp_path / "source")
    target = tmp_path / "release.tar.gz"
    retained = tmp_path / "retained"
    retained.write_bytes(b"foreign bytes")
    if kind == "plain-file":
        target.write_bytes(b"foreign bytes")
    elif kind == "symlink":
        target.symlink_to(retained)
    else:
        with tarfile.open(target, "w:gz") as archive:
            entry = tarfile.TarInfo("release/config.json" if kind == "large-metadata" else "foreign/file")
            if kind == "tar-link":
                entry.type = tarfile.SYMTYPE
                entry.linkname = str(retained)
                archive.addfile(entry)
            else:
                data = b" " * (1024 * 1024 + 1) if kind == "large-metadata" else b"foreign"
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    before = target.read_bytes()
    assert not pack.package(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()
    assert target.read_bytes() == before
    assert retained.read_bytes() == b"foreign bytes"


def test_publication_can_replace_recognized_archive(tmp_path):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "release"
    assert pack.package(source, output)
    assert pack.package(source, output)
    for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"):
        np.testing.assert_array_equal(_ort_result(source / name), _ort_result(output / name))


def test_package_rejects_graphs_used_as_rewritten_tensor_storage(tmp_path):
    source = _release_input(tmp_path / "source")
    data = (source / "decoder_init.onnx").read_bytes()
    offset = data.index(np.arange(512, dtype=np.float32).tobytes())
    _write_external_identity(source, "encoder.onnx", "decoder_init.onnx", offset)
    np.testing.assert_array_equal(_ort_result(source / "encoder.onnx"), [0])
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "release"
    output.mkdir()
    (output / "sentinel").write_bytes(b"previous release")
    assert not pack.package(source, output)
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    assert (output / "sentinel").read_bytes() == b"previous release"


def test_package_rejects_audio_destination_used_as_tensor_storage(tmp_path):
    source = _release_input(tmp_path / "source")
    (source / "test_wavs").mkdir()
    np.ones(1, dtype=np.float32).tofile(source / "test_wavs" / "0.wav")
    _write_external_identity(source, "encoder.onnx", "test_wavs/0.wav")
    np.testing.assert_array_equal(_ort_result(source / "encoder.onnx"), [1])
    replacement = tmp_path / "replacement.wav"
    np.full(1, 7, dtype=np.float32).tofile(replacement)
    assert not pack.package(source, tmp_path / "release", test_wavs_src=replacement)
    assert not (tmp_path / "release").exists()
    np.testing.assert_array_equal(_ort_result(source / "encoder.onnx"), [1])


@pytest.mark.parametrize("layout", ["inline-step", "external-int64"])
def test_existing_shared_layout_is_preserved_on_share_and_package(tmp_path, layout):
    import share_weights as share

    source = _release_input(tmp_path / "source")
    assert share.share_external_models(source)
    if layout == "inline-step":
        _write_model(source, "decoder_step.onnx", weight=np.ones((32, 16), dtype=np.float32))
    else:
        canonical = source / "decoder_weights.data"
        offset = canonical.stat().st_size
        with canonical.open("ab") as stream:
            stream.write(np.full(256, 9, dtype=np.int64).tobytes())
        _write_external_identity(source, "decoder_step.onnx", canonical.name, offset, np.int64)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    expected = {name: _ort_result(source / name) for name in share._decoder_filenames()}
    if layout == "inline-step":
        with pytest.raises(ValueError, match="Canonical shared data has incompatible bytes"):
            share.share_external_models(source)
        assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
        assert not pack.package(source, tmp_path / "release")
        for name, values in expected.items():
            np.testing.assert_array_equal(_ort_result(source / name), values)
        return
    assert share.share_external_models(source)
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    output = tmp_path / "release"
    assert pack.package(source, output)
    for name, values in expected.items():
        np.testing.assert_array_equal(_ort_result(output / name), values)


@pytest.mark.parametrize("entrypoint", ["helper", "cli"])
def test_package_preserves_existing_regular_file_output(tmp_path, entrypoint):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "release"
    output.write_bytes(b"independent retained user file")
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    if entrypoint == "helper":
        assert not pack.package(source, output)
    else:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "package.py"),
                "--input",
                str(source),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode != 0, result.stdout + result.stderr
    assert output.is_file()
    assert output.read_bytes() == b"independent retained user file"
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == before
    assert not (tmp_path / "release.tar.gz").exists()
    assert not (tmp_path / "release-int4.tar.gz").exists()
