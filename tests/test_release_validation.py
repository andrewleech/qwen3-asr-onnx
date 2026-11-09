import json
import os
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

import create_tar
from src.release_validation import graph_names, validate_release
from tests.artifact_fixtures import _ort_result, _release_input, _write_external_identity, _write_model


def _archive(source):
    return {"name": "model", "source_dir": str(source), "repo": "unused/local", "files": graph_names()}


@pytest.mark.parametrize(
    "problem",
    [
        "short-embedding",
        "nonfinite-embedding",
        "config-shape",
        "missing-preprocessor",
        "bad-preprocessor",
        "incomplete-decoder",
        "invalid-graph",
        "runtime-graph",
    ],
)
def test_invalid_release_cannot_replace_archive_or_upload(tmp_path, monkeypatch, problem):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "archives"
    output.mkdir()
    previous = output / "model.tar.gz"
    previous.write_bytes(b"previous archive")
    if problem == "short-embedding":
        (source / "embed_tokens.bin").write_bytes(b"x")
    elif problem == "nonfinite-embedding":
        values = np.arange(32, dtype=np.float32)
        values[3] = np.nan
        values.tofile(source / "embed_tokens.bin")
    elif problem == "config-shape":
        config = json.loads((source / "config.json").read_text())
        config["embed_tokens_shape"] = [4, 8]
        (source / "config.json").write_text(json.dumps(config))
    elif problem == "missing-preprocessor":
        (source / "preprocessor_config.json").unlink()
    elif problem == "bad-preprocessor":
        (source / "preprocessor_config.json").write_text("[]")
    elif problem == "incomplete-decoder":
        (source / "decoder_step.onnx").unlink()
    elif problem == "invalid-graph":
        (source / "decoder_step.onnx").write_bytes(b"invalid ONNX")
    else:
        import onnx

        graph = onnx.load(source / "decoder_step.onnx")
        graph.graph.node[0].domain = "unavailable.backend"
        graph.opset_import.add(domain="unavailable.backend", version=1)
        onnx.save(graph, source / "decoder_step.onnx")
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    archive = _archive(source)
    assert create_tar.create_tar(archive, output) is None
    monkeypatch.setattr(create_tar, "ARCHIVES", [archive])
    uploads = []
    monkeypatch.setattr(create_tar, "upload_tar", lambda *args: uploads.append(args))
    monkeypatch.setattr(sys, "argv", ["create_tar.py", "--output-dir", str(output)])
    assert create_tar.main() == 1
    assert uploads == []
    assert previous.read_bytes() == b"previous archive"
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("layout", ["inline", "separate", "shared"])
def test_archive_uses_actual_graph_dependencies(tmp_path, layout):
    source = _release_input(tmp_path / "source")
    for name in graph_names():
        _write_model(source, name, external=layout != "inline", data_name="shared.data" if layout == "shared" else None)
    output = tmp_path / "archives"
    output.mkdir()
    path = create_tar.create_tar(_archive(source), output)
    assert path is not None
    with tarfile.open(path) as archive:
        actual = {member.name.removeprefix("model/") for member in archive.getmembers()}
    expected = set(graph_names()) | {
        "config.json",
        "preprocessor_config.json",
        "embed_tokens.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    }
    if layout == "shared":
        expected.add("shared.data")
    elif layout == "separate":
        expected.update(f"{name}.data" for name in graph_names())
    assert actual == expected
    assert actual == validate_release(source, graph_names())


def test_archive_refuses_incomplete_selected_graph_family(tmp_path):
    source = _release_input(tmp_path / "source")
    with pytest.raises(ValueError, match="complete encoder"):
        validate_release(source, ["encoder.onnx", "decoder_init.onnx"])


def test_archive_destination_cannot_replace_graph_storage(tmp_path):
    source = _release_input(tmp_path / "source")
    _write_model(source, "encoder.onnx", external=True, data_name="model.tar.gz")
    original = (source / "model.tar.gz").read_bytes()
    assert create_tar.create_tar(_archive(source), source) is None
    assert (source / "model.tar.gz").read_bytes() == original
    validate_release(source, graph_names())


@pytest.mark.parametrize("directory_alias", [False, True])
def test_archive_container_ignores_unrelated_broken_models(tmp_path, directory_alias):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "archives"
    output.mkdir()
    foreign = output / "unrelated"
    foreign.mkdir()
    _write_model(foreign, "unrelated.onnx", external=True)
    (foreign / "unrelated.onnx.data").unlink()
    previous = (foreign / "unrelated.onnx").read_bytes()
    if directory_alias:
        alias = tmp_path / "archive-alias"
        alias.symlink_to(output, target_is_directory=True)
        output = alias
    path = create_tar.create_tar(_archive(source), output)
    assert path is not None
    with tarfile.open(path) as archive:
        members = {item.name.removeprefix("model/") for item in archive.getmembers()}
    assert members == validate_release(source, graph_names())
    assert (foreign / "unrelated.onnx").read_bytes() == previous
    assert not (foreign / "unrelated.onnx.data").exists()


def test_archive_cli_default_temporary_directory(tmp_path, monkeypatch):
    source = _release_input(tmp_path / "source")
    archive = _archive(source) | {"name": f"kepler-test-{uuid.uuid4().hex}"}
    target = Path("/tmp") / f"{archive['name']}.tar.gz"
    assert not target.exists() and not target.is_symlink()
    monkeypatch.setattr(create_tar, "ARCHIVES", [archive])
    monkeypatch.setattr(sys, "argv", ["create_tar.py", "--dry-run"])
    try:
        with tempfile.TemporaryDirectory(prefix="kepler-foreign-", dir="/tmp") as directory:
            foreign = Path(directory)
            _write_model(foreign, "unrelated.onnx", external=True)
            (foreign / "unrelated.onnx.data").unlink()
            previous = (foreign / "unrelated.onnx").read_bytes()
            assert create_tar.main() == 0
            with tarfile.open(target) as stored:
                assert f"{archive['name']}/encoder.onnx" in stored.getnames()
            assert (foreign / "unrelated.onnx").read_bytes() == previous
    finally:
        target.unlink(missing_ok=True)


def test_archive_preserves_physical_source_metadata_alias(tmp_path):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "archives"
    output.mkdir()
    target = output / "model.tar.gz"
    target.write_text("{}")
    metadata = source / "preprocessor_config.json"
    metadata.unlink()
    metadata.symlink_to(target)
    assert create_tar.create_tar(_archive(source), output) is None
    assert target.read_text() == "{}"
    validate_release(source, graph_names())


@pytest.mark.parametrize("nested_output", [False, True])
def test_archive_preserves_unselected_source_graph_storage(tmp_path, nested_output):
    source = _release_input(tmp_path / "source")
    output = source / "archives" if nested_output else source
    output.mkdir(exist_ok=True)
    relative = "archives/model.tar.gz" if nested_output else "model.tar.gz"
    target = source / relative
    target.write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(source, "aux.onnx", relative)
    assert create_tar.create_tar(_archive(source), output) is None
    np.testing.assert_array_equal(_ort_result(source / "aux.onnx"), [42])
    assert target.read_bytes() == np.array([42], np.float32).tobytes()


@pytest.mark.parametrize("kind", ["directory", "symlink", "dangling", "hardlink"])
def test_archive_refuses_nonregular_or_linked_output_leaf(tmp_path, kind):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "archives"
    output.mkdir()
    target = output / "model.tar.gz"
    retained = tmp_path / "retained"
    retained.write_bytes(b"retained bytes")
    if kind == "directory":
        target.mkdir()
        (target / "sentinel").write_bytes(b"directory bytes")
    elif kind == "symlink":
        target.symlink_to(retained)
    elif kind == "dangling":
        target.symlink_to(tmp_path / "missing")
    else:
        os.link(retained, target)
    assert create_tar.create_tar(_archive(source), output) is None
    assert retained.read_bytes() == b"retained bytes"
    if kind == "directory":
        assert (target / "sentinel").read_bytes() == b"directory bytes"
    elif kind in {"symlink", "dangling"}:
        assert target.is_symlink()


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "/absolute", "nested/name", "nested\\name"])
def test_archive_requires_a_safe_member_root_and_leaf(tmp_path, name):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "archives"
    output.mkdir()
    assert create_tar.create_tar(_archive(source) | {"name": name}, output) is None
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("kind", ["metadata", "unselected-graph"])
def test_archive_source_identity_uses_filesystem_case_semantics(tmp_path, kind):
    source = _release_input(tmp_path / "source")
    output = source / "archives" if kind == "unselected-graph" else tmp_path / "archives"
    output.mkdir()
    existing = output / "MODEL.tar.gz"
    target = output / "model.tar.gz"
    if kind == "metadata":
        existing.write_text("{}")
        metadata = source / "preprocessor_config.json"
        metadata.unlink()
        metadata.symlink_to(existing)
    else:
        existing.write_bytes(np.array([42], np.float32).tobytes())
        _write_external_identity(source, "aux.onnx", "archives/MODEL.tar.gz")
    previous = existing.read_bytes()
    aliases_source = target.exists() and target.samefile(existing)
    result = create_tar.create_tar(_archive(source), output)
    if aliases_source:
        assert result is None
    else:
        assert result is not None
        with tarfile.open(result) as stored:
            assert "model/encoder.onnx" in stored.getnames()
    assert existing.read_bytes() == previous
    validate_release(source, graph_names())
    if kind == "unselected-graph":
        np.testing.assert_array_equal(_ort_result(source / "aux.onnx"), [42])


def test_archive_can_replace_an_unrelated_regular_output(tmp_path):
    source = _release_input(tmp_path / "source")
    output = tmp_path / "archives"
    output.mkdir()
    target = output / "model.tar.gz"
    target.write_bytes(b"previous archive")
    before = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    assert create_tar.create_tar(_archive(source), output) == str(target)
    with tarfile.open(target) as stored:
        assert "model/encoder.onnx" in stored.getnames()
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == before
