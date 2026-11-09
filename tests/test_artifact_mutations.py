import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from google.protobuf.message import DecodeError

import package
import share_weights
from src.onnx_artifacts import preflight_artifact_mutations
from tests.artifact_fixtures import _ort_result, _release_input, _write_external_identity, _write_model
from tests.converter_fixtures import ROOT, SCRIPTS, _converter_command, bytes_under, command, input_for


def test_preflight_preserves_original_dependencies_after_identical_copies_are_pruned(tmp_path):
    source = tmp_path / "source"
    target = source / "out"
    target.mkdir(parents=True)
    (target / "out").mkdir()
    weights = np.array([42], dtype=np.float32).tobytes()
    (target / "weights.data").write_bytes(weights)
    (target / "out/weights.data").write_bytes(weights)
    _write_external_identity(source, "aux.onnx", "out/weights.data")
    (target / "aux.onnx").write_bytes((source / "aux.onnx").read_bytes())
    _write_external_identity(target, "keeper-graph.onnx", "aux.onnx")
    _write_external_identity(target, "keeper-weights.onnx", "out/weights.data")
    before = bytes_under(source)

    copies = preflight_artifact_mutations(source, target, {"aux.onnx"}, read_files={"aux.onnx"})
    assert copies == set()
    assert (
        preflight_artifact_mutations(source, target, copies, generated={"unrelated.data"}, read_files={"aux.onnx"})
        == set()
    )
    with pytest.raises(ValueError, match="source graph destination conflict"):
        preflight_artifact_mutations(source, target, copies, generated={"weights.data"}, read_files={"aux.onnx"})
    assert bytes_under(source) == before


def test_package_preserves_source_metadata_referencing_output(tmp_path):
    from src.embeddings import load_embed_tokens

    source = _release_input(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    config = source / "config.json"
    (target / config.name).write_bytes(config.read_bytes())
    config.unlink()
    config.symlink_to(target / config.name)
    before = bytes_under(source)
    expected = load_embed_tokens(source)
    assert not package.package(source, target)
    assert bytes_under(source) == before
    np.testing.assert_array_equal(load_embed_tokens(source), expected)


def test_package_preserves_intermediate_source_directory_link(tmp_path):
    source = _release_input(tmp_path / "safe")
    target = tmp_path / "target"
    target.mkdir()
    bridge = target / "bridge"
    bridge.symlink_to(source, target_is_directory=True)
    original = tmp_path / "source"
    original.symlink_to(bridge, target_is_directory=True)
    before = bytes_under(source)
    assert not package.package(original, target)
    assert original.is_dir()
    assert bytes_under(original) == before


@pytest.mark.parametrize("script", SCRIPTS)
def test_converter_preserves_source_config_referencing_output(tmp_path, script):
    source = input_for(tmp_path, script)
    target = tmp_path / "target"
    target.mkdir()
    config = source / "config.json"
    (target / config.name).write_bytes(config.read_bytes())
    config.unlink()
    config.symlink_to(target / config.name)
    before_source, before_target = bytes_under(source), bytes_under(target)
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "source" in (result.stdout + result.stderr).lower()
    assert bytes_under(source) == before_source
    assert bytes_under(target) == before_target


def test_embedding_sidecar_preserves_canonical_source_alias(tmp_path):
    from convert_embed_fp16 import convert_embed_fp16
    from src.embeddings import load_embed_tokens

    source = _release_input(tmp_path / "source")
    canonical = source / "embed_tokens.bin"
    sidecar = source / "embed_tokens.fp16.bin"
    canonical.rename(sidecar)
    canonical.symlink_to(sidecar.name)
    expected = load_embed_tokens(source)
    before = bytes_under(source)
    with pytest.raises(ValueError, match="source"):
        convert_embed_fp16(source)
    assert bytes_under(source) == before
    np.testing.assert_array_equal(load_embed_tokens(source), expected)


@pytest.mark.parametrize("script", SCRIPTS)
def test_all_producers_preserve_other_precision_storage_before_tokenizer_removal(tmp_path, script):
    source = input_for(tmp_path, script)
    target = tmp_path / "target"
    target.mkdir()
    np.array([42], dtype=np.float32).tofile(target / "chat_template.jinja")
    _write_external_identity(target, "decoder_init.int16.onnx", "chat_template.jinja")
    before = bytes_under(target)
    expected = _ort_result(target / "decoder_init.int16.onnx")
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "retained graph" in result.stdout + result.stderr
    assert bytes_under(target) == before
    np.testing.assert_array_equal(_ort_result(target / "decoder_init.int16.onnx"), expected)


@pytest.mark.parametrize("script", SCRIPTS)
def test_all_producers_reject_conflicting_copied_storage(tmp_path, script):
    source = input_for(tmp_path, script)
    target = tmp_path / "target"
    target.mkdir()
    np.array([42], dtype=np.float32).tofile(target / "retained.data")
    np.array([99], dtype=np.float32).tofile(source / "retained.data")
    _write_external_identity(target, "aux.onnx", "retained.data")
    if script in {"convert_fp16.py", "quantize_static.py"}:
        _write_external_identity(source, "encoder.onnx", "retained.data")
    before = bytes_under(target)
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "retained graph" in result.stdout + result.stderr
    assert bytes_under(target) == before
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])


@pytest.mark.parametrize("script", SCRIPTS)
def test_inplace_producers_preserve_surviving_graph_metadata_storage(tmp_path, script):
    source = input_for(tmp_path, script)
    _write_external_identity(source, "decoder_init.int16.onnx", "config.json")
    before = bytes_under(source)
    result = subprocess.run(command(tmp_path, script, source, source), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "destination conflict" in result.stdout + result.stderr
    assert bytes_under(source) == before


def test_sharing_rejects_selected_graph_used_as_selected_tensor_storage(tmp_path):
    _write_model(tmp_path, "decoder_init.onnx")
    data = (tmp_path / "decoder_init.onnx").read_bytes()
    offset = data.index(np.arange(512, dtype=np.float32).tobytes()) + 4
    _write_external_identity(tmp_path, "decoder_step.onnx", "decoder_init.onnx", offset)
    expected = _ort_result(tmp_path / "decoder_step.onnx")
    before = bytes_under(tmp_path)
    for _ in range(2):
        with pytest.raises(ValueError, match="destination conflict"):
            share_weights.share_external_models(tmp_path)
        assert bytes_under(tmp_path) == before
        np.testing.assert_array_equal(_ort_result(tmp_path / "decoder_step.onnx"), expected)


def test_inventory_preserves_graph_role_when_graph_is_also_tensor_storage(tmp_path):
    _write_model(tmp_path, "decoder_init.onnx")
    data = (tmp_path / "decoder_init.onnx").read_bytes()
    offset = data.index(np.arange(512, dtype=np.float32).tobytes()) + 4
    _write_external_identity(tmp_path, "both.onnx", "decoder_init.onnx", offset)
    _write_external_identity(tmp_path, "decoder_step.onnx", "both.onnx")
    before = bytes_under(tmp_path)
    with pytest.raises(ValueError, match="destination conflict"):
        share_weights.share_external_models(tmp_path)
    assert bytes_under(tmp_path) == before


def test_raw_onnx_named_storage_survives_sharing_and_packaging(tmp_path):
    source = _release_input(tmp_path / "source")
    _write_model(source, "decoder_init.onnx", external=True, data_name="weights.onnx")
    expected = _ort_result(source / "decoder_init.onnx")
    before = (source / "weights.onnx").read_bytes()
    assert package.package(source, tmp_path / "release")
    np.testing.assert_array_equal(_ort_result(tmp_path / "release" / "decoder_init.onnx"), expected)
    assert (source / "weights.onnx").read_bytes() == before
    assert share_weights.share_external_models(source)
    assert share_weights.share_external_models(source)
    np.testing.assert_array_equal(_ort_result(source / "decoder_init.onnx"), expected)
    assert (source / "weights.onnx").read_bytes() == before


@pytest.mark.parametrize("consumer", ["package", "tokenizer"])
def test_committed_publication_reports_cleanup_failure_as_committed(tmp_path, monkeypatch, capsys, consumer):
    from src import artifact_publication
    from src.tokenizer_assets import copy_tokenizer_assets, validate_tokenizer_assets
    from tests.tokenizer_fixtures import save_tokenizer_assets

    original = artifact_publication.shutil.rmtree

    def fail_backup(path, *args, **kwargs):
        if Path(path).name.startswith(".release-backup-"):
            raise PermissionError("cleanup denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(artifact_publication.shutil, "rmtree", fail_backup)
    if consumer == "package":
        source = _release_input(tmp_path / "source")
        target = tmp_path / "target"
        target.mkdir()
        (target / "old").write_text("previous")
        assert package.package(source, target)
        assert not (target / "old").exists()
        assert (target / "decoder_init.onnx").is_file()
        assert (tmp_path / "target.tar.gz").is_file()
    else:
        source, target = tmp_path / "source", tmp_path / "target"
        save_tokenizer_assets(source, "json")
        save_tokenizer_assets(target, "jinja")
        copy_tokenizer_assets(source, target)
        validate_tokenizer_assets(target)
        assert not (target / "chat_template.jinja").exists()
        assert (target / "chat_template.json").is_file()
    output = capsys.readouterr().out
    assert "Publication completed" in output
    assert "cleanup denied" in output
    backup_root = tmp_path if consumer == "package" else target
    backups = list(backup_root.glob(".release-backup-*"))
    assert backups and str(backups[0]) in output


def test_postsharing_conversion_preserves_retained_opaque_storage(tmp_path):
    source = _release_input(tmp_path / "source")
    _write_model(source, "decoder_init.onnx", external=True, data_name="weights.onnx")
    expected = _ort_result(source / "decoder_init.onnx")
    storage = (source / "weights.onnx").read_bytes()
    assert share_weights.share_external_models(source)
    target = tmp_path / "converted"
    result = subprocess.run(
        _converter_command(tmp_path, "quantize.py", source, target), capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(_ort_result(target / "decoder_init.onnx"), expected)
    assert (source / "weights.onnx").read_bytes() == storage
    assert (target / "weights.onnx").read_bytes() == storage


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py"])
def test_partial_conversion_preserves_unselected_decoder(tmp_path, script):
    source = input_for(tmp_path, script)
    (source / "decoder_step.onnx").unlink()
    target = tmp_path / "target"
    target.mkdir()
    _write_model(target, "decoder_step.onnx", external=True)
    before = bytes_under(target)
    expected = _ort_result(target / "decoder_step.onnx")
    args = command(tmp_path, script, source, target)
    if script == "convert_fp16.py":
        args += ["--files", "decoder_init.onnx"]
    result = subprocess.run(args, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all((target / name).read_bytes() == value for name, value in before.items())
    np.testing.assert_array_equal(_ort_result(target / "decoder_step.onnx"), expected)
    assert not (target / "decoder_weights.data").exists()


@pytest.mark.parametrize("api", ["graph_file_closure", "transport_graph_files", "create_tar"])
def test_selected_opaque_graph_cannot_borrow_tensor_storage_role(tmp_path, api):
    from create_tar import create_tar
    from src import onnx_artifacts

    source = _release_input(tmp_path / "source")
    np.array([42], dtype=np.float32).tofile(source / "encoder.onnx")
    _write_external_identity(source, "decoder_init.onnx", "encoder.onnx")
    np.testing.assert_array_equal(_ort_result(source / "decoder_init.onnx"), [42])
    if api == "create_tar":
        destination = tmp_path / "bundle.tar.gz"
        destination.write_bytes(b"old archive")
        spec = {
            "name": "bundle",
            "source_dir": str(source),
            "files": ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"],
        }
        assert create_tar(spec, str(tmp_path)) is None
        assert destination.read_bytes() == b"old archive"
    else:
        with pytest.raises(DecodeError):
            getattr(onnx_artifacts, api)(source, ["encoder.onnx", "decoder_init.onnx"])


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("linked", [False, True])
def test_producers_reject_reader_template_directory_before_mutation(tmp_path, script, linked):
    source = input_for(tmp_path, script)
    target = tmp_path / "target"
    target.mkdir()
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "default.jinja").write_text("stale invalid template")
    destination = target / "additional_chat_templates"
    if linked:
        destination.symlink_to(templates, target_is_directory=True)
    else:
        destination.mkdir()
        (destination / "default.jinja").write_text("stale invalid template")
    before = bytes_under(target)
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "Unsupported tokenizer template directory" in result.stdout + result.stderr
    assert bytes_under(target) == before
    assert (templates / "default.jinja").read_text() == "stale invalid template"
    assert not (target / "decoder_init.onnx").exists()


@pytest.mark.parametrize("api", ["copy", "export"])
def test_tokenizer_writers_preserve_unsupported_destination(tmp_path, api):
    import export
    from src.tokenizer_assets import copy_tokenizer_assets

    source = input_for(tmp_path, "convert_fp16.py")
    target = tmp_path / "target"
    templates = target / "additional_chat_templates"
    templates.mkdir(parents=True)
    (templates / "default.jinja").write_text("stale invalid template")
    before = bytes_under(target)
    with pytest.raises(ValueError, match="Unsupported tokenizer template directory"):
        if api == "copy":
            copy_tokenizer_assets(source, target)
        else:
            export.copy_tokenizer(str(source), str(target))
    assert bytes_under(target) == before


def test_source_reader_template_directory_rejected_without_publishing(tmp_path):
    from src.tokenizer_assets import copy_tokenizer_assets

    source = input_for(tmp_path, "convert_fp16.py")
    templates = source / "additional_chat_templates"
    templates.mkdir()
    (templates / "default.jinja").write_text("stale invalid template")
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="Unsupported tokenizer template directory"):
        copy_tokenizer_assets(source, target)
    assert not target.exists()


@pytest.mark.parametrize(
    "operation,prefix",
    [
        ("package", ".package-"),
        ("tokenizer", ".tokenizer-"),
        ("archive", ".archive-"),
        ("archive-family", "qwen-archive-family-"),
        ("sharing", ".share-"),
        ("copy", ".artifact-"),
        ("metadata", ".artifact-"),
    ],
)
def test_published_artifacts_survive_staging_cleanup_error(tmp_path, monkeypatch, capsys, operation, prefix):
    from create_tar import create_tar
    from src import artifact_publication
    from src.tokenizer_assets import copy_tokenizer_assets, validate_tokenizer_assets

    source = _release_input(tmp_path / "source")
    target = tmp_path / "target"
    cleanup = artifact_publication.shutil.rmtree
    leftovers = []

    def fail_cleanup(path, *args, **kwargs):
        path = Path(path)
        if path.name.startswith(prefix):
            leftovers.append(path)
            raise PermissionError("staging cleanup failed")
        return cleanup(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(artifact_publication.shutil, "rmtree", fail_cleanup)
        if operation == "package":
            assert package.package(str(source), str(target))
            assert (target / "decoder_init.onnx").is_file()
            assert (tmp_path / "target.tar.gz").is_file()
        elif operation == "tokenizer":
            copy_tokenizer_assets(source, target)
            validate_tokenizer_assets(target)
        elif operation == "copy":
            share_weights.copy_artifact_entry(source, target, "tokenizer_config.json")
            assert (target / "tokenizer_config.json").read_bytes() == (source / "tokenizer_config.json").read_bytes()
        elif operation == "metadata":
            share_weights.write_artifact_json(target, "config.json", {"written": True})
            assert json.loads((target / "config.json").read_text()) == {"written": True}
        elif operation == "sharing":
            assert share_weights.share_external_models(source)
            assert share_weights.external_data_files(source / "decoder_step.onnx") == {source / "decoder_weights.data"}
        else:
            spec = {
                "name": "bundle",
                "source_dir": str(source),
                "files": ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"],
            }
            assert create_tar(spec, str(tmp_path)) == str(tmp_path / "bundle.tar.gz")
            import tarfile

            with tarfile.open(tmp_path / "bundle.tar.gz") as archive:
                config = archive.extractfile("bundle/config.json")
                assert config is not None
                assert json.load(config)["embed_tokens_dtype"] == "float32"
    output = capsys.readouterr().out
    assert "Publication completed; staging cleanup" in output
    assert leftovers and all(str(path) in output for path in leftovers)
    for path in leftovers:
        cleanup(path)


@pytest.mark.parametrize("script", SCRIPTS)
def test_graph_only_producers_reject_unsupported_destination_before_mutation(tmp_path, script):
    from src.tokenizer_assets import TOKENIZER_FILES

    source = input_for(tmp_path, script)
    for name in TOKENIZER_FILES:
        (source / name).unlink(missing_ok=True)
    target = tmp_path / "target"
    templates = target / "additional_chat_templates"
    templates.mkdir(parents=True)
    (templates / "default.jinja").write_text("stale invalid template")
    before = bytes_under(target)
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "Unsupported tokenizer template directory" in result.stdout + result.stderr
    assert bytes_under(target) == before


@pytest.mark.parametrize("script", ["export.py", "awq_smooth.py"])
def test_exporter_cli_rejects_unsupported_destination_before_checkpoint_loading(tmp_path, script):
    target = tmp_path / "target"
    templates = target / "additional_chat_templates"
    templates.mkdir(parents=True)
    (templates / "default.jinja").write_text("stale invalid template")
    before = bytes_under(target)
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--model", str(tmp_path / "missing-checkpoint"), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert "Unsupported tokenizer template directory" in result.stdout + result.stderr
    assert "Loading model" not in result.stdout
    assert bytes_under(target) == before


@pytest.mark.parametrize("linked", [False, True])
def test_package_rejects_unsupported_source_template_directory_before_projection(tmp_path, linked):
    source = _release_input(tmp_path / "source")
    target = tmp_path / "release"
    assert package.package(str(source), str(target))
    archive = tmp_path / "release.tar.gz"
    before, archive_before = bytes_under(target), archive.read_bytes()
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "default.jinja").write_text("stale invalid template")
    directory = source / "additional_chat_templates"
    if linked:
        directory.symlink_to(templates, target_is_directory=True)
    else:
        directory.mkdir()
        (directory / "default.jinja").write_text("stale invalid template")
    assert not package.package(str(source), str(target))
    assert bytes_under(target) == before
    assert archive.read_bytes() == archive_before
