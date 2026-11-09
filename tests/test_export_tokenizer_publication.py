from pathlib import Path

import numpy as np
import pytest

import export
from src import artifact_publication
from src.tokenizer_assets import validate_tokenizer_assets
from tests.artifact_fixtures import _ort_result, _write_external_identity, _write_model
from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE, save_tokenizer_assets


def snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_invalid_serialized_template_preserves_previous_family(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    save_tokenizer_assets(target, "json")
    from transformers import PreTrainedTokenizerFast

    save_pretrained = PreTrainedTokenizerFast.save_pretrained

    def corrupt_serialization(tokenizer, destination, *args, **kwargs):
        result = save_pretrained(tokenizer, destination, *args, **kwargs)
        (Path(destination) / "chat_template.jinja").write_text("WRONG PROMPT")
        return result

    monkeypatch.setattr(PreTrainedTokenizerFast, "save_pretrained", corrupt_serialization)
    before = snapshot(target)
    with pytest.raises(ValueError, match="Invalid ASR tokenizer assets"):
        export.copy_tokenizer(str(source), str(target))
    assert snapshot(target) == before
    validate_tokenizer_assets(target)


def test_serialization_failure_preserves_previous_family(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    tokenizer = save_tokenizer_assets(source)
    save_tokenizer_assets(target, "json")
    before = snapshot(target)
    monkeypatch.setattr(export, "load_prompt_tokenizer", lambda model_id: tokenizer)

    def fail(directory):
        (Path(directory) / "tokenizer.json").write_text("partial tokenizer")
        raise OSError("serialization failed")

    monkeypatch.setattr(tokenizer, "save_pretrained", fail)
    with pytest.raises(OSError, match="serialization failed"):
        export.copy_tokenizer(str(source), str(target))
    assert snapshot(target) == before
    validate_tokenizer_assets(target)


@pytest.mark.parametrize("asset", ["tokenizer_config.json", "chat_template.jinja"])
def test_export_refuses_to_replace_retained_tensor_storage(tmp_path, asset):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    save_tokenizer_assets(target)
    (target / "chat_template.jinja").write_text("{# prior template #}" + OFFICIAL_CHAT_TEMPLATE)
    _write_external_identity(target, "aux.onnx", asset)
    expected = _ort_result(target / "aux.onnx")
    before = snapshot(target)
    with pytest.raises(ValueError, match="retained graph"):
        export.copy_tokenizer(str(source), str(target))
    assert snapshot(target) == before
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), expected)


def test_export_can_reuse_identical_retained_storage(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    export.copy_tokenizer(str(source), str(target))
    _write_external_identity(target, "aux.onnx", "tokenizer_config.json")
    expected = _ort_result(target / "aux.onnx")
    export.copy_tokenizer(str(source), str(target))
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), expected)


def test_export_preserves_unrelated_extension_named_storage(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    target.mkdir()
    _write_model(target, "aux.onnx", external=True, data_name="weights.txt")
    expected = _ort_result(target / "aux.onnx")
    before = (target / "weights.txt").read_bytes()
    export.copy_tokenizer(str(source), str(target))
    assert (target / "weights.txt").read_bytes() == before
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), expected)


def test_export_publication_failure_restores_previous_family(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, "jinja")
    save_tokenizer_assets(target, "json")
    before = snapshot(target)
    replace = artifact_publication.os.replace
    failed = False

    def interrupt(original, destination):
        nonlocal failed
        result = replace(original, destination)
        if (
            not failed
            and Path(original).name == "tokenizer_config.json"
            and Path(destination) == target / "tokenizer_config.json"
        ):
            failed = True
            raise OSError("injected publication failure")
        return result

    monkeypatch.setattr(artifact_publication.os, "replace", interrupt)
    with pytest.raises(OSError, match="injected publication failure"):
        export.copy_tokenizer(str(source), str(target))
    assert failed
    assert snapshot(target) == before
    validate_tokenizer_assets(target)
