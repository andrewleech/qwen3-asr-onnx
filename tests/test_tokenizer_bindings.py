import builtins
import errno
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import export
import package as pack
from src.prompt import load_prompt_tokenizer
from src.tokenizer_assets import copy_tokenizer_assets, validate_tokenizer_assets
from tests.artifact_fixtures import _write_external_identity
from tests.converter_fixtures import ROOT, _converter_command, _source, command, input_for
from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE, save_tokenizer_assets


def _configure(root, **values):
    path = root / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config.update(values)
    path.write_text(json.dumps(config))


def _snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _redirected_backend(source, target):
    save_tokenizer_assets(source)
    target.mkdir()
    backend = json.loads((source / "tokenizer.json").read_text())
    backend["model"]["vocab"]["EXTERNAL_BACKEND"] = 91
    external = target / "vocab.json"
    external.write_text(json.dumps(backend))
    _configure(source, tokenizer_file=str(external))
    assert "EXTERNAL_BACKEND" not in json.loads((source / "tokenizer.json").read_text())["model"]["vocab"]
    assert load_prompt_tokenizer(str(source), local_files_only=True).convert_tokens_to_ids("EXTERNAL_BACKEND") == 91


def test_family_rejects_external_backend_before_deleting_its_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    _redirected_backend(source, target)
    before_source, before_target = _snapshot(source), _snapshot(target)
    with pytest.raises(ValueError, match="tokenizer.*binding"):
        copy_tokenizer_assets(source, target)
    assert _snapshot(source) == before_source
    assert _snapshot(target) == before_target
    assert load_prompt_tokenizer(str(source), local_files_only=True).convert_tokens_to_ids("EXTERNAL_BACKEND") == 91


def test_package_rejects_external_backend_before_copying_metadata(tmp_path):
    source, external, target = tmp_path / "source", tmp_path / "external", tmp_path / "release"
    _source(source, "embedded")
    (source / "tokenizer.json").unlink()
    _redirected_backend(source, external)
    before_source, before_external = _snapshot(source), _snapshot(external)
    assert not pack.package(source, target)
    assert not target.exists()
    assert _snapshot(source) == before_source
    assert _snapshot(external) == before_external


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py", "quantize_nbits.py"])
def test_converter_rejects_backend_binding_before_output_mutations(tmp_path, script):
    source, target = tmp_path / "source", tmp_path / "target"
    _source(source, "embedded")
    _redirected_backend(source, target)
    before_source, before_target = _snapshot(source), _snapshot(target)
    command = (
        [sys.executable, str(ROOT / script), "--input", str(source), "--output", str(target)]
        if script == "quantize_nbits.py"
        else _converter_command(tmp_path, script, source, target)
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert "tokenizer file binding" in result.stdout + result.stderr
    assert _snapshot(source) == before_source
    assert _snapshot(target) == before_target


@pytest.mark.parametrize("asset", ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"])
@pytest.mark.parametrize("chained", [False, True])
def test_export_preserves_original_source_access_after_serialization(tmp_path, asset, chained):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    target.mkdir()
    retained = target / "vocab.json"
    retained.write_bytes((source / asset).read_bytes())
    (source / asset).unlink()
    if chained:
        bridge = tmp_path / "bridge"
        bridge.symlink_to(retained)
        (source / asset).symlink_to(bridge)
    else:
        (source / asset).symlink_to(retained)
    load_prompt_tokenizer(str(source), local_files_only=True)
    before_source, before_target = _snapshot(source), _snapshot(target)
    with pytest.raises(ValueError, match="source access path"):
        export.copy_tokenizer(str(source), str(target))
    assert _snapshot(source) == before_source
    assert _snapshot(target) == before_target
    load_prompt_tokenizer(str(source), local_files_only=True)


def test_export_can_preserve_safe_cache_links_and_normalize_raw_checkpoint(tmp_path):
    cache, source, target = tmp_path / "blobs", tmp_path / "snapshot", tmp_path / "output"
    save_tokenizer_assets(cache)
    source.mkdir()
    for path in cache.iterdir():
        (source / path.name).symlink_to(path)
    before = _snapshot(source)
    export.copy_tokenizer(str(source), str(target))
    assert _snapshot(source) == before
    assert validate_tokenizer_assets(target).get_vocab() == load_prompt_tokenizer(str(source)).get_vocab()
    assert all(path.is_symlink() for path in source.iterdir())


def test_export_rejects_raw_backend_binding_without_changing_identity(tmp_path):
    source, external, target = tmp_path / "source", tmp_path / "external", tmp_path / "output"
    _redirected_backend(source, external)
    before = _snapshot(source), _snapshot(external)
    with pytest.raises(ValueError, match="tokenizer.*binding"):
        export.copy_tokenizer(str(source), str(target))
    assert not target.exists()
    assert (_snapshot(source), _snapshot(external)) == before
    assert load_prompt_tokenizer(str(source)).convert_tokens_to_ids("EXTERNAL_BACKEND") == 91


def test_export_preserves_model_config_read_by_original_large_tokenizer(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    path = source / "tokenizer.json"
    document = json.loads(path.read_text())
    document["model"]["vocab"].update({f"word_{index}": index + 200 for index in range(100001)})
    path.write_text(json.dumps(document))
    target.mkdir()
    retained = target / "vocab.json"
    retained.write_text('{"model_type": "qwen3_asr"}')
    model_config = source / "config.json"
    model_config.symlink_to(retained)
    opened = []
    original_open = builtins.open

    def observe(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).absolute() == model_config.absolute():
            opened.append(file)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", observe)
    before_source, before_target = _snapshot(source), _snapshot(target)
    with pytest.raises(ValueError, match="source access path"):
        export.copy_tokenizer(str(source), str(target))
    assert opened
    assert _snapshot(source) == before_source
    assert _snapshot(target) == before_target


def _split_qwen_checkpoint(root, *, versioned, tokenizer_class="Qwen2TokenizerFast"):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    save_tokenizer_assets(root, "embedded")
    canonical = json.loads((root / "tokenizer.json").read_text())["model"]["vocab"]
    special = {token: index for token, index in canonical.items() if index > 151000}
    vocab = {token: index for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab.update(special)
    vocab["SLOW_BACKEND"] = 300
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.model.save(str(root))
    _configure(root, tokenizer_class=tokenizer_class, unk_token="<|endoftext|>")
    if versioned:
        _configure(root, fast_tokenizer_files=["tokenizer.4.0.0.json"])
    else:
        (root / "tokenizer.json").unlink()
    return load_prompt_tokenizer(str(root), local_files_only=True)


def test_family_rejects_missing_version_even_when_real_qwen_slow_fallback_loads(tmp_path):
    source = tmp_path / "source"
    tokenizer = _split_qwen_checkpoint(source, versioned=True)
    assert tokenizer.convert_tokens_to_ids("SLOW_BACKEND") == 300
    assert "SLOW_BACKEND" not in json.loads((source / "tokenizer.json").read_text())["model"]["vocab"]
    with pytest.raises(ValueError, match="tokenizer.*binding"):
        validate_tokenizer_assets(source)


@pytest.mark.parametrize("tokenizer_class", ["Qwen2Tokenizer", "Qwen2TokenizerFast"])
def test_export_normalizes_ordinary_split_vocab_checkpoint(tmp_path, tokenizer_class):
    source, target = tmp_path / "source", tmp_path / "target"
    expected = _split_qwen_checkpoint(source, versioned=False, tokenizer_class=tokenizer_class)
    before = _snapshot(source)
    export.copy_tokenizer(str(source), str(target))
    assert _snapshot(source) == before
    actual = validate_tokenizer_assets(target)
    assert actual.get_vocab() == expected.get_vocab()
    assert actual.encode("Kepler") == expected.encode("Kepler")


@pytest.mark.parametrize("slot", ["tokenizer_file", "vocab_file", "merges_file"])
def test_family_rejects_constructor_file_overrides_even_when_null(tmp_path, slot):
    source = tmp_path / "source"
    save_tokenizer_assets(source)
    _configure(source, tokenizer_class="Qwen2TokenizerFast", **{slot: None})
    with pytest.raises(ValueError, match="tokenizer.*binding"):
        validate_tokenizer_assets(source)


@pytest.mark.parametrize("value", [["tokenizer.4.0.0.json"], ["/external/tokenizer.4.0.0.json"]])
def test_family_rejects_selected_noncanonical_backend_even_when_missing(tmp_path, value):
    source = tmp_path / "source"
    save_tokenizer_assets(source)
    _configure(source, fast_tokenizer_files=value)
    with pytest.raises(ValueError, match="tokenizer.*binding"):
        validate_tokenizer_assets(source)


@pytest.mark.parametrize("value", [[], ["tokenizer.json"], ["tokenizer.999.0.0.json"]])
def test_family_accepts_version_lists_selecting_canonical_backend(tmp_path, value):
    source = tmp_path / "source"
    expected = save_tokenizer_assets(source)
    _configure(source, fast_tokenizer_files=value, from_slow=False, tokenizer_object=None, gguf_file=None)
    assert validate_tokenizer_assets(source).get_vocab() == expected.get_vocab()


def test_canonical_family_same_directory_remains_a_noop(tmp_path):
    save_tokenizer_assets(tmp_path)
    before = _snapshot(tmp_path)
    copy_tokenizer_assets(tmp_path, tmp_path)
    assert _snapshot(tmp_path) == before


def _infer_class_from_model_config(source):
    config_path = source / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config.pop("tokenizer_class")
    config_path.write_text(json.dumps(config))
    model_path = source / "config.json"
    model = json.loads(model_path.read_text()) if model_path.exists() else {}
    model["model_type"] = "qwen2"
    model_path.write_text(json.dumps(model))
    return validate_tokenizer_assets(source)


@pytest.mark.parametrize("template_format", ["jinja", "json", "embedded", "named"])
@pytest.mark.parametrize("special_map", [False, True])
def test_model_inferred_family_is_portable_without_changing_model_metadata(tmp_path, template_format, special_map):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, template_format="embedded" if template_format == "named" else template_format)
    if template_format == "named":
        _configure(source, chat_template={"default": OFFICIAL_CHAT_TEMPLATE, "alternate": OFFICIAL_CHAT_TEMPLATE})
    if not special_map:
        (source / "special_tokens_map.json").unlink()
    expected = _infer_class_from_model_config(source)
    before_source = _snapshot(source)
    target.mkdir()
    model_path = target / "config.json"
    model_path.write_text('{"model_type": "bert", "runtime_owner": "retained"}')
    before_model = model_path.read_bytes()
    copy_tokenizer_assets(source, target)
    assert _snapshot(source) == before_source
    assert model_path.read_bytes() == before_model
    actual = validate_tokenizer_assets(target)
    assert type(actual) is type(expected)
    assert actual.get_vocab() == expected.get_vocab()
    assert actual.encode("Kepler") == expected.encode("Kepler")
    assert actual.chat_template == expected.chat_template
    assert json.loads((target / "tokenizer_config.json").read_text())["origin_marker"] == "tiny-local-fixture"
    model_path.unlink()
    assert validate_tokenizer_assets(target).get_vocab() == expected.get_vocab()


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py", "quantize_nbits.py"])
@pytest.mark.parametrize("storage", ["incoming", "retained"])
def test_prepared_family_conflicts_fail_before_converter_writes(tmp_path, script, storage):
    source = input_for(tmp_path, script)
    _infer_class_from_model_config(source)
    target = tmp_path / "target"
    target.mkdir()
    if storage == "incoming":
        _write_external_identity(source, "encoder.onnx", "tokenizer_config.json")
    else:
        (target / "tokenizer_config.json").write_bytes((source / "tokenizer_config.json").read_bytes())
        _write_external_identity(target, "aux.onnx", "tokenizer_config.json")
    before_source, before_target = _snapshot(source), _snapshot(target)
    arguments = command(tmp_path, script, source, target)
    if script == "convert_fp16.py":
        arguments += ["--files", "decoder_init.onnx", "decoder_step.onnx"]
    result = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert _snapshot(source) == before_source
    assert _snapshot(target) == before_target
    assert "destination conflict" in result.stdout + result.stderr


def test_normalized_family_can_reuse_identical_retained_storage(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    expected = _infer_class_from_model_config(source)
    copy_tokenizer_assets(source, target)
    _write_external_identity(target, "aux.onnx", "tokenizer_config.json")
    before = _snapshot(target)
    copy_tokenizer_assets(source, target)
    assert _snapshot(target) == before
    assert validate_tokenizer_assets(target).get_vocab() == expected.get_vocab()


def test_model_inferred_family_same_directory_remains_a_noop(tmp_path):
    save_tokenizer_assets(tmp_path)
    _infer_class_from_model_config(tmp_path)
    before = _snapshot(tmp_path)
    copy_tokenizer_assets(tmp_path, tmp_path)
    assert _snapshot(tmp_path) == before


def test_family_copy_does_not_validate_unselected_source_graphs(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    _write_external_identity(source, "unrelated.onnx", "missing.data")
    copy_tokenizer_assets(source, target)
    validate_tokenizer_assets(target)
    assert not (target / "unrelated.onnx").exists()


def test_staging_directory_name_cannot_supply_published_tokenizer_class(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    expected = _infer_class_from_model_config(source)
    staging_root = tmp_path / "qwen2-staging"
    staging_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging_root))
    copy_tokenizer_assets(source, target)
    config = json.loads((target / "tokenizer_config.json").read_text())
    assert config["tokenizer_class"]
    assert validate_tokenizer_assets(target).get_vocab() == expected.get_vocab()


def test_family_publication_uses_destination_filesystem(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    save_tokenizer_assets(target, "json")
    replace = os.replace

    def same_volume_replace(original, destination):
        if Path(destination).is_relative_to(target) and not Path(original).is_relative_to(target):
            raise OSError(errno.EXDEV, "cross-device link")
        return replace(original, destination)

    monkeypatch.setattr(os, "replace", same_volume_replace)
    copy_tokenizer_assets(source, target)
    assert validate_tokenizer_assets(target).get_vocab() == validate_tokenizer_assets(source).get_vocab()


def _deferred_backend_source(source, source_applies):
    from tokenizers import pre_tokenizers

    save_tokenizer_assets(source)
    path = source / "tokenizer.json"
    document = json.loads(path.read_text())
    document["model"]["vocab"].update({f"word_{index}": index + 200 for index in range(100001)})
    document["pre_tokenizer"] = json.loads(pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit()]).__getstate__())
    path.write_text(json.dumps(document))
    _configure(source, fix_mistral_regex=True)
    model = {"model_type": "qwen2"}
    if not source_applies:
        model["transformers_version"] = "4.57.2"
    (source / "config.json").write_text(json.dumps(model))
    return validate_tokenizer_assets(source)


@pytest.mark.parametrize("source_applies", [False, True])
def test_destination_model_config_cannot_activate_deferred_backend_change(tmp_path, source_applies):
    source, target = tmp_path / "source", tmp_path / "target"
    expected = _deferred_backend_source(source, source_applies)
    before_source = _snapshot(source)
    target.mkdir()
    model_config = target / "config.json"
    model_config.write_text('{"model_type": "qwen2"}')
    before_model = model_config.read_bytes()
    copy_tokenizer_assets(source, target)
    actual = validate_tokenizer_assets(target)
    assert actual.get_vocab() == expected.get_vocab()
    assert actual.encode("Kepler") == expected.encode("Kepler")
    assert json.loads(actual.backend_tokenizer.to_str()) == json.loads(expected.backend_tokenizer.to_str())
    assert _snapshot(source) == before_source
    assert model_config.read_bytes() == before_model
    model_config.unlink()
    assert validate_tokenizer_assets(target).get_vocab() == expected.get_vocab()


def test_export_can_reuse_normalized_retained_tokenizer_storage(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    expected = _deferred_backend_source(source, False)
    export.copy_tokenizer(str(source), str(target))
    _write_external_identity(target, "aux.onnx", "tokenizer_config.json")
    before_source, before_target = _snapshot(source), _snapshot(target)
    export.copy_tokenizer(str(source), str(target))
    assert _snapshot(source) == before_source
    assert _snapshot(target) == before_target
    actual = validate_tokenizer_assets(target)
    assert json.loads(actual.backend_tokenizer.to_str()) == json.loads(expected.backend_tokenizer.to_str())
