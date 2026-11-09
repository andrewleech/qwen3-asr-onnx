import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.prompt import build_prompt_ids, load_prompt_tokenizer
from tests.converter_fixtures import _source
from tests.tokenizer_fixtures import save_tokenizer_assets

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("representation", ["json", "embedded"])
@pytest.mark.parametrize(
    "script", ["convert_fp16.py", "quantize_static.py", "package.py", "quantize.py", "quantize_nbits.py"]
)
def test_conflicting_prompt_representation_fails_before_output(tmp_path, representation, script):
    source, target = tmp_path / "source", tmp_path / "target"
    _source(source, "jinja")
    if representation == "json":
        (source / "chat_template.json").write_text(json.dumps({"chat_template": "WRONG PROMPT"}))
    else:
        path = source / "tokenizer_config.json"
        config = json.loads(path.read_text())
        config["chat_template"] = "WRONG PROMPT"
        path.write_text(json.dumps(config))
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--input", str(source), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert "Invalid ASR tokenizer assets" in result.stdout + result.stderr
    assert not target.exists()


def test_export_removes_stale_processor_template_and_both_readers_agree(tmp_path):
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    import export

    checkpoint, target = tmp_path / "checkpoint", tmp_path / "target"
    save_tokenizer_assets(checkpoint)
    config_path = checkpoint / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "Qwen2TokenizerFast"
    config["extra_special_tokens"] = {
        "audio_token": "<|audio_pad|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
    }
    config_path.write_text(json.dumps(config))
    target.mkdir()
    (target / "chat_template.json").write_text(json.dumps({"chat_template": "WRONG PROMPT"}))
    export.copy_tokenizer(str(checkpoint), str(target))
    export.write_preprocessor_config(str(target))
    assert not (target / "chat_template.json").exists()
    tokenizer = load_prompt_tokenizer(str(target), local_files_only=True)
    processor = Qwen3ASRProcessor.from_pretrained(str(target), local_files_only=True)
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    assert processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ) == tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def test_normalized_family_preserves_official_processor_template(tmp_path):
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    import export
    from src.tokenizer_assets import copy_tokenizer_assets

    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    config_path = source / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config.pop("tokenizer_class")
    config["extra_special_tokens"] = {
        "audio_token": "<|audio_pad|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
    }
    config_path.write_text(json.dumps(config))
    (source / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    export.write_preprocessor_config(str(source))
    processor = Qwen3ASRProcessor.from_pretrained(str(source), local_files_only=True)
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    expected = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    copy_tokenizer_assets(source, target)
    export.write_preprocessor_config(str(target))
    copied = Qwen3ASRProcessor.from_pretrained(str(target), local_files_only=True)
    assert copied.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) == expected
    assert (
        load_prompt_tokenizer(str(target), local_files_only=True).apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        == expected
    )


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py", "quantize_nbits.py"])
def test_conflicting_reused_target_is_repaired_by_family_publication(tmp_path, script):
    from src.tokenizer_assets import validate_tokenizer_assets
    from tests.artifact_fixtures import _write_model
    from tests.converter_fixtures import _converter_command

    source, target = tmp_path / "source", tmp_path / "target"
    _source(source, "jinja")
    expected = load_prompt_tokenizer(str(source), local_files_only=True)
    save_tokenizer_assets(target, "jinja")
    (target / "chat_template.json").write_text(json.dumps({"chat_template": "WRONG PROMPT"}))
    if script == "quantize_nbits.py":
        for name in ("decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, weight=np.ones((64, 64), dtype=np.float32))
        command = [
            sys.executable,
            str(ROOT / script),
            "--input",
            str(source),
            "--output",
            str(target),
            "--bits",
            "4",
            "--block-size",
            "32",
        ]
    else:
        command = _converter_command(tmp_path, script, source, target)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (target / "chat_template.json").exists()
    actual = validate_tokenizer_assets(target)
    assert build_prompt_ids(3, actual) == build_prompt_ids(3, expected)


def test_export_normalizes_selected_asr_template(tmp_path):
    import export
    from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE

    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, "embedded")
    config_path = source / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["chat_template"] = {"default": OFFICIAL_CHAT_TEMPLATE, "other": "unused alternate"}
    config_path.write_text(json.dumps(config))
    export.copy_tokenizer(str(source), str(target))
    assert (target / "chat_template.jinja").read_text() == OFFICIAL_CHAT_TEMPLATE
    assert not (target / "chat_templates").exists()
    assert load_prompt_tokenizer(str(target), local_files_only=True).chat_template == OFFICIAL_CHAT_TEMPLATE


@pytest.mark.parametrize("route", ["copy", "inplace", "package", "archive"])
def test_embedded_family_completes_official_processor(tmp_path, route):
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    import export
    import package
    from src.tokenizer_assets import copy_tokenizer_assets
    from tests.converter_fixtures import bytes_under

    source = _source(tmp_path / "source", "embedded")
    config_path = source / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "Qwen2TokenizerFast"
    config["extra_special_tokens"] = {
        "audio_token": "<|audio_pad|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
    }
    config_path.write_text(json.dumps(config))
    export.write_preprocessor_config(str(source))
    expected = load_prompt_tokenizer(str(source), local_files_only=True)
    before = bytes_under(source)
    target = source if route == "inplace" else tmp_path / "target"
    if route == "package":
        package._package(source, target)
    elif route == "archive":
        import tarfile

        from create_tar import create_tar
        from src.release_validation import graph_names

        (tmp_path / "bundle.tar.gz").write_bytes(b"previous archive")
        archive = create_tar({"name": "bundle", "source_dir": str(source), "files": graph_names()}, tmp_path)
        assert archive is not None
        with tarfile.open(archive) as stream:
            stream.extractall(target, filter="data")
        target = target / "bundle"
    else:
        copy_tokenizer_assets(source, target)
        if route != "inplace":
            export.write_preprocessor_config(str(target))
    processor = Qwen3ASRProcessor.from_pretrained(str(target), local_files_only=True)
    messages = [{"role": "system", "content": "Kepler"}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    assert processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ) == expected.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    after = bytes_under(source)
    assert all(after[name] == value for name, value in before.items())
    if route != "inplace":
        assert after == before


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("view", ["legacy", "named", "processor-config"])
def test_transport_preserves_original_processor_view(tmp_path, canonical, view):
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

    import export
    from src.tokenizer_assets import copy_tokenizer_assets
    from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE

    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, "embedded")
    path = source / "tokenizer_config.json"
    config = json.loads(path.read_text())
    if canonical:
        config.pop("tokenizer_class")
        (source / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    else:
        config["tokenizer_class"] = "Qwen2TokenizerFast"
    config["extra_special_tokens"] = {
        "audio_token": "<|audio_pad|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
    }
    config["chat_template"] = {"default": OFFICIAL_CHAT_TEMPLATE, "other": OFFICIAL_CHAT_TEMPLATE + "{# tokenizer #}"}
    path.write_text(json.dumps(config))
    original = OFFICIAL_CHAT_TEMPLATE + "{% if messages[0].content == 'distinct' %}ORIGINAL{% endif %}"
    value = {"default": original, "other": OFFICIAL_CHAT_TEMPLATE} if view == "named" else original
    template_file = "processor_config.json" if view == "processor-config" else "chat_template.json"
    (source / template_file).write_text(json.dumps({"chat_template": value}))
    export.write_preprocessor_config(str(source))
    source_processor = Qwen3ASRProcessor.from_pretrained(str(source), local_files_only=True)
    tokenizer = load_prompt_tokenizer(str(source), local_files_only=True)
    copy_tokenizer_assets(source, target)
    export.write_preprocessor_config(str(target))
    copied = Qwen3ASRProcessor.from_pretrained(str(target), local_files_only=True)
    messages = [
        {"role": "system", "content": "distinct"},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    expected = source_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    assert expected.endswith("ORIGINAL")
    assert copied.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) == expected
    assert copied.chat_template == source_processor.chat_template
    actual = load_prompt_tokenizer(str(target), local_files_only=True)
    assert actual.chat_template == tokenizer.chat_template
    assert actual.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ) == tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if view == "named":
        assert copied.apply_chat_template(
            messages, chat_template="other", tokenize=False
        ) == source_processor.apply_chat_template(messages, chat_template="other", tokenize=False)


@pytest.mark.parametrize("value", [[{"name": "default", "template": "unused"}], {"other": "unused"}])
def test_unusable_present_processor_template_fails_before_publication(tmp_path, value):
    from src.tokenizer_assets import copy_tokenizer_assets
    from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE

    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    if isinstance(value, list):
        value[0]["template"] = OFFICIAL_CHAT_TEMPLATE
    else:
        value["other"] = OFFICIAL_CHAT_TEMPLATE
    (source / "chat_template.json").write_text(json.dumps({"chat_template": value}))
    with pytest.raises(ValueError, match="Processor chat template"):
        copy_tokenizer_assets(source, target)
    assert not target.exists()


@pytest.mark.parametrize("canonical", [False, True])
def test_package_checks_actual_prepared_tokenizer_dependency_bytes(tmp_path, canonical):
    import package
    from tests.artifact_fixtures import _write_external_identity
    from tests.converter_fixtures import bytes_under

    source = _source(tmp_path / "source", "embedded")
    if canonical:
        path = source / "tokenizer_config.json"
        config = json.loads(path.read_text())
        config.pop("tokenizer_class")
        path.write_text(json.dumps(config))
        path = source / "config.json"
        config = json.loads(path.read_text())
        config["model_type"] = "qwen2"
        path.write_text(json.dumps(config))
    _write_external_identity(source, "encoder.onnx", "tokenizer_config.json")
    before = bytes_under(source)
    target = tmp_path / "target"
    if canonical:
        with pytest.raises(ValueError, match="transport destination conflict"):
            package._package(source, target)
        assert not target.exists()
    else:
        package._package(source, target)
        assert (target / "tokenizer_config.json").read_bytes() == (source / "tokenizer_config.json").read_bytes()
    assert bytes_under(source) == before


def test_processor_config_original_access_survives_family_cleanup(tmp_path):
    from src.tokenizer_assets import copy_tokenizer_assets
    from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE

    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, "embedded")
    target.mkdir()
    original = json.dumps({"chat_template": OFFICIAL_CHAT_TEMPLATE}).encode()
    (target / "vocab.json").write_bytes(original)
    (source / "processor_config.json").symlink_to(target / "vocab.json")
    with pytest.raises(ValueError, match="source"):
        copy_tokenizer_assets(source, target)
    assert (source / "processor_config.json").read_bytes() == original
    assert (target / "vocab.json").read_bytes() == original


@pytest.mark.parametrize("canonical", [False, True])
def test_archive_checks_actual_prepared_tokenizer_dependency_bytes(tmp_path, canonical):
    from create_tar import create_tar
    from src.release_validation import graph_names
    from tests.artifact_fixtures import _write_external_identity
    from tests.converter_fixtures import bytes_under

    source = _source(tmp_path / "source", "embedded")
    if canonical:
        path = source / "tokenizer_config.json"
        config = json.loads(path.read_text())
        config.pop("tokenizer_class")
        path.write_text(json.dumps(config))
        path = source / "config.json"
        config = json.loads(path.read_text())
        config["model_type"] = "qwen2"
        path.write_text(json.dumps(config))
    _write_external_identity(source, "encoder.onnx", "tokenizer_config.json")
    before = bytes_under(source)
    previous = tmp_path / "bundle.tar.gz"
    previous.write_bytes(b"previous archive")
    result = create_tar({"name": "bundle", "source_dir": str(source), "files": graph_names()}, tmp_path)
    if canonical:
        assert result is None
        assert previous.read_bytes() == b"previous archive"
    else:
        assert result is not None
    assert bytes_under(source) == before


@pytest.mark.parametrize("entry", ["public", "cli"])
def test_package_completion_reports_success(tmp_path, monkeypatch, entry):
    import package

    source = _source(tmp_path / "source", "embedded")
    target = tmp_path / "target"
    if entry == "public":
        assert package.package(source, target) is True
    else:
        monkeypatch.setattr(sys, "argv", ["package.py", "--input", str(source), "--output", str(target)])
        assert package.main() == 0
    assert (target / "chat_template.json").is_file()
