import builtins
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

from src.prompt import build_prompt_ids, get_audio_pad_range
from src.tokenizer_assets import (
    TOKENIZER_FILES,
    copy_tokenizer_assets,
    tokenizer_asset_names,
    validate_tokenizer_assets,
)
from tests.artifact_fixtures import _write_model
from tests.converter_fixtures import ROOT, _converter_command, _source
from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE, save_tokenizer_assets


def _assert_source_assets(source, target):
    expected_names = tokenizer_asset_names(source)
    if not expected_names & {"chat_template.json", "chat_template.jinja"}:
        expected_names.add("chat_template.json")
        assert (
            json.loads((target / "chat_template.json").read_text())["chat_template"]
            == validate_tokenizer_assets(source).get_chat_template()
        )
    assert tokenizer_asset_names(target) == expected_names
    for name in tokenizer_asset_names(source):
        assert (target / name).read_bytes() == (source / name).read_bytes()
    tokenizer = validate_tokenizer_assets(target)
    ids = build_prompt_ids(3, tokenizer, context="Kepler", language="English")
    start, end = get_audio_pad_range(ids, expected_count=3)
    assert end - start == 3
    assert json.loads((target / "tokenizer_config.json").read_text())["origin_marker"] == "tiny-local-fixture"


@pytest.mark.parametrize("template_format", ["jinja", "json", "embedded"])
def test_asset_copy_replaces_stale_templates_and_preserves_origin(tmp_path, template_format):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, template_format=template_format)
    save_tokenizer_assets(target, template_format="jinja")
    (target / "chat_template.jinja").write_text("obsolete template")
    (target / "chat_template.json").write_text("{}")
    copy_tokenizer_assets(source, target)
    _assert_source_assets(source, target)


@pytest.mark.parametrize("template_format", ["embedded", "json"])
@pytest.mark.parametrize("failed_name", ["tokenizer.json", "tokenizer_config.json"])
def test_asset_publication_failure_restores_prior_tokenizer(tmp_path, monkeypatch, template_format, failed_name):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, template_format=template_format)
    save_tokenizer_assets(target, template_format="jinja")
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    previous_ids = build_prompt_ids(3, validate_tokenizer_assets(target))
    replace = os.replace
    failed = False

    def interrupt(src, dst):
        nonlocal failed
        if not failed and Path(src).name == failed_name and Path(dst) == target / failed_name:
            failed = True
            raise OSError("injected tokenizer publication failure")
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(OSError, match="injected tokenizer"):
        copy_tokenizer_assets(source, target)
    assert failed
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    assert build_prompt_ids(3, validate_tokenizer_assets(target)) == previous_ids
    copy_tokenizer_assets(source, target)
    _assert_source_assets(source, target)


@pytest.mark.parametrize("problem", ["missing-template", "bad-template", "missing-tokenizer", "invalid-json"])
def test_invalid_source_is_rejected_before_target_changes(tmp_path, problem):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source, include_template=problem != "missing-template")
    if problem == "bad-template":
        (source / "chat_template.jinja").write_text(OFFICIAL_CHAT_TEMPLATE.replace("system\\n", "broken\\n"))
    if problem == "missing-tokenizer":
        (source / "tokenizer.json").unlink()
    if problem == "invalid-json":
        (source / "tokenizer.json").write_text("not-json")
    target.mkdir()
    (target / "chat_template.jinja").write_text("previous")
    with pytest.raises(ValueError):
        copy_tokenizer_assets(source, target)
    assert (target / "chat_template.jinja").read_text() == "previous"


@pytest.mark.parametrize("template_format", ["jinja", "json", "embedded"])
def test_validation_is_offline(tmp_path, template_format, monkeypatch):
    import transformers.utils.hub

    source = tmp_path / "source"
    save_tokenizer_assets(source, template_format=template_format)

    def no_download(*args, **kwargs):
        raise AssertionError("Tokenizer validation must not download replacement assets")

    monkeypatch.setattr(transformers.utils.hub, "hf_hub_download", no_download)
    validate_tokenizer_assets(source)


@pytest.mark.parametrize("template_format", ["jinja", "json", "embedded"])
@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py"])
def test_converter_cli_transports_complete_tokenizer(tmp_path, script, template_format):
    source = _source(tmp_path / "source", template_format)
    target = tmp_path / "target"
    save_tokenizer_assets(target, template_format="jinja")
    (target / "chat_template.jinja").write_text("obsolete template")
    command = _converter_command(tmp_path, script, source, target)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_source_assets(source, target)
    for name in ("encoder.onnx", "embed_tokens.bin"):
        assert (target / name).read_bytes() == (source / name).read_bytes()


@pytest.mark.parametrize("template_format", ["jinja", "json", "embedded"])
def test_archive_retains_complete_renderable_tokenizer(tmp_path, template_format):
    from create_tar import create_tar

    source = _source(tmp_path / "source", template_format)
    names = ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"]
    archive_path = create_tar({"name": "bundle", "source_dir": str(source), "files": names}, str(tmp_path))
    assert archive_path is not None
    extracted = tmp_path / "extracted"
    with tarfile.open(archive_path) as archive:
        archive.extractall(extracted, filter="data")
    target = extracted / "bundle"
    _assert_source_assets(source, target)
    for name in names + ["embed_tokens.bin", "config.json", "preprocessor_config.json"]:
        assert (target / name).read_bytes() == (source / name).read_bytes()


def test_archive_rejects_missing_template_before_replacing_archive(tmp_path):
    from create_tar import create_tar

    source = _source(tmp_path / "source", "jinja")
    (source / "chat_template.jinja").unlink()
    destination = tmp_path / "bundle.tar.gz"
    destination.write_bytes(b"previous archive")
    assert create_tar({"name": "bundle", "source_dir": str(source), "files": []}, str(tmp_path)) is None
    assert destination.read_bytes() == b"previous archive"


def test_comparison_wrapper_uses_template_bearing_local_tokenizer(tmp_path, monkeypatch):
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

    import validate
    from compare import run_wrapper_pytorch

    tokenizer = save_tokenizer_assets(tmp_path / "tokenizer")
    config = Qwen3ASRAudioEncoderConfig(
        d_model=8,
        encoder_attention_heads=2,
        encoder_layers=1,
        encoder_ffn_dim=16,
        downsample_hidden_size=2,
        output_dim=4,
        n_window=50,
        n_window_infer=800,
    )
    config._attn_implementation = "eager"
    model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=Qwen3ASRAudioEncoder(config).eval()))
    prompts = []

    def decode(model, features, prompt_ids, **kwargs):
        prompts.append(prompt_ids)
        return [151645]

    monkeypatch.setattr(validate, "greedy_decode_pytorch", decode)
    result = run_wrapper_pytorch(model, np.zeros(1600, dtype=np.float32), tokenizer)
    assert prompts == [build_prompt_ids(result["audio_token_count"], tokenizer)]


@pytest.mark.parametrize("missing_template", [False, True])
def test_archive_cli_preserves_tokenizer_or_fails_without_overwriting(tmp_path, missing_template):
    from share_weights import share_external_models

    source = _source(tmp_path / "release" / "qwen3-asr-0.6b", "jinja")
    assert share_external_models(source)
    archive_path = tmp_path / "qwen3-asr-0.6b.tar.gz"
    if missing_template:
        (source / "chat_template.jinja").unlink()
        archive_path.write_bytes(b"previous archive")
    command = [
        sys.executable,
        str(ROOT / "create_tar.py"),
        "--dry-run",
        "--only",
        "qwen3-asr-0.6b",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    if missing_template:
        assert result.returncode != 0
        assert archive_path.read_bytes() == b"previous archive"
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        with tarfile.open(archive_path) as archive:
            archive.extractall(tmp_path / "extracted", filter="data")
        _assert_source_assets(source, tmp_path / "extracted" / "qwen3-asr-0.6b")


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py"])
def test_graph_only_conversion_does_not_invent_tokenizer_assets(tmp_path, script):
    source = _source(tmp_path / "source", "jinja")
    for name in TOKENIZER_FILES:
        (source / name).unlink(missing_ok=True)
    target = tmp_path / "target"
    command = _converter_command(tmp_path, script, source, target)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not tokenizer_asset_names(target)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_asset_copy_does_not_mutate_linked_retained_metadata(tmp_path, link_kind):
    source = tmp_path / "source"
    save_tokenizer_assets(source)
    target = tmp_path / "target"
    target.mkdir()
    retained = tmp_path / "retained.json"
    retained.write_bytes(b"retained metadata")
    path = target / "tokenizer_config.json"
    if link_kind == "symlink":
        path.symlink_to(retained)
    else:
        os.link(retained, path)
    copy_tokenizer_assets(source, target)
    assert retained.read_bytes() == b"retained metadata"
    assert not path.is_symlink()
    _assert_source_assets(source, target)


def test_archive_materializes_linked_template_bytes(tmp_path):
    from create_tar import create_tar

    source = _source(tmp_path / "source", "jinja")
    retained_template = tmp_path / "template.jinja"
    (source / "chat_template.jinja").rename(retained_template)
    (source / "chat_template.jinja").symlink_to(retained_template)
    names = ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"]
    path = create_tar({"name": "bundle", "source_dir": str(source), "files": names}, str(tmp_path))
    with tarfile.open(path) as archive:
        assert archive.getmember("bundle/chat_template.jinja").isfile()
        archive.extractall(tmp_path / "extracted", filter="data")
    _assert_source_assets(source, tmp_path / "extracted" / "bundle")


@pytest.mark.parametrize("script", ["quantize.py", "quantize_nbits.py"])
@pytest.mark.parametrize("link", ["symlink", "hardlink"])
@pytest.mark.parametrize("external_asset_dependency", [False, True])
def test_quantizer_cli_preserves_retained_tokenizer_links(tmp_path, script, link, external_asset_dependency):
    source = _source(tmp_path / "source", "embedded")
    target = tmp_path / "target"
    target.mkdir()
    weight = np.ones((64, 64), dtype=np.float32) if script == "quantize_nbits.py" else None
    for name in ["decoder_init.onnx", "decoder_step.onnx"]:
        _write_model(source, name, external=False, weight=weight)
    if external_asset_dependency:
        _write_external_reference(source, "aux.onnx", "tokenizer_config.json")
    retained = tmp_path / "retained"
    retained.mkdir()
    before = {}
    for name in tokenizer_asset_names(source):
        previous = retained / name
        previous.write_bytes(("retained-" + name).encode())
        before[name] = previous.read_bytes()
        if link == "symlink":
            (target / name).symlink_to(previous)
        else:
            os.link(previous, target / name)
    if script == "quantize.py":
        command = _converter_command(tmp_path, script, source, target)
    else:
        command = [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(source),
            "--output",
            str(target),
            "--bits",
            "4",
            "--block-size",
            "32",
        ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {name: (retained / name).read_bytes() for name in before} == before
    assert all((target / name).read_bytes() == (source / name).read_bytes() for name in tokenizer_asset_names(source))


def test_nbits_invalid_tokenizer_rejected_before_output(tmp_path):
    source = _source(tmp_path / "source", "jinja")
    target = tmp_path / "target"
    for name in ["decoder_init.onnx", "decoder_step.onnx"]:
        _write_model(source, name, external=False)
    (source / "chat_template.jinja").unlink()
    command = [
        sys.executable,
        str(ROOT / "quantize_nbits.py"),
        "--input",
        str(source),
        "--output",
        str(target),
        "--bits",
        "4",
        "--block-size",
        "32",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert not target.exists()


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py", "quantize_nbits.py"])
def test_converter_cli_copies_external_encoder_closure(tmp_path, script):
    source = _source(tmp_path / "source", "embedded")
    (source / "weights").mkdir()
    _write_model(source, "encoder.onnx", external=True, data_name="weights/encoder.data")
    if script == "quantize_nbits.py":
        for name in ("decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, weight=np.ones((64, 64), dtype=np.float32))
    target = tmp_path / "target"
    if script == "quantize_nbits.py":
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
    assert (target / "weights" / "encoder.data").read_bytes() == (source / "weights" / "encoder.data").read_bytes()
    onnx.load(target / "encoder.onnx")


@pytest.mark.parametrize("script", ["quantize.py", "quantize_nbits.py"])
def test_quantizer_cli_copies_nested_decoder_dependencies(tmp_path, script):
    source = _source(tmp_path / "source", "embedded")
    (source / "weights").mkdir()
    weight = np.ones((64, 64), dtype=np.float32) if script == "quantize_nbits.py" else None
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        _write_model(source, name, external=True, data_name=f"weights/{name}.data", weight=weight)
    target = tmp_path / "target"
    if script == "quantize_nbits.py":
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
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        assert (target / "weights" / f"{name}.data").read_bytes() == (source / "weights" / f"{name}.data").read_bytes()
        onnx.load(target / name)


def test_local_tokenizer_metadata_cannot_execute_python(tmp_path):
    source = _source(tmp_path / "source", "embedded")
    marker = tmp_path / "executed"
    (source / "tokenizer_probe.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "from transformers import PreTrainedTokenizerFast\n"
        "class ProbeTokenizer(PreTrainedTokenizerFast): pass\n"
    )
    config_path = source / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config["auto_map"] = {"AutoTokenizer": ["tokenizer_probe.ProbeTokenizer", None]}
    config_path.write_text(json.dumps(config))
    validate_tokenizer_assets(source)
    assert not marker.exists()


def _write_external_reference(root, name, location):
    tensor = numpy_helper.from_array(np.ones((1, 1), dtype=np.float32), "w")
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", "0"), ("length", "4")):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "external-reference",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1])],
        [tensor],
    )
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10), root / name)


@pytest.mark.parametrize("script", ["quantize.py", "quantize_nbits.py"])
def test_quantizer_rejects_graph_dependency_on_generated_output_before_writes(tmp_path, script):
    source = _source(tmp_path / "source", "embedded")
    suffix = "int8" if script == "quantize.py" else "int4"
    reserved = f"decoder_init.{suffix}.onnx"
    (source / reserved).write_bytes(b"source bytes claimed by auxiliary graph")
    _write_external_reference(source, "aux.onnx", reserved)
    if script == "quantize_nbits.py":
        for name in ("decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, weight=np.ones((64, 64), dtype=np.float32))
        command = [
            sys.executable,
            str(ROOT / script),
            "--input",
            str(source),
            "--output",
            str(tmp_path / "target"),
            "--bits",
            "4",
            "--block-size",
            "32",
        ]
    else:
        command = _converter_command(tmp_path, script, source, tmp_path / "target")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert "conflict" in (result.stdout + result.stderr).lower()
    assert not (tmp_path / "target").exists()
    assert (source / reserved).read_bytes() == b"source bytes claimed by auxiliary graph"


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py"])
def test_converter_rejects_encoder_dependency_on_generated_decoder_before_writes(tmp_path, script):
    source = _source(tmp_path / "source", "embedded")
    _write_external_reference(source, "encoder.onnx", "decoder_init.onnx")
    target = tmp_path / "target"
    command = _converter_command(tmp_path, script, source, target)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert "conflict" in (result.stdout + result.stderr).lower()
    assert not target.exists()


def test_inplace_quantizer_replaces_linked_config_without_mutating_retained_bytes(tmp_path):
    source = _source(tmp_path / "source", "embedded")
    retained = tmp_path / "retained.json"
    (source / "config.json").replace(retained)
    original = retained.read_bytes()
    (source / "config.json").symlink_to(retained)
    command = _converter_command(tmp_path, "quantize.py", source, source)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert retained.read_bytes() == original
    assert not (source / "config.json").is_symlink()
    assert json.loads((source / "config.json").read_text())["quantization"]["decoder_init"] == "int8_dynamic"


@pytest.mark.parametrize("script,suffix", [("quantize.py", "int8"), ("quantize_nbits.py", "int4")])
@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_quantizer_rejects_linked_generated_graph_before_writes(tmp_path, script, suffix, link):
    source = _source(tmp_path / "source", "embedded")
    if script == "quantize_nbits.py":
        for name in ("decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, weight=np.ones((64, 64), dtype=np.float32))
    target = tmp_path / "target"
    target.mkdir()
    retained = tmp_path / "retained.onnx"
    retained.write_bytes(b"retained graph")
    generated = target / f"decoder_init.{suffix}.onnx"
    if link == "symlink":
        generated.symlink_to(retained)
    else:
        os.link(retained, generated)
    if script == "quantize.py":
        command = _converter_command(tmp_path, script, source, target)
    else:
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
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert "destination conflict" in result.stdout + result.stderr
    assert retained.read_bytes() == b"retained graph"
    assert not (target / f"decoder_step.{suffix}.onnx").exists()


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py"])
def test_conversion_to_package_transports_metadata(tmp_path, script):
    import package as pack

    source = _source(tmp_path / "source", "embedded")
    (source / "preprocessor_config.json").write_text('{"feature_size": 128, "sampling_rate": 16000}\n')
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "converted"
    result = subprocess.run(
        _converter_command(tmp_path, script, source, output), capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "preprocessor_config.json").read_bytes() == before[Path("preprocessor_config.json")]
    assert pack.package(output, tmp_path / "release")
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("asset", ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"])
@pytest.mark.parametrize("chained", [False, True])
def test_family_copy_preserves_source_access_through_stale_destination(tmp_path, asset, chained):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    target.mkdir()
    source_asset = source / asset
    retained = target / "vocab.json"
    retained.write_bytes(source_asset.read_bytes())
    source_asset.unlink()
    if chained:
        bridge = tmp_path / "bridge"
        bridge.symlink_to(retained)
        source_asset.symlink_to(bridge)
    else:
        source_asset.symlink_to(retained)
    assert not (source / "vocab.json").exists()
    validate_tokenizer_assets(source)
    before_source = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    before_target = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="source access path"):
        copy_tokenizer_assets(source, target)
    assert source_asset.is_symlink()
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == before_source
    assert {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()} == before_target
    validate_tokenizer_assets(source)


def test_family_copy_can_remove_distinct_hardlink_to_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    target.mkdir()
    retained = target / "vocab.json"
    os.link(source / "tokenizer.json", retained)
    before = (source / "tokenizer.json").read_bytes()
    copy_tokenizer_assets(source, target)
    assert not retained.exists()
    assert (source / "tokenizer.json").read_bytes() == before
    validate_tokenizer_assets(source)
    _assert_source_assets(source, target)


@pytest.mark.parametrize("chained", [False, True])
def test_family_copy_preserves_source_access_through_replaced_destination(tmp_path, chained):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    target.mkdir()
    source_config = source / "tokenizer_config.json"
    retained = target / "tokenizer.json"
    retained.write_bytes(source_config.read_bytes())
    source_config.unlink()
    if chained:
        bridge = tmp_path / "bridge"
        bridge.symlink_to(retained)
        source_config.symlink_to(bridge)
    else:
        source_config.symlink_to(retained)
    validate_tokenizer_assets(source)
    before_source = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    before_target = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="source access path"):
        copy_tokenizer_assets(source, target)
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == before_source
    assert {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()} == before_target
    validate_tokenizer_assets(source)


@pytest.mark.parametrize("chained", [False, True])
def test_family_copy_preserves_model_config_read_by_large_tokenizer(tmp_path, monkeypatch, chained):
    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    tokenizer_path = source / "tokenizer.json"
    document = json.loads(tokenizer_path.read_text())
    document["model"]["vocab"].update({f"word_{index}": index + 200 for index in range(100001)})
    tokenizer_path.write_text(json.dumps(document))
    target.mkdir()
    retained = target / "vocab.json"
    retained.write_text(json.dumps({"model_type": "qwen3_asr", "retained_model_config": "original"}))
    model_config = source / "config.json"
    if chained:
        bridge = tmp_path / "bridge"
        bridge.symlink_to(retained)
        model_config.symlink_to(bridge)
    else:
        model_config.symlink_to(retained)
    opened = []
    original_open = builtins.open

    def observe_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).absolute() == model_config.absolute():
            opened.append(file)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", observe_open)
    tokenizer = validate_tokenizer_assets(source)
    assert tokenizer.vocab_size > 100000
    assert opened
    before_source = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    before_target = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}
    with pytest.raises(ValueError, match="source access path"):
        copy_tokenizer_assets(source, target)
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == before_source
    assert {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()} == before_target
    validate_tokenizer_assets(source)
