import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

from src.prompt import build_prompt_ids, load_prompt_tokenizer
from tests.tokenizer_fixtures import save_tokenizer_assets

ROOT = Path(__file__).resolve().parents[1]


def make_prompt_artifact(root, template_format):
    tokenizer = save_tokenizer_assets(root, template_format)
    config = {"decoder": {"vocab_size": 8, "hidden_size": 4}, "embed_tokens_shape": [8, 4]}
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text("{}")
    np.arange(32, dtype=np.float32).tofile(root / "embed_tokens.bin")
    for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"):
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"])],
            "prompt-transport",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])],
            [numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3) / 16, "w")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
        onnx.save(model, root / name)
    return tokenizer


@pytest.mark.parametrize("template_format", ["embedded", "json", "jinja"])
@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "package.py"])
def test_prompt_survives_artifact_cli(tmp_path, template_format, script):
    source, target = tmp_path / "source", tmp_path / "target"
    tokenizer = make_prompt_artifact(source, template_format)
    command = [sys.executable, str(ROOT / script), "--input", str(source), "--output", str(target)]
    if script != "package.py":
        command += ["--no-share-weights"]
    if script == "quantize_static.py":
        cache = tmp_path / "cache"
        cache.mkdir()
        collect = (
            "import numpy as np; import quantize_static; "
            "quantize_static.collect_calibration_data = lambda *a, **k: "
            "([{ 'x': np.ones((1, 4), dtype=np.float32) }], [{ 'x': np.ones((1, 4), dtype=np.float32) }]); "
            "quantize_static.main()"
        )
        command = [sys.executable, "-c", collect, *command[2:]]
        command += ["--cache-dir", str(cache), "--n-samples", "1", "--n-steps", "1"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if script == "quantize_static.py":
        assert (cache / "calib_n1_s1.npz").is_file()
    actual = load_prompt_tokenizer(str(target), local_files_only=True)
    assert build_prompt_ids(3, actual, language="English", context="Kepler") == build_prompt_ids(
        3, tokenizer, language="English", context="Kepler"
    )


@pytest.mark.parametrize("template_format", ["embedded", "json", "jinja"])
def test_standalone_archive_preserves_prompt(tmp_path, template_format):
    from create_tar import create_tar

    source = tmp_path / "bundle"
    tokenizer = make_prompt_artifact(source, template_format)
    output = tmp_path / "archives"
    output.mkdir()
    path = create_tar({"name": "bundle", "source_dir": str(source), "files": []}, str(output))
    assert path is not None
    extracted = tmp_path / "extracted"
    with tarfile.open(path) as archive:
        archive.extractall(extracted, filter="data")
    actual = load_prompt_tokenizer(str(extracted / "bundle"), local_files_only=True)
    assert build_prompt_ids(3, actual) == build_prompt_ids(3, tokenizer)


@pytest.mark.parametrize("representation", ["json", "embedded"])
@pytest.mark.parametrize(
    "script", ["convert_fp16.py", "quantize_static.py", "package.py", "quantize.py", "quantize_nbits.py"]
)
def test_conflicting_prompt_representation_fails_before_output(tmp_path, representation, script):
    source, target = tmp_path / "source", tmp_path / "target"
    make_prompt_artifact(source, "jinja")
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


def test_export_rejects_stale_processor_template_before_writing(tmp_path):
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
    legacy = target / "chat_template.json"
    legacy.write_text(json.dumps({"chat_template": "WRONG PROMPT"}))
    before = {path.name: path.read_bytes() for path in target.iterdir()}
    with pytest.raises(ValueError, match="clean output directory"):
        export.copy_tokenizer(str(checkpoint), str(target))
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    legacy.unlink()
    export.copy_tokenizer(str(checkpoint), str(target))
    export.write_preprocessor_config(str(target))
    assert not (target / "chat_template.json").exists()
    tokenizer = load_prompt_tokenizer(str(target), local_files_only=True)
    processor = Qwen3ASRProcessor.from_pretrained(str(target), local_files_only=True)
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    assert processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ) == tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize.py", "quantize_nbits.py"])
def test_conflicting_reused_target_is_rejected_before_conversion(tmp_path, script):
    source, target = tmp_path / "source", tmp_path / "target"
    make_prompt_artifact(source, "jinja")
    save_tokenizer_assets(target, "jinja")
    (target / "chat_template.json").write_text(json.dumps({"chat_template": "WRONG PROMPT"}))
    before = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--input", str(source), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode != 0
    assert "Invalid ASR tokenizer assets" in result.stdout + result.stderr
    assert {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()} == before


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
