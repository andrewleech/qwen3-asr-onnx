import json
import sys
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

import awq_smooth
import collect_gptq_calib
from src import prompt
from tests.artifact_fixtures import _release_input
from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE, save_tokenizer_assets


def _change_template(root, representation, template):
    if representation == "jinja":
        (root / "chat_template.jinja").write_text(template)
    else:
        path = root / ("chat_template.json" if representation == "json" else "tokenizer_config.json")
        config = json.loads(path.read_text())
        config["chat_template"] = template
        path.write_text(json.dumps(config))


@pytest.mark.parametrize("representation", ["jinja", "json", "embedded"])
@pytest.mark.parametrize("role,replacement", [("system", "user"), ("user", "system"), ("assistant", "system")])
def test_loader_rejects_incorrect_roles_with_valid_audio(tmp_path, representation, role, replacement):
    tokenizer = save_tokenizer_assets(tmp_path, representation)
    bad_template = OFFICIAL_CHAT_TEMPLATE.replace(role + "\\n", replacement + "\\n")
    tokenizer.chat_template = bad_template
    with pytest.raises(ValueError, match="prompt contract"):
        prompt.build_prompt_ids(1, tokenizer)
    _change_template(tmp_path, representation, bad_template)
    with pytest.raises(ValueError, match="prompt contract"):
        prompt.load_prompt_tokenizer(str(tmp_path), local_files_only=True)


def test_calibration_rejects_bad_roles_without_replacing_prior_data(tmp_path, monkeypatch):
    source = _release_input(tmp_path / "model")
    features = numpy_helper.from_array(np.ones((1, 1, 4), dtype=np.float32))
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["audio_features"], value=features)],
        "local_encoder",
        [helper.make_tensor_value_info("mel", onnx.TensorProto.FLOAT, [1, 128, "frames"])],
        [helper.make_tensor_value_info("audio_features", onnx.TensorProto.FLOAT, [1, 1, 4])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10), source / "encoder.onnx"
    )
    monkeypatch.setattr(collect_gptq_calib, "stream_audio", lambda count: iter([np.zeros(8000, dtype=np.float32)]))
    target = tmp_path / "calibration.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_gptq_calib.py",
            "--model",
            str(source),
            "--target",
            "decoder_init",
            "--n-samples",
            "1",
            "--threads",
            "1",
            "--output",
            str(target),
        ],
    )
    collect_gptq_calib.main()
    with np.load(target) as data:
        assert data["_n_samples"] == 1
        np.testing.assert_array_equal(
            data["0_input_ids"], [prompt.build_prompt_ids(1, prompt.load_prompt_tokenizer(str(source)))]
        )
    previous = target.read_bytes()
    _change_template(source, "jinja", OFFICIAL_CHAT_TEMPLATE.replace("assistant\\n", "system\\n"))
    with pytest.raises(ValueError, match="prompt contract"):
        collect_gptq_calib.main()
    assert target.read_bytes() == previous


def test_awq_rejects_bad_roles_before_calibration_persistence(tmp_path, monkeypatch):
    source, output = tmp_path / "model", tmp_path / "output"
    save_tokenizer_assets(source)
    _change_template(source, "jinja", OFFICIAL_CHAT_TEMPLATE.replace("assistant\\n", "system\\n"))
    output.mkdir()
    cache = output / "calibration_activations.npz"
    cache.write_bytes(b"retained activation cache")
    monkeypatch.setattr(awq_smooth, "load_model", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(sys, "argv", ["awq_smooth.py", "--model", str(source), "--output", str(output)])
    with pytest.raises(ValueError, match="prompt contract"):
        awq_smooth.main()
    assert cache.read_bytes() == b"retained activation cache"
    assert not (output / "smoothing_scales.npz").exists()
