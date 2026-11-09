import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from tests.artifact_fixtures import _write_model
from tests.tokenizer_fixtures import save_tokenizer_assets

ROOT = Path(__file__).resolve().parents[1]


def _source(root, template_format):
    save_tokenizer_assets(root, template_format=template_format)
    config = {
        "decoder": {"vocab_size": 8, "hidden_size": 4},
        "embed_tokens_shape": [8, 4],
        "embed_tokens_dtype": "float32",
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text("{}")
    np.arange(32, dtype=np.float32).tofile(root / "embed_tokens.bin")
    for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"):
        weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 16
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"])],
            "toy",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])],
            [numpy_helper.from_array(weight, "w")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
        onnx.save_model(model, root / name)
    return root


def _converter_command(tmp_path, script, source, target):
    command = [
        sys.executable,
        str(ROOT / script),
        "--input",
        str(source),
        "--output",
        str(target),
        "--no-share-weights",
    ]
    if script == "quantize_static.py":
        cache = tmp_path / "cache"
        cache.mkdir()
        from quantize_static import calibration_identity
        from src.calibration_cache import cache_key, pack_samples, save_cache

        identity = calibration_identity(source, 1, 1)
        inputs = [{"x": np.ones((1, 4), dtype=np.float32)}]
        save_cache(cache / f"calib_{cache_key(identity)}.npz", identity, pack_samples({"init": inputs, "step": inputs}))
        command += ["--cache-dir", str(cache), "--n-samples", "1", "--n-steps", "1"]
    return command


SCRIPTS = ["convert_fp16.py", "quantize_static.py", "quantize.py", "quantize_nbits.py"]


def command(root, script, source, target):
    if script == "quantize_nbits.py":
        return [
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
    return _converter_command(root, script, source, target)


def input_for(root, script):
    source = _source(root / "source", "embedded")
    if script == "quantize_nbits.py":
        for name in ("decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, weight=np.ones((64, 64), dtype=np.float32))
    return source


def bytes_under(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
