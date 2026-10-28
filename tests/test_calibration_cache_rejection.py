import json
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import awq_smooth
import quantize_static
from tests.tokenizer_fixtures import save_tokenizer_assets


def _source(root):
    save_tokenizer_assets(root)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])],
        "cache-input",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(np.eye(4, dtype=np.float32), "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    for name in ("encoder", "decoder_init", "decoder_step"):
        onnx.save(model, root / f"{name}.onnx")
    (root / "config.json").write_text(
        json.dumps(
            {
                "decoder": {"vocab_size": 8, "hidden_size": 4},
                "embed_tokens_shape": [8, 4],
                "embed_tokens_dtype": "float32",
            }
        )
    )
    np.zeros((8, 4), dtype=np.float32).tofile(root / "embed_tokens.bin")
    return root


@pytest.mark.parametrize("stem", [False, True])
@pytest.mark.parametrize("dangling", [False, True])
def test_awq_rejects_unverifiable_existing_cache_before_model_or_writes(tmp_path, monkeypatch, dangling, stem):
    output = tmp_path / "output"
    output.mkdir()
    cache = output / "calibration_activations.npz"
    if dangling:
        cache.symlink_to(tmp_path / "missing.npz")
    else:
        np.savez(cache, norm=np.full(4, 99, dtype=np.float32))
    before = None if dangling else cache.read_bytes()
    monkeypatch.setattr(awq_smooth, "load_model", lambda *a, **k: pytest.fail("Unknown cache reached model loading"))
    args = ["awq_smooth.py", "--model", "fixture", "--output", str(output)]
    if stem:
        args += ["--activations-cache", str(cache.with_suffix(""))]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="Unverifiable calibration cache"):
        awq_smooth.main()
    if dangling:
        assert cache.is_symlink() and cache.readlink() == tmp_path / "missing.npz"
    else:
        assert cache.read_bytes() == before
    assert not (output / "smoothing_scales.npz").exists()


def _static_args(source, output, cache):
    return [
        "quantize_static.py",
        "--input",
        str(source),
        "--output",
        str(output),
        "--cache-dir",
        str(cache),
        "--n-samples",
        "1",
        "--n-steps",
        "1",
        "--no-share-weights",
    ]


def test_static_rejects_unverifiable_cache_before_collection_or_quantization(tmp_path, monkeypatch):
    source = _source(tmp_path / "source")
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "calib_n1_s1.npz"
    samples = np.array([{"x": np.full((1, 4), 99, dtype=np.float32)}], dtype=object)
    np.savez(path, init_inputs=samples, step_inputs=samples)
    before = path.read_bytes()
    monkeypatch.setattr(
        quantize_static, "collect_calibration_data", lambda *a, **k: pytest.fail("Unknown cache reached collection")
    )
    monkeypatch.setattr(
        quantize_static, "quantize_decoder", lambda *a, **k: pytest.fail("Unknown cache reached quantization")
    )
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", _static_args(source, output, cache))
    with pytest.raises(ValueError, match="Unverifiable calibration cache"):
        quantize_static.main()
    assert path.read_bytes() == before
    assert not list(output.glob("*.onnx"))


def test_static_clean_path_collects_and_quantizes_then_preserves_existing_cache(tmp_path, monkeypatch):
    source = _source(tmp_path / "source")
    cache, output = tmp_path / "cache", tmp_path / "output"
    collected = []

    def collect(*args, **kwargs):
        collected.append(kwargs)
        sample = {"x": np.ones((1, 4), dtype=np.float32)}
        return [sample], [sample]

    monkeypatch.setattr(quantize_static, "collect_calibration_data", collect)
    monkeypatch.setattr(sys, "argv", _static_args(source, output, cache))
    quantize_static.main()
    assert len(collected) == 1
    path = cache / "calib_n1_s1.npz"
    previous = path.read_bytes()
    generated = {p.name: p.read_bytes() for p in output.glob("*.onnx")}
    assert {"decoder_init.onnx", "decoder_step.onnx"} <= generated.keys()
    with pytest.raises(ValueError, match="Unverifiable calibration cache"):
        quantize_static.main()
    assert len(collected) == 1 and path.read_bytes() == previous
    assert {p.name: p.read_bytes() for p in output.glob("*.onnx")} == generated
