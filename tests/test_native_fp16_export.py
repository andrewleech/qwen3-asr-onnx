import copy
import sys
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

import export
import export_encoder_native_fp16 as native_fp16
from src.native_encoder import run_native_encoder
from tests.encoder_helpers import FRAME_LENGTHS


@pytest.mark.parametrize("retained_data", [False, True])
def test_native_fp16_main_exports_dynamic_fp32_io(tmp_path, monkeypatch, capsys, retained_data):
    with torch.random.fork_rng():
        torch.manual_seed(4342)
        config = Qwen3ASRAudioEncoderConfig(
            d_model=32,
            encoder_attention_heads=4,
            encoder_layers=2,
            encoder_ffn_dim=64,
            downsample_hidden_size=4,
            output_dim=16,
            n_window=50,
            n_window_infer=800,
            initializer_range=0.2,
        )
        config._attn_implementation = "eager"
        model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=Qwen3ASRAudioEncoder(config).eval()))

        reference = copy.deepcopy(model.thinker.audio_tower).half()

        def load_tiny_model(model_name, dtype):
            assert model_name == "fixture/tiny"
            assert dtype == torch.float32
            return model

        path = tmp_path / "encoder.fp16.onnx"
        if retained_data:
            tensor = onnx.numpy_helper.from_array(np.array([42], dtype=np.float32), "retained")
            onnx.external_data_helper.set_external_data(tensor, path.name + ".data", offset=0, length=4)
            tensor.ClearField("raw_data")
            aux = onnx.helper.make_model(
                onnx.helper.make_graph(
                    [onnx.helper.make_node("Identity", ["retained"], ["out"])],
                    "retained",
                    [],
                    [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1])],
                    [tensor],
                ),
                opset_imports=[onnx.helper.make_opsetid("", 17)],
                ir_version=9,
            )
            (tmp_path / (path.name + ".data")).write_bytes(np.array([42], dtype=np.float32).tobytes())
            onnx.save(aux, tmp_path / "aux.onnx")
        monkeypatch.setattr(export, "load_model", load_tiny_model)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "export_encoder_native_fp16.py",
                "--model",
                "fixture/tiny",
                "--output",
                str(path),
                "--opset",
                "17",
                "--verify",
            ],
        )
        native_fp16.main()

    if retained_data:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(tmp_path / "aux.onnx"), options, providers=["CPUExecutionProvider"])
        assert float(session.run(None, {})[0][0]) == 42.0
    assert "ORT OK:" in capsys.readouterr().out
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert [(item.domain, item.version) for item in graph.opset_import] == [("", 17)]
    assert [(item.name, item.type.tensor_type.elem_type) for item in graph.graph.input] == [
        ("mel", onnx.TensorProto.FLOAT)
    ]
    assert [(item.name, item.type.tensor_type.elem_type) for item in graph.graph.output] == [
        ("audio_features", onnx.TensorProto.FLOAT)
    ]
    assert any(item.data_type == onnx.TensorProto.FLOAT16 for item in graph.graph.initializer)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for frames in FRAME_LENGTHS:
        mel = np.random.default_rng(frames).standard_normal((1, 128, frames), dtype=np.float32)
        features = session.run(["audio_features"], {"mel": mel})[0]
        expected_tokens = (frames // 100) * 13 + ((frames % 100) + 7) // 8
        assert features.shape == (1, expected_tokens, 16)
        assert features.dtype == np.float32
        with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.float16):
            expected = run_native_encoder(reference, torch.from_numpy(mel).half()).float().numpy()
        _assert_fp16_features(features, expected)
        with pytest.raises(AssertionError):
            _assert_fp16_features(features * 1.01, expected)
        with pytest.raises(AssertionError):
            _assert_fp16_features(np.zeros_like(features), expected)


def _assert_fp16_features(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    # Tiny CPU fixtures show up to 3.66 epsilon absolute and 1.83 epsilon relative L2
    # from half rounding across two attention layers; these budgets are not release tolerances.
    epsilon = np.finfo(np.float16).eps
    assert np.max(np.abs(actual - expected)) < 5 * epsilon
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 3 * epsilon
