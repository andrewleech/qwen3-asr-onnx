import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def source(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    rng = np.random.default_rng(44)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w1"], ["h"], name="projection1"),
            helper.make_node("MatMul", ["h", "w2"], ["y"], name="projection2"),
        ],
        "precision",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 32])],
        [
            numpy_helper.from_array(rng.normal(size=(64, 64)).astype(np.float32), "w1"),
            numpy_helper.from_array(rng.normal(size=(64, 32)).astype(np.float32), "w2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    for name in ["decoder_init", "decoder_step"]:
        onnx.save(model, inputs / f"{name}.onnx")
    (inputs / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_asr",
                "decoder": {"vocab_size": 4, "hidden_size": 64},
                "embed_tokens_shape": [4, 64],
                "embed_tokens_dtype": "float32",
            }
        )
    )
    calib = tmp_path / "calib.npz"
    np.savez(calib, _n_samples=40, **{f"{i}_x": rng.normal(size=(2, 64)).astype(np.float32) for i in range(40)})
    return inputs, calib
