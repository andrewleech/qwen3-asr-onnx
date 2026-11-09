import json

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from tests.tokenizer_fixtures import save_tokenizer_assets


def _write_model(root, name, *, external=False, data_name=None, weight=None):
    weight = np.arange(512, dtype=np.float32).reshape(32, 16) if weight is None else weight
    tensor = numpy_helper.from_array(weight, name="weight")
    dtype = helper.np_dtype_to_tensor_dtype(weight.dtype)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", dtype, [1, weight.shape[0]])],
        [helper.make_tensor_value_info("y", dtype, [1, weight.shape[1]])],
        [tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.save_model(
        model,
        root / name,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location=data_name or f"{name}.data",
        size_threshold=0,
    )


def _release_input(root, dtype="float32", *, external=False):
    root.mkdir()
    for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"):
        _write_model(root, name, external=external)
    config = {
        "decoder": {"vocab_size": 8, "hidden_size": 4},
        "embed_tokens_shape": [8, 4],
        "embed_tokens_dtype": dtype,
    }
    (root / "preprocessor_config.json").write_text("{}")
    save_tokenizer_assets(root)
    (root / "config.json").write_text(json.dumps(config))
    np.arange(32, dtype=dtype).reshape(8, 4).tofile(root / "embed_tokens.bin")
    return root


def _ort_result(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    feed = {
        item.name: np.ones(item.shape, dtype=np.float16 if item.type == "tensor(float16)" else np.float32)
        for item in session.get_inputs()
    }
    return session.run(None, feed)[0]


def _write_external_identity(root, name, location, offset=0, dtype=np.float32):
    from share_weights import set_tensor_external

    tensor = numpy_helper.from_array(np.zeros(1, dtype=dtype), name="value")
    set_tensor_external(tensor, location, offset, np.dtype(dtype).itemsize)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["value"], ["output"])],
        "external-identity",
        [],
        [helper.make_tensor_value_info("output", tensor.data_type, [1])],
        [tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    (root / name).write_bytes(model.SerializeToString())
