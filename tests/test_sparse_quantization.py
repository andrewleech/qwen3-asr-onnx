import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from quantize_nbits import _quantization_input
from src.onnx_artifacts import isolated_onnx_input


@pytest.mark.parametrize("reader", ["shared", "rtn", "gptq"])
@pytest.mark.parametrize("storage", ["values", "indices"])
@pytest.mark.parametrize("placement", ["root", "subgraph", "function"])
def test_quantizer_input_rejects_external_sparse_storage(tmp_path, storage, placement, reader):
    values = numpy_helper.from_array(np.array([3], dtype=np.float32), "weight")
    indices = numpy_helper.from_array(np.array([0], dtype=np.int64))
    sparse = helper.make_sparse_tensor(values, indices, [1])
    tensor = getattr(sparse, storage)
    location = "sparse.data"
    (tmp_path / location).write_bytes(tensor.raw_data)
    onnx.external_data_helper.set_external_data(tensor, location, offset=0, length=len(tensor.raw_data))
    tensor.ClearField("raw_data")
    output = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])
    if placement == "root":
        graph = helper.make_graph([helper.make_node("Identity", ["weight"], ["out"])], "root", [], [output])
        graph.sparse_initializer.append(sparse)
        functions = []
    else:
        constant = helper.make_node("Constant", [], ["out"], sparse_value=sparse)
        body = helper.make_graph([constant], "nested", [], [output])
        if placement == "subgraph":
            condition = numpy_helper.from_array(np.array(True), "condition")
            graph = helper.make_graph(
                [helper.make_node("If", ["condition"], ["out"], then_branch=body, else_branch=body)],
                "root",
                [],
                [output],
                [condition],
            )
            functions = []
        else:
            graph = helper.make_graph([helper.make_node("Sparse", [], ["out"], domain="fixture")], "root", [], [output])
            functions = [
                helper.make_function("fixture", "Sparse", [], ["out"], [constant], [helper.make_opsetid("", 18)])
            ]
    model = helper.make_model(graph, functions=functions, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    path = tmp_path / "input.onnx"
    path.write_bytes(model.SerializeToString())
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    input_context = isolated_onnx_input(path) if reader == "shared" else _quantization_input(str(path), reader)
    with pytest.raises(ValueError, match="external tensor storage is unsupported"), input_context:
        pytest.fail("Unsupported storage reached the delegated writer")
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
