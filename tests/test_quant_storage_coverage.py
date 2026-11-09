import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper

from quantize_nbits import _quantization_input, quantize_decoder
from src.onnx_artifacts import external_data_files, isolated_onnx_input, validate_quantization_storage
from tests.artifact_fixtures import _write_model
from tests.quantization_fixtures import source


def _input_with_external_metadata(tmp_path, placement, duplicate=False):
    inputs, calibration = source(tmp_path)
    path = inputs / "decoder_init.onnx"
    model = onnx.load(path)
    tensor = (
        model.graph.initializer[0]
        if duplicate
        else numpy_helper.from_array(np.ones((1, 32), dtype=np.float32), "extra")
    )
    (inputs / "extra.data").write_bytes(tensor.raw_data)
    onnx.external_data_helper.set_external_data(tensor, location="extra.data", offset=0, length=len(tensor.raw_data))
    tensor.ClearField("raw_data")
    if placement == "training":
        training = model.training_info.add()
        training.algorithm.CopyFrom(
            helper.make_graph(
                [helper.make_node("Identity", [tensor.name], ["trained"])],
                "training",
                [],
                [helper.make_tensor_value_info("trained", tensor.data_type, list(tensor.dims))],
                [tensor],
            )
        )
    else:
        function = helper.make_function(
            "fixture",
            "Unused",
            ["x"],
            ["y"],
            [helper.make_node("Identity", ["x"], ["y"])],
            [helper.make_opsetid("", 18)],
        )
        function.attribute_proto.append(helper.make_attribute("default_tensor", tensor))
        model.functions.append(function)
        model.opset_import.append(helper.make_opsetid("fixture", 1))
    path.write_bytes(model.SerializeToString())
    onnx.checker.check_model(str(path))
    return path, calibration


def _snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("placement", ["training", "defaults"])
@pytest.mark.parametrize("reader", ["shared", "rtn", "gptq"])
def test_quantizer_rejects_unsupported_external_metadata(tmp_path, placement, reader):
    path, _ = _input_with_external_metadata(tmp_path, placement)
    before = _snapshot(path.parent)
    context = isolated_onnx_input(path) if reader == "shared" else _quantization_input(str(path), reader)
    with pytest.raises(ValueError, match="external tensor storage is unsupported"), context:
        pytest.fail("Unsupported metadata reached quantization")
    assert _snapshot(path.parent) == before


@pytest.mark.parametrize("placement", ["training", "defaults"])
@pytest.mark.parametrize("algorithm", ["rtn", "gptq"])
def test_quantizer_preserves_previous_output_on_unsupported_metadata(tmp_path, placement, algorithm):
    path, calibration = _input_with_external_metadata(tmp_path, placement)
    output = tmp_path / "output"
    output.mkdir()
    _write_model(output, "model.int4.onnx")
    previous = _snapshot(output)
    original = _snapshot(path.parent)
    with pytest.raises(ValueError, match="external tensor storage is unsupported"):
        quantize_decoder(
            str(path),
            str(output / "model.int4.onnx"),
            4,
            32,
            4,
            algorithm,
            str(calibration) if algorithm == "gptq" else None,
        )
    assert _snapshot(output) == previous
    assert _snapshot(path.parent) == original


@pytest.mark.parametrize("placement", ["training", "defaults"])
def test_identical_supported_tensor_does_not_hide_unsupported_occurrence(tmp_path, placement):
    path, _ = _input_with_external_metadata(tmp_path, placement, duplicate=True)
    model = onnx.load(path, load_external_data=False)
    extra = (
        model.training_info[0].algorithm.initializer[0]
        if placement == "training"
        else model.functions[0].attribute_proto[0].t
    )
    assert extra.SerializeToString() == model.graph.initializer[0].SerializeToString()
    with pytest.raises(ValueError, match="external tensor storage is unsupported"):
        validate_quantization_storage(model)


def test_supported_function_tensor_storage_is_preserved(tmp_path):
    inputs, _ = source(tmp_path)
    path = inputs / "decoder_init.onnx"
    model = onnx.load(path)
    tensor = numpy_helper.from_array(np.array([3], dtype=np.float32), "extra")
    (inputs / "extra.data").write_bytes(tensor.raw_data)
    onnx.external_data_helper.set_external_data(tensor, location="extra.data", offset=0, length=len(tensor.raw_data))
    tensor.ClearField("raw_data")
    function = helper.make_function(
        "fixture",
        "Unused",
        ["x"],
        ["y"],
        [helper.make_node("Constant", [], ["unused"], value=tensor), helper.make_node("Identity", ["x"], ["y"])],
        [helper.make_opsetid("", 18)],
    )
    model.functions.append(function)
    model.opset_import.append(helper.make_opsetid("fixture", 1))
    path.write_bytes(model.SerializeToString())
    original = _snapshot(inputs)
    with isolated_onnx_input(path) as (staged, _):
        dependencies = external_data_files(staged)
        assert len(dependencies) == 1
        assert next(iter(dependencies)).read_bytes() == (inputs / "extra.data").read_bytes()
    assert _snapshot(inputs) == original


@pytest.mark.parametrize("reader", ["shared", "rtn", "gptq"])
def test_graph_file_alias_preserves_lexical_external_data_base(tmp_path, reader):
    physical, lexical = tmp_path / "physical", tmp_path / "lexical"
    physical.mkdir()
    lexical.mkdir()
    _write_model(
        physical, "model.onnx", external=True, data_name="weights.data", weight=np.full((4, 3), 99, np.float32)
    )
    (lexical / "model.onnx").symlink_to(physical / "model.onnx")
    (lexical / "weights.data").write_bytes(np.full((4, 3), 42, np.float32).tobytes())
    path = lexical / "model.onnx"
    expected = onnx.load(path).graph.initializer[0]
    assert external_data_files(path) == {lexical / "weights.data"}
    context = isolated_onnx_input(path) if reader == "shared" else _quantization_input(str(path), reader)
    before = _snapshot(tmp_path)
    with context as (model, _):
        actual = model if isinstance(model, onnx.ModelProto) else onnx.load(model)
        np.testing.assert_array_equal(
            numpy_helper.to_array(actual.graph.initializer[0]), numpy_helper.to_array(expected)
        )
    assert _snapshot(tmp_path) == before
