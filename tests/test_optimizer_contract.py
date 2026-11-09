import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnxconverter_common.float16 import convert_float_to_float16 as common_float16
from onnxruntime.transformers.float16 import convert_float_to_float16 as ort_float16
from onnxruntime.transformers.onnx_model import OnnxModel

from optimize_graphs import RMS_DOMAIN, _portable_rms_norms, _rms_function, optimize_graph
from quantize_nbits import quantize_decoder
from share_weights import get_external_info, share_external_models


def _model(nodes, initializers, opset=18, width=8):
    return helper.make_model(
        helper.make_graph(
            nodes,
            "normalization",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, width])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, width])],
            [numpy_helper.from_array(value, name) for name, value in initializers.items()],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )


def _run(path, values, residual=None):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    feeds = {"x": values}
    if residual is not None:
        feeds["residual"] = residual
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(None, feeds)[0]


def _rms_model(opset=18, width=8, epsilon=1e-6, projection=False):
    source = "projected" if projection else "x"
    nodes = [
        helper.make_node("Pow", [source, "power"], ["square"]),
        helper.make_node(
            "ReduceMean",
            ["square", "axes"] if opset >= 18 else ["square"],
            ["mean"],
            keepdims=1,
            **({} if opset >= 18 else {"axes": [-1]}),
        ),
        helper.make_node("Add", ["mean", "epsilon"], ["variance"]),
        helper.make_node("Sqrt", ["variance"], ["root"]),
        helper.make_node("Div", [source, "root"], ["normalized"]),
        helper.make_node("Mul", ["normalized", "scale"], ["y"]),
    ]
    initializers = {
        "power": np.array(2, dtype=np.float32),
        "epsilon": np.array(epsilon, dtype=np.float32),
        "scale": np.linspace(0.5, 1.5, width, dtype=np.float32),
    }
    if opset >= 18:
        initializers["axes"] = np.array([-1], dtype=np.int64)
    if projection:
        nodes.insert(0, helper.make_node("MatMul", ["x", "weight"], [source], name="projection"))
        initializers["weight"] = np.eye(width, dtype=np.float32)
    return _model(nodes, initializers, opset, width)


def _portable_contract(model):
    functions = [f for f in model.functions if f.domain == RMS_DOMAIN and f.name == "RMSNormalization"]
    assert len(functions) == 1
    assert [(item.domain, item.version) for item in model.opset_import if item.domain == RMS_DOMAIN] == [
        (RMS_DOMAIN, 1)
    ]
    assert any(node.domain == RMS_DOMAIN and node.op_type == "RMSNormalization" for node in model.graph.node)
    assert not any(node.domain == "" and node.op_type == "SimplifiedLayerNormalization" for node in model.graph.node)
    return functions[0].SerializeToString()


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize("external", [False, True])
def test_optimized_rms_norm_remains_standard_onnx(tmp_path, external, opset):
    width = 512
    model = _rms_model(opset, width=width)
    path = tmp_path / "decoder.onnx"
    onnx.save(model, path)
    values = np.random.default_rng(4342).standard_normal((1, 2, width), dtype=np.float32)
    expected = _run(path, values)
    optimize_graph(str(path), num_heads=2, hidden_size=width, use_external_data=external)
    onnx.checker.check_model(str(path), full_check=True)
    saved = onnx.load(path, load_external_data=False)
    assert [(item.domain, item.version) for item in saved.opset_import if item.domain == ""] == [("", opset)]
    assert [(item.domain, item.version) for item in saved.functions[0].opset_import] == [("", opset)]
    assert any(tensor.data_location == TensorProto.EXTERNAL for tensor in saved.graph.initializer) == external
    function = _portable_contract(onnx.load(path))
    np.testing.assert_allclose(_run(path, values), expected, atol=1e-6, rtol=1e-6)
    optimize_graph(str(path), num_heads=2, hidden_size=width, use_external_data=external)
    onnx.checker.check_model(str(path), full_check=True)
    assert _portable_contract(onnx.load(path)) == function
    np.testing.assert_allclose(_run(path, values), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "convert,keep_io_types", [(ort_float16, False), (common_float16, True)], ids=["ort", "onnxconverter-common"]
)
def test_float16_converters_preserve_fp32_rms_statistics(tmp_path, convert, keep_io_types):
    epsilon = 1e-7
    path = tmp_path / "decoder.onnx"
    onnx.save(_rms_model(epsilon=epsilon), path)
    optimize_graph(str(path), num_heads=2, hidden_size=8, use_external_data=False)
    model = onnx.load(path)
    function = _portable_contract(model)
    converted = convert(model, keep_io_types=keep_io_types)
    assert _portable_contract(converted) == function
    io_type = TensorProto.FLOAT if keep_io_types else TensorProto.FLOAT16
    assert converted.graph.input[0].type.tensor_type.elem_type == io_type
    assert converted.graph.output[0].type.tensor_type.elem_type == io_type
    onnx.save(converted, path)
    onnx.checker.check_model(str(path), full_check=True)
    values = np.array(
        [[[1000, -1000, 700, -700, 500, -500, 900, -900], [0, 1e-5, -1e-5, 3e-5, -3e-5, 1e-4, -1e-4, 2e-4]]],
        dtype=np.float16,
    )
    stats = values.astype(np.float32)
    normalized = (stats / np.sqrt(np.mean(stats * stats, axis=-1, keepdims=True) + epsilon)).astype(np.float16)
    expected = normalized * np.linspace(0.5, 1.5, 8, dtype=np.float32).astype(np.float16)
    io_dtype = np.float32 if keep_io_types else np.float16
    actual = _run(path, values.astype(io_dtype))
    assert actual.dtype == io_dtype
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual[0, 0])) > 1
    assert np.max(np.abs(actual[0, 1])) > 0.5
    np.testing.assert_allclose(actual, expected, atol=np.finfo(np.float16).eps, rtol=np.finfo(np.float16).eps)


def test_quantization_and_weight_sharing_preserve_portable_rms(tmp_path):
    source = tmp_path / "decoder.onnx"
    onnx.save(_rms_model(width=64, projection=True), source)
    optimize_graph(str(source), num_heads=2, hidden_size=64, use_external_data=False)
    function = _portable_contract(onnx.load(source))
    values = np.random.default_rng(4342).standard_normal((1, 2, 64), dtype=np.float32)
    paths = [tmp_path / f"decoder_{name}.int4.onnx" for name in ("init", "step")]
    expected = []
    for path in paths:
        quantize_decoder(str(source), str(path), 4, 32, 4, "rtn", None)
        model = onnx.load(path)
        assert any(node.domain == "com.microsoft" and node.op_type == "MatMulNBits" for node in model.graph.node)
        assert _portable_contract(model) == function
        expected.append(_run(path, values))
    assert share_external_models(tmp_path, "int4")
    assert (tmp_path / "decoder_weights.int4.data").stat().st_size >= 1024
    for path, before in zip(paths, expected):
        onnx.checker.check_model(str(path), full_check=True)
        model = onnx.load(path, load_external_data=False)
        assert _portable_contract(model) == function
        assert any(
            get_external_info(tensor).get("location") == "decoder_weights.int4.data"
            for tensor in model.graph.initializer
        )
        np.testing.assert_array_equal(_run(path, values), before)


def _raw_rms_model():
    return _model(
        [helper.make_node("SimplifiedLayerNormalization", ["x", "scale"], ["y"])],
        {"scale": np.ones(8, dtype=np.float32)},
    )


@pytest.mark.parametrize("axis", [None, 2], ids=["defaults", "positive-last-axis"])
def test_rms_defaults_and_proven_last_axis_are_supported(tmp_path, axis):
    model = _raw_rms_model()
    if axis is not None:
        model.graph.node[0].attribute.append(helper.make_attribute("axis", axis))
    assert _portable_rms_norms(OnnxModel(model)) == 1
    _portable_contract(model)
    node = model.graph.node[0]
    assert len(node.attribute) == 1
    assert node.attribute[0].name == "epsilon"
    assert node.attribute[0].f == pytest.approx(1e-5)
    path = tmp_path / "decoder.onnx"
    onnx.save(model, path)
    onnx.checker.check_model(str(path), full_check=True)
    values = np.random.default_rng(4342).standard_normal((1, 2, 8), dtype=np.float32)
    expected = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(_run(path, values), expected, atol=1e-6, rtol=1e-6)


def _assert_rejected_without_mutation(model):
    model.graph.node.insert(0, helper.make_node("SimplifiedLayerNormalization", ["x", "scale"], ["first"]))
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        _portable_rms_norms(OnnxModel(model))
    assert model.SerializeToString() == before


@pytest.mark.parametrize(
    "attributes",
    [
        pytest.param([helper.make_attribute("axis", 0)], id="non-last-axis"),
        pytest.param([helper.make_attribute("stash_type", TensorProto.FLOAT16)], id="fp16-statistics"),
        pytest.param([helper.make_attribute("unexpected", 1)], id="unknown-attribute"),
        pytest.param([helper.make_attribute("axis", -1.0)], id="attribute-type"),
        pytest.param([helper.make_attribute("axis", -1)] * 2, id="duplicate-attribute"),
        pytest.param(
            [onnx.AttributeProto(name="axis", type=onnx.AttributeProto.INT, ref_attr_name="axis")],
            id="referenced-attribute",
        ),
        pytest.param([helper.make_attribute("epsilon", float("nan"))], id="nonfinite-epsilon"),
    ],
)
def test_unsupported_rms_attributes_fail_before_mutation(attributes):
    model = _raw_rms_model()
    model.graph.node[0].attribute.extend(attributes)
    _assert_rejected_without_mutation(model)


@pytest.mark.parametrize(
    "inputs,outputs,overload",
    [
        pytest.param(["x", "scale"], ["y", "inverse_rms"], "", id="auxiliary-output"),
        pytest.param(["x", ""], ["y"], "", id="missing-input"),
        pytest.param(["x", "scale"], [""], "", id="empty-output"),
        pytest.param(["x", "scale"], ["y"], "unsupported", id="overload"),
    ],
)
def test_unsupported_rms_signature_fails_before_mutation(inputs, outputs, overload):
    model = _raw_rms_model()
    model.graph.node[0].CopyFrom(helper.make_node("SimplifiedLayerNormalization", inputs, outputs, overload=overload))
    _assert_rejected_without_mutation(model)


def test_positive_axis_requires_known_rank():
    model = _raw_rms_model()
    model.graph.input[0].type.tensor_type.ClearField("shape")
    model.graph.node[0].attribute.append(helper.make_attribute("axis", 2))
    _assert_rejected_without_mutation(model)


def test_rms_rejects_unsupported_opset():
    model = _raw_rms_model()
    model.opset_import[0].version = 16
    _assert_rejected_without_mutation(model)


@pytest.mark.parametrize("duplicate", [False, True], ids=["conflicting-body", "duplicate-definition"])
def test_function_collisions_fail_before_mutation(duplicate):
    model = _raw_rms_model()
    function = _rms_function(18)
    if duplicate:
        model.functions.extend([function, function])
    else:
        function.node[-1].op_type = "Add"
        model.functions.append(function)
    _assert_rejected_without_mutation(model)


@pytest.mark.parametrize("versions", [[2], [1, 1]], ids=["conflicting-version", "duplicate-import"])
def test_opset_collisions_fail_before_mutation(versions):
    model = _raw_rms_model()
    model.opset_import.extend(helper.make_opsetid(RMS_DOMAIN, version) for version in versions)
    _assert_rejected_without_mutation(model)


def test_matching_existing_function_and_import_are_reused():
    model = _raw_rms_model()
    model.functions.append(_rms_function(18))
    model.opset_import.append(helper.make_opsetid(RMS_DOMAIN, 1))
    function = model.functions[0].SerializeToString()
    assert _portable_rms_norms(OnnxModel(model)) == 1
    assert _portable_contract(model) == function
    before = model.SerializeToString()
    assert _portable_rms_norms(OnnxModel(model)) == 0
    assert model.SerializeToString() == before


def test_unrelated_custom_normalization_domain_is_untouched():
    model = _raw_rms_model()
    model.graph.node[0].domain = "unrelated"
    model.opset_import.append(helper.make_opsetid("unrelated", 7))
    model.functions.append(
        helper.make_function(
            "unrelated",
            "SimplifiedLayerNormalization",
            ["x", "scale"],
            ["y"],
            [helper.make_node("Mul", ["x", "scale"], ["y"])],
            [helper.make_opsetid("", 18)],
        )
    )
    before = model.SerializeToString()
    assert _portable_rms_norms(OnnxModel(model)) == 0
    assert model.SerializeToString() == before


def test_standard_layer_norm_can_still_fuse_with_its_residual(tmp_path):
    nodes = [
        helper.make_node("Add", ["x", "residual"], ["sum"]),
        helper.make_node("LayerNormalization", ["sum", "scale", "bias"], ["y"], axis=-1, epsilon=1e-5),
    ]
    model = _model(
        nodes,
        {
            "scale": np.ones(8, dtype=np.float32),
            "bias": np.zeros(8, dtype=np.float32),
        },
    )
    model.graph.input.append(helper.make_tensor_value_info("residual", TensorProto.FLOAT, [1, 2, 8]))
    path = tmp_path / "encoder.onnx"
    onnx.save(model, path)
    values = np.random.default_rng(4342).standard_normal((1, 2, 8), dtype=np.float32)
    residual = np.ones_like(values)
    expected = _run(path, values, residual)
    optimize_graph(str(path), num_heads=2, hidden_size=8, use_external_data=False)
    onnx.checker.check_model(str(path))
    assert any(node.op_type == "SkipLayerNormalization" for node in onnx.load(path).graph.node)
    np.testing.assert_allclose(_run(path, values, residual), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize("input_dtype,scale_dtype", [(np.float32, np.float16), (np.float16, np.float32)])
def test_portable_rms_preserves_independent_scale_output_dtype(tmp_path, opset, input_dtype, scale_dtype):
    scale = np.linspace(0.5, 1.5, 8, dtype=scale_dtype)
    model = _model(
        [
            helper.make_node(
                "SimplifiedLayerNormalization",
                ["x", "scale"],
                ["y"],
                axis=-1,
                epsilon=1e-7,
                stash_type=TensorProto.FLOAT,
            )
        ],
        {"scale": scale},
        opset=opset,
    )
    model.graph.input[0].type.tensor_type.elem_type = helper.np_dtype_to_tensor_dtype(np.dtype(input_dtype))
    model.graph.output[0].type.tensor_type.elem_type = helper.np_dtype_to_tensor_dtype(np.dtype(scale_dtype))
    path = tmp_path / "mixed-rms.onnx"
    onnx.save(model, path)
    values = np.array(
        [[[1000, -1000, 700, -700, 500, -500, 900, -900], [0, 1e-5, -1e-5, 3e-5, -3e-5, 1e-4, -1e-4, 2e-4]]],
        dtype=input_dtype,
    )
    expected = _run(path, values)
    assert expected.dtype == scale_dtype
    assert _portable_rms_norms(OnnxModel(model)) == 1
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    actual = _run(path, values)
    assert actual.dtype == scale_dtype
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=np.finfo(scale_dtype).eps, rtol=np.finfo(scale_dtype).eps)
