import gc
import inspect
import traceback
import weakref
from contextlib import contextmanager

import numpy as np
import onnx
import pytest
from onnxruntime.transformers import optimizer

import optimize_graphs
from tests.artifact_fixtures import _ort_result, _write_model


@pytest.fixture(autouse=True)
def close_failed_ort_temporary_directories(monkeypatch):
    original = optimizer.tempfile.TemporaryDirectory
    directories = []

    def tracked(*args, **kwargs):
        directory = original(*args, **kwargs)
        directories.append(directory)
        return directory

    monkeypatch.setattr(optimizer.tempfile, "TemporaryDirectory", tracked)
    try:
        yield
    finally:
        for directory in directories:
            directory.cleanup()


@contextmanager
def manual_gc():
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if enabled:
            gc.enable()
        else:
            gc.disable()


def tiny_graph(tmp_path, fusion=False):
    path = tmp_path / "tiny.onnx"
    _write_model(tmp_path, path.name, weight=np.eye(16, dtype=np.float32))
    model = onnx.load(path)
    if fusion:
        model.graph.node[0].output[0] = "mm"
        model.graph.node.extend(
            [
                onnx.helper.make_node("Add", ["mm", "x"], ["sum"]),
                onnx.helper.make_node("LayerNormalization", ["sum", "scale", "bias"], ["y"], axis=-1),
            ]
        )
        model.graph.initializer.extend(
            [
                onnx.numpy_helper.from_array(np.ones(16, np.float32), "scale"),
                onnx.numpy_helper.from_array(np.zeros(16, np.float32), "bias"),
            ]
        )
        onnx.save(model, path)
    assert sum(len(t.raw_data) for t in model.graph.initializer) <= 1024 * 1024
    return path


def test_obsolete_helper_dies_before_next_constructor(tmp_path, monkeypatch):
    from shape_infer_helper import SymbolicShapeInferenceHelper

    path = tiny_graph(tmp_path)
    original = SymbolicShapeInferenceHelper.__init__
    references = []

    def observed(self, *args, **kwargs):
        assert all(reference() is None for reference in references)
        original(self, *args, **kwargs)
        references.append(weakref.ref(self))

    monkeypatch.setattr(SymbolicShapeInferenceHelper, "__init__", observed)
    with manual_gc():
        optimize_graphs.optimize_graph(str(path), 2, 16)
        assert len(references) >= 2


@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("active_error", [False, True])
def test_graph_owner_collects_real_wrapper_on_return(tmp_path, monkeypatch, fusion, active_error):
    path = tiny_graph(tmp_path, fusion=fusion)
    original = optimize_graphs.optimize_model
    references = []

    def observed(*args, **kwargs):
        model = original(*args, **kwargs)
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr(optimize_graphs, "optimize_model", observed)
    with manual_gc():
        if active_error:
            try:
                raise ValueError("caller context")
            except ValueError as caller_error:
                optimize_graphs.optimize_graph(str(path), 2, 16)
                assert str(caller_error) == "caller context"
                assert caller_error.__traceback__ is not None
        else:
            optimize_graphs.optimize_graph(str(path), 2, 16)
        assert len(references) == 1
        assert references[0]() is None


def _assert_graph_diagnostics(error, reference):
    assert reference() is not None
    owner_locals = next(
        frame.f_locals
        for frame, _ in traceback.walk_tb(error.__traceback__)
        if frame.f_code is optimize_graphs.optimize_graph.__code__
    )
    assert owner_locals["model"] is reference()
    assert len(reference().model.graph.node) > 0


@pytest.mark.parametrize("failure", ["before-return", "after-return", "chained", "checker", "abort"])
def test_graph_errors_keep_wrapper_until_caller_releases_traceback(tmp_path, monkeypatch, failure):
    path = tiny_graph(tmp_path, fusion=failure == "checker")
    references = []

    class OperationAborted(BaseException):
        pass

    error_type = OperationAborted if failure == "abort" else RuntimeError
    injected = error_type("injected optimizer failure")

    def inner(model):
        raise ValueError("inner optimizer failure")

    def failed(model, *args):
        references.append(weakref.ref(model))
        assert model.infer_runtime_shape(update=True).get_edge_shape("y") == [1, 16]
        if failure == "chained":
            try:
                inner(model)
            except ValueError as error:
                raise injected from error
        raise injected

    if failure == "before-return":
        monkeypatch.setattr(optimizer.MODEL_TYPES["gpt2"][0], "optimize", failed)
    elif failure == "checker":
        original = optimize_graphs.optimize_model

        def observed(*args, **kwargs):
            model = original(*args, **kwargs)
            references.append(weakref.ref(model))
            return model

        def failed_checker(*args):
            raise injected

        monkeypatch.setattr(optimize_graphs, "optimize_model", observed)
        monkeypatch.setattr(optimize_graphs.onnx.checker, "check_model", failed_checker)
    else:
        monkeypatch.setattr(optimize_graphs, "_portable_rms_norms", failed)
    with manual_gc():
        with pytest.raises(error_type, match="injected optimizer failure") as caught:
            optimize_graphs.optimize_graph(str(path), 2, 16)
        assert caught.value is injected
        assert caught.value.__traceback__ is not None
        if failure == "chained":
            assert isinstance(caught.value.__cause__, ValueError)
            assert str(caught.value.__cause__) == "inner optimizer failure"
        assert len(references) == 1
        assert references[0]() is not None
        if failure != "before-return":
            _assert_graph_diagnostics(caught.value, references[0])
        del caught
        injected = None
        gc.collect()
        assert references[0]() is None


def test_graph_error_retry_collects_only_after_exception_release(tmp_path, monkeypatch):
    paths = []
    for name in ("a", "b", "c"):
        directory = tmp_path / name
        directory.mkdir()
        paths.append(tiny_graph(directory, fusion=name == "a"))
    original = optimize_graphs.optimize_model
    checker = optimize_graphs.onnx.checker.check_model
    references = []
    injected = RuntimeError("checker failure")

    def observed(*args, **kwargs):
        model = original(*args, **kwargs)
        references.append(weakref.ref(model))
        return model

    def failed_checker(*args):
        raise injected

    monkeypatch.setattr(optimize_graphs, "optimize_model", observed)
    monkeypatch.setattr(optimize_graphs.onnx.checker, "check_model", failed_checker)
    with manual_gc():
        with pytest.raises(RuntimeError, match="checker failure") as caught:
            optimize_graphs.optimize_graph(str(paths[0]), 2, 16)
        _assert_graph_diagnostics(caught.value, references[0])
        monkeypatch.setattr(optimize_graphs.onnx.checker, "check_model", checker)
        optimize_graphs.optimize_graph(str(paths[1]), 2, 16)
        _assert_graph_diagnostics(caught.value, references[0])
        assert references[1]() is None
        del caught
        injected = None
        optimize_graphs.optimize_graph(str(paths[2]), 2, 16)
        assert len(references) == 3
        assert all(reference() is None for reference in references)


def test_resource_factory_preserves_real_gpt2_graph_and_options(tmp_path):
    from src.optimizer_resources import ResourceGpt2OnnxModel, resource_model_type

    path = tiny_graph(tmp_path)
    original = optimizer.MODEL_TYPES["gpt2"]
    bert = optimizer.MODEL_TYPES["bert"]
    key = resource_model_type()
    assert resource_model_type() == key
    assert key not in {"bert", "gpt2"}
    assert optimizer.MODEL_TYPES["gpt2"] == original
    assert optimizer.MODEL_TYPES["bert"] == bert
    assert ResourceGpt2OnnxModel.__bases__ == (original[0],)
    assert ResourceGpt2OnnxModel.fuse_attention is original[0].fuse_attention
    assert ResourceGpt2OnnxModel.postprocess is original[0].postprocess
    assert inspect.signature(ResourceGpt2OnnxModel.infer_runtime_shape) == inspect.signature(
        original[0].infer_runtime_shape
    )
    reference = optimizer.optimize_model(str(path), model_type="gpt2", num_heads=2, hidden_size=16, opt_level=1)
    candidate = optimizer.optimize_model(
        str(path),
        model_type=key,
        num_heads=2,
        hidden_size=16,
        opt_level=1,
        optimization_options=optimizer.FusionOptions("gpt2"),
    )
    assert type(reference) is original[0]
    assert type(candidate) is ResourceGpt2OnnxModel
    assert candidate.get_fused_operator_statistics() == reference.get_fused_operator_statistics()
    assert candidate.model.SerializeToString() == reference.model.SerializeToString()
    for name, model in (("reference.onnx", reference.model), ("candidate.onnx", candidate.model)):
        onnx.save(model, tmp_path / name)
    np.testing.assert_array_equal(_ort_result(tmp_path / "candidate.onnx"), _ort_result(tmp_path / "reference.onnx"))


def test_current_cache_mapping_and_disabled_inference_are_preserved(tmp_path):
    from src.optimizer_resources import ResourceGpt2OnnxModel

    model = ResourceGpt2OnnxModel(onnx.load(tiny_graph(tmp_path)), 2, 16)
    mapping = {"batch": 1}
    with manual_gc():
        helper = model.infer_runtime_shape(mapping, update=True)
        assert helper.get_edge_shape("y") == [1, 16]
        assert helper.dynamic_axis_mapping_ is mapping
        assert model.infer_runtime_shape(mapping, update=False) is helper
        model.disable_shape_inference()
        assert model.infer_runtime_shape(mapping, update=True) is None
        assert model.shape_infer_helper is helper


def test_invalid_none_mapping_retains_original_failure_behavior(tmp_path):
    from src.optimizer_resources import ResourceGpt2OnnxModel

    proto = onnx.load(tiny_graph(tmp_path))
    for cls in (optimizer.MODEL_TYPES["gpt2"][0], ResourceGpt2OnnxModel):
        model = cls(proto, 2, 16)
        assert model.infer_runtime_shape(None, update=True) is None
        assert model.enable_shape_infer is False


def test_graph_error_keeps_unrelated_caller_context_locals(tmp_path, monkeypatch):
    path = tiny_graph(tmp_path)
    references = []

    class CallerState:
        pass

    def caller_failure():
        state = CallerState()
        references.append(weakref.ref(state))
        raise ValueError("caller context")

    def graph_failure(model):
        raise RuntimeError("graph failure")

    monkeypatch.setattr(optimize_graphs, "_portable_rms_norms", graph_failure)
    try:
        caller_failure()
    except ValueError as caller_error:
        with pytest.raises(RuntimeError, match="graph failure") as caught:
            optimize_graphs.optimize_graph(str(path), 2, 16)
        assert caught.value.__context__ is caller_error
        assert references[0]() is not None


@pytest.mark.parametrize("problem", ["version", "tuple", "signature", "collision"])
def test_unsupported_optimizer_refuses_instead_of_using_unfused_fallback(monkeypatch, problem):
    import src.optimizer_resources as resources

    foreign = (object, "foreign", 0)
    if problem == "version":
        monkeypatch.setattr(resources.onnxruntime, "__version__", "unsupported")
    elif problem == "tuple":
        monkeypatch.setitem(optimizer.MODEL_TYPES, "gpt2", foreign)
    elif problem == "signature":
        monkeypatch.setattr(resources._BASE, "infer_runtime_shape", lambda self: None)
    else:
        monkeypatch.setitem(optimizer.MODEL_TYPES, resources.MODEL_TYPE, foreign)
    with pytest.raises(RuntimeError):
        resources.resource_model_type()
    if problem == "collision":
        assert optimizer.MODEL_TYPES[resources.MODEL_TYPE] == foreign
