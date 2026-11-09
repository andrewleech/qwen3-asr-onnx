import builtins
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

import share_weights as share
from tests.artifact_fixtures import _ort_result, _write_external_identity, _write_model
from tests.test_native_decoder import (
    assert_decoder_matches,
    prefill_inputs,
)
from tests.test_native_decoder import decoder_exports as decoder_exports
from tests.test_native_decoder import decoder_threads as decoder_threads
from tests.test_native_decoder import text_model as text_model


def _pair(root, inline=False):
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        _write_model(root, name, external=not inline)
    return {name: _ort_result(root / name) for name in ("decoder_init.onnx", "decoder_step.onnx")}


def _assert_pair(root, expected):
    for name, values in expected.items():
        np.testing.assert_array_equal(_ort_result(root / name), values)


def _assert_shared(root):
    assert {"decoder_init.onnx", "decoder_step.onnx", "decoder_weights.data"} <= {path.name for path in root.iterdir()}
    assert all(path.is_file() for path in root.iterdir())


def test_interruption_after_data_publication_is_recoverable(tmp_path, monkeypatch):
    expected = _pair(tmp_path)
    replace, rename = share.os.replace, share.os.rename

    def interrupt(operation):
        def run(source, target, *args, **kwargs):
            result = operation(source, target, *args, **kwargs)
            if Path(target).name == "decoder_weights.data":
                raise OSError("injected interruption after publishing data")
            return result

        return run

    with monkeypatch.context() as patch:
        patch.setattr(share.os, "replace", interrupt(replace))
        patch.setattr(share.os, "rename", interrupt(rename))
        with pytest.raises(OSError, match="injected"):
            share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    assert share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    _assert_shared(tmp_path)


def test_sharing_uses_graph_alias_directory_for_weights(tmp_path):
    physical, source = tmp_path / "physical", tmp_path / "source"
    physical.mkdir()
    source.mkdir()
    expected = {}
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        data_name = name + ".data"
        _write_model(physical, name, external=True, data_name=data_name, weight=np.full((32, 32), 99, np.float32))
        (source / name).symlink_to(physical / name)
        (source / data_name).write_bytes(np.full((32, 32), 42, np.float32).tobytes())
        expected[name] = _ort_result(source / name)
    before = {p.name: p.read_bytes() for p in physical.iterdir()}
    assert share.share_external_models(source)
    _assert_pair(source, expected)
    assert {p.name: p.read_bytes() for p in physical.iterdir()} == before


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("graph_name", ["decoder_init.onnx", "decoder_step.onnx"])
def test_partial_proto_write_keeps_original_pair_usable(tmp_path, monkeypatch, inline, graph_name):
    expected = _pair(tmp_path, inline=inline)
    original_open = builtins.open

    class PartialWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, key):
            return getattr(self.stream, key)

        def write(self, data):
            self.stream.write(data[: len(data) // 2])
            raise OSError("injected partial proto write")

    def injected(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if Path(path).name == graph_name and mode == "wb":
            return PartialWrite(stream)
        return stream

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", injected)
        with pytest.raises(OSError, match="partial proto"):
            share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    assert share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    _assert_shared(tmp_path)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("graph_index", [0, 1])
def test_interruption_after_graph_replace_can_retry(tmp_path, monkeypatch, inline, graph_index):
    expected = _pair(tmp_path, inline=inline)
    original_replace = share.os.replace
    graph_name = ("decoder_init.onnx", "decoder_step.onnx")[graph_index]

    def injected(source, target):
        original_replace(source, target)
        if Path(target).name == graph_name:
            raise OSError("injected interruption after graph replace")

    with monkeypatch.context() as patch:
        patch.setattr(share.os, "replace", injected)
        with pytest.raises(OSError, match="after graph"):
            share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    assert share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    _assert_shared(tmp_path)

    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        model = onnx.load(tmp_path / name, load_external_data=False)
        weight = next(tensor for tensor in model.graph.initializer if tensor.name == "weight")
        assert weight.data_location == TensorProto.EXTERNAL
        assert share.get_external_info(weight)["location"] == "decoder_weights.data"


def test_sharing_retains_original_data_without_unlinking(tmp_path, monkeypatch):
    expected = _pair(tmp_path)
    retained = {p.name: p.read_bytes() for p in tmp_path.glob("*.onnx.data")}
    original_unlink = share.os.unlink

    def injected(path, *args, **kwargs):
        if str(path).endswith("decoder_step.onnx.data"):
            raise OSError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(share.os, "unlink", injected)
        assert share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    assert share.share_external_models(tmp_path)
    _assert_shared(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.onnx.data")} == retained


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_sharing_preserves_unmatched_tensors_and_storage_contract(tmp_path, dtype):
    init_weight = np.arange(512, dtype=dtype).reshape(32, 16)
    step_weight = init_weight + 2
    _write_model(tmp_path, "decoder_init.onnx", external=True, weight=init_weight)
    _write_model(tmp_path, "decoder_step.onnx", external=True, weight=step_weight)
    expected = {name: _ort_result(tmp_path / name) for name in ("decoder_init.onnx", "decoder_step.onnx")}
    assert share.share_external_models(tmp_path)
    for name, weight in (("decoder_init.onnx", init_weight), ("decoder_step.onnx", step_weight)):
        tensor = onnx.load(tmp_path / name).graph.initializer[0]
        actual = numpy_helper.to_array(tensor)
        assert actual.dtype == weight.dtype
        assert actual.shape == weight.shape
        assert actual.tobytes() == weight.tobytes()
    _assert_pair(tmp_path, expected)
    _assert_shared(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert share.share_external_models(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_shared_existence_does_not_hide_invalid_references(tmp_path):
    _pair(tmp_path)
    (tmp_path / "decoder_weights.data").write_bytes(b"unrelated")
    (tmp_path / "decoder_step.onnx.data").unlink()
    with pytest.raises((ValueError, onnx.checker.ValidationError, FileNotFoundError)):
        share.share_external_models(tmp_path)


def test_live_incompatible_canonical_data_is_not_overwritten(tmp_path):
    _pair(tmp_path)
    assert share.share_external_models(tmp_path)
    canonical = tmp_path / "decoder_weights.data"
    original = canonical.read_bytes()
    _write_model(
        tmp_path,
        "decoder_step.onnx",
        external=True,
        data_name="replacement.bin",
        weight=np.ones((32, 16), dtype=np.float32),
    )
    expected = {name: _ort_result(tmp_path / name) for name in ("decoder_init.onnx", "decoder_step.onnx")}
    with pytest.raises(ValueError, match="rebuild"):
        share.share_external_models(tmp_path)
    assert canonical.read_bytes() == original
    _assert_pair(tmp_path, expected)


@pytest.mark.parametrize("data_name", ["embed_tokens.bin", "weights.bin", "sub/weights.data"])
def test_sharing_retains_independently_owned_dependencies(tmp_path, data_name):
    (tmp_path / data_name).parent.mkdir(parents=True, exist_ok=True)
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        _write_model(tmp_path, name, external=True, data_name=data_name)
    expected = {name: _ort_result(tmp_path / name) for name in share._decoder_filenames()}
    data = tmp_path / data_name
    original = data.read_bytes()
    auxiliary = None
    if data_name.startswith("sub/"):
        auxiliary = data.parent / "aux.onnx"
        proto = onnx.load(tmp_path / "decoder_init.onnx", load_external_data=False)
        for entry in proto.graph.initializer[0].external_data:
            if entry.key == "location":
                entry.value = data.name
        onnx.save(proto, auxiliary)
        auxiliary_before = _ort_result(auxiliary)
    if data_name == "embed_tokens.bin":
        import json

        from src.embeddings import load_embed_tokens

        config = {"decoder": {"vocab_size": 64, "hidden_size": 16}}
        (tmp_path / "config.json").write_text(json.dumps(config))
        embeddings_before = load_embed_tokens(tmp_path)
    assert share.share_external_models(tmp_path)
    assert data.read_bytes() == original
    _assert_pair(tmp_path, expected)
    if auxiliary is not None:
        np.testing.assert_array_equal(_ort_result(auxiliary), auxiliary_before)
    if data_name == "embed_tokens.bin":
        np.testing.assert_array_equal(load_embed_tokens(tmp_path), embeddings_before)


def _run_model(path, feed):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(None, feed)


def test_shared_native_decoder_preserves_prefill_and_step(tmp_path, decoder_exports, text_model):
    for path in decoder_exports["init"].parent.iterdir():
        shutil.copy2(path, tmp_path / path.name)
    with torch.no_grad():
        _, prefill_feed = prefill_inputs(text_model, 1, 7, 1, 2)
        before = _run_model(tmp_path / "decoder_init.onnx", prefill_feed)
        token = torch.tensor([[3]], dtype=torch.int64)
        step_feed = {
            "input_embeds": text_model.thinker.model.embed_tokens(token).numpy(),
            "position_ids": np.array([[7]], dtype=np.int64),
            "past_keys": before[1],
            "past_values": before[2],
        }
        before_step = _run_model(tmp_path / "decoder_step.onnx", step_feed)
    assert share.share_external_models(tmp_path)
    assert_decoder_matches(_run_model(tmp_path / "decoder_init.onnx", prefill_feed), before)
    assert_decoder_matches(_run_model(tmp_path / "decoder_step.onnx", step_feed), before_step)
    _assert_shared(tmp_path)


@pytest.mark.parametrize("kind", ["Constant", "ConstantOfShape", "Split"])
def test_shape_and_attribute_constants_stay_executable(tmp_path, kind):
    shape = numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name="shape")
    if kind == "Constant":
        nodes = [
            helper.make_node("Constant", [], ["shape"], value=shape),
            helper.make_node("Reshape", ["x", "shape"], ["y"]),
        ]
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])]
        initializers = []
        output_shape = [2, 2]
        feed = {"x": np.arange(4, dtype=np.float32)}
    elif kind == "ConstantOfShape":
        nodes = [
            helper.make_node(
                "ConstantOfShape", ["shape"], ["y"], value=numpy_helper.from_array(np.array([1.0], dtype=np.float32))
            )
        ]
        inputs = []
        initializers = [shape]
        output_shape = [2, 2]
        feed = {}
    else:
        initializers = [numpy_helper.from_array(np.ones(129, dtype=np.int64), name="splits")]
        outputs = ["y", *[f"unused{i}" for i in range(128)]]
        nodes = [helper.make_node("Split", ["x", "splits"], outputs, axis=0)]
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [129])]
        output_shape = [1]
        feed = {"x": np.arange(129, dtype=np.float32)}
    graph = helper.make_graph(
        nodes, kind, inputs, [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)], initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    for name in share._decoder_filenames():
        onnx.save(model, tmp_path / name)
    before = _run_model(tmp_path / "decoder_init.onnx", feed)
    assert share.share_external_models(tmp_path)
    for name in share._decoder_filenames():
        np.testing.assert_array_equal(_run_model(tmp_path / name, feed)[0], before[0])
        proto = onnx.load(tmp_path / name, load_external_data=False)
        assert all(t.data_location != TensorProto.EXTERNAL for t in share.model_tensors(proto))


@pytest.mark.parametrize("inline", [False, True])
def test_staged_runtime_failure_preserves_source_pair(tmp_path, monkeypatch, inline):
    expected = _pair(tmp_path, inline=inline)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original = ort.InferenceSession

    def fail_staged(path, *args, **kwargs):
        if Path(path).parent.name.startswith(".share-"):
            raise RuntimeError("injected staged runtime failure")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ort, "InferenceSession", fail_staged)
        with pytest.raises(RuntimeError, match="staged runtime"):
            share.share_external_models(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    _assert_pair(tmp_path, expected)


@pytest.mark.parametrize("dtype,suffix", [(np.float32, "q4"), (np.float16, "q4f16")])
def test_shared_quantized_variants_pass_real_cpu_sessions(tmp_path, dtype, suffix):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    tensor_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    weight = numpy_helper.from_array(np.arange(64 * 32, dtype=dtype).reshape(64, 32) / 2048, name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"], name="matmul")],
        "quantized",
        [helper.make_tensor_value_info("x", tensor_type, [1, 64])],
        [helper.make_tensor_value_info("y", tensor_type, [1, 32])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=32, is_symmetric=False, accuracy_level=4)
    quantizer.process()
    for name in share._decoder_filenames(suffix):
        onnx.save(quantizer.model.model, tmp_path / name)
    feed = {"x": np.ones((1, 64), dtype=dtype)}
    before = _run_model(tmp_path / share._decoder_filenames(suffix)[0], feed)
    assert share.share_external_models(tmp_path, suffix=suffix)
    for name in share._decoder_filenames(suffix):
        np.testing.assert_array_equal(_run_model(tmp_path / name, feed)[0], before[0])


@pytest.mark.parametrize("inline", [False, True])
def test_conflicting_canonical_data_is_preserved(tmp_path, inline):
    expected = _pair(tmp_path, inline=inline)
    path = tmp_path / "decoder_weights.data"
    path.write_bytes(b"independently retained canonical bytes")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="incompatible"):
        share.share_external_models(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    _assert_pair(tmp_path, expected)


@pytest.mark.parametrize("kind,suffix", [("symlink", None), ("hardlink", None), ("hardlink", "q4f16")])
def test_linked_canonical_data_rejected_before_graph_publication(tmp_path, kind, suffix):
    reference = tmp_path / "reference"
    reference.mkdir()
    names = share._decoder_filenames(suffix)
    for root in (tmp_path, reference):
        for name in names:
            _write_model(root, name, external=True)
    expected = {name: _ort_result(tmp_path / name) for name in names}
    canonical_name = share._shared_data_name(suffix)
    assert share.share_external_models(reference, suffix=suffix)
    retained = tmp_path / "retained.bin"
    retained.write_bytes((reference / canonical_name).read_bytes())
    shutil.rmtree(reference)
    canonical = tmp_path / canonical_name
    if kind == "symlink":
        canonical.symlink_to(retained)
    else:
        os.link(retained, canonical)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(Exception) as caught:
        share.share_external_models(tmp_path, suffix=suffix)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    assert isinstance(caught.value, ValueError)
    assert canonical.is_symlink() == (kind == "symlink")
    assert canonical.stat().st_ino == retained.stat().st_ino
    _assert_pair(tmp_path, expected)
    for name in names:
        onnx.checker.check_model(str(tmp_path / name))


def test_sharing_keeps_an_explicit_model_directory_alias(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    expected = _pair(model)
    alias = tmp_path / "model-alias"
    alias.symlink_to(model, target_is_directory=True)
    assert share.share_external_models(alias)
    _assert_pair(model, expected)
    for name in expected:
        onnx.checker.check_model(str(model / name))


def test_small_inline_pair_rejects_foreign_canonical_data(tmp_path):
    weight = np.ones((2, 2), dtype=np.float32)
    for name in share._decoder_filenames():
        _write_model(tmp_path, name, weight=weight)
    canonical = tmp_path / "decoder_weights.data"
    canonical.write_bytes(b"foreign retained bytes")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="incompatible"):
        share.share_external_models(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_atomic_json_writer_removes_staging_after_serialization_failure(tmp_path):
    with pytest.raises(TypeError):
        share.write_artifact_json(tmp_path, "config.json", {"invalid": object()})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("auxiliary", ["aux.onnx", "nested/aux.onnx"])
@pytest.mark.parametrize("already_shared", [False, True])
def test_unrelated_invalid_graph_rejected_before_decoder_publication(tmp_path, auxiliary, already_shared):
    expected = _pair(tmp_path)
    retained = {p.name: p.read_bytes() for p in tmp_path.glob("*.onnx.data")}
    if already_shared:
        assert share.share_external_models(tmp_path)
    graph = tmp_path / auxiliary
    graph.parent.mkdir(parents=True, exist_ok=True)
    _write_model(graph.parent, graph.name, external=True)
    Path(f"{graph}.data").unlink()
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    for _ in range(2):
        with pytest.raises(ValueError, match="Cannot inventory independent graph") as caught:
            share.share_external_models(tmp_path)
        assert isinstance(caught.value.__cause__, ValueError)
        assert {
            path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
        } == before
    graph.unlink()
    assert share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.onnx.data")} == retained


ROOT = Path(share.__file__).resolve().parent


@pytest.mark.parametrize("suffix", [None, "int4", "q4f16"])
@pytest.mark.parametrize("foreign", [False, True])
def test_abrupt_exit_retry_preserves_models_and_source_data(tmp_path, suffix, foreign):
    model = tmp_path / "models"
    model.mkdir()
    names = share._decoder_filenames(suffix)
    for name in names:
        _write_model(model, name, external=True)
    original_data = {p.name: p.read_bytes() for p in model.glob("*.onnx.data")}
    expected = {name: _ort_result(model / name) for name in names}
    retained = {}
    if foreign:
        shutil.copy2(model / names[0], model / "foreign.onnx")
        nested = model / ".share-foreign"
        nested.mkdir()
        _write_model(nested, "aux.onnx", external=True)
        retained = {
            p.relative_to(model): p.read_bytes()
            for p in [model / "foreign.onnx", model / (names[0] + ".data"), *nested.iterdir()]
        }
    code = """import os, sys, share_weights as s
real = s._publish_data
def crash(staged, target):
    real(staged, target)
    os._exit(77)
s._publish_data = crash
s.share_external_models(sys.argv[1], suffix=sys.argv[2] or None)
"""
    env = os.environ | {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code, str(model), suffix or ""], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 77, result.stdout + result.stderr
    for name, values in expected.items():
        np.testing.assert_array_equal(_ort_result(model / name), values)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,share_weights as s;s.share_external_models(sys.argv[1],suffix=sys.argv[2] or None)",
            str(model),
            suffix or "",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name, values in expected.items():
        np.testing.assert_array_equal(_ort_result(model / name), values)
    if foreign:
        for name, data in retained.items():
            assert (model / name).read_bytes() == data
        _ort_result(model / "foreign.onnx")
        _ort_result(model / ".share-foreign" / "aux.onnx")
    else:
        selected = {*names, share._shared_data_name(suffix)}
        assert {p.name for p in model.iterdir()} == selected | set(original_data)
        copied = tmp_path / "copied"
        copied.mkdir()
        for name in selected:
            shutil.move(model / name, copied / name)
        assert {p.name: p.read_bytes() for p in model.iterdir()} == original_data
        for name, values in expected.items():
            np.testing.assert_array_equal(_ort_result(copied / name), values)


def test_copy_containment_precedes_mkdir(tmp_path):
    source, output, outside = (tmp_path / name for name in ("source", "output", "outside"))
    (source / "sub" / "new").mkdir(parents=True)
    (source / "sub" / "new" / "file").write_bytes(b"file")
    output.mkdir()
    outside.mkdir()
    (output / "sub").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="linked parent directory"):
        share.copy_artifact_entry(source, output, "sub/new/file")
    assert list(outside.iterdir()) == []


def test_legacy_unmarked_staging_is_preserved_and_diagnosed(tmp_path):
    _pair(tmp_path)
    abandoned = tmp_path / ".share-legacy"
    abandoned.mkdir()
    _write_model(abandoned, "decoder_init.onnx", external=True)
    (abandoned / "decoder_init.onnx.data").unlink()
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError) as caught:
        share.share_external_models(tmp_path)
    assert str(abandoned / "decoder_init.onnx") in str(caught.value)
    assert "before retrying" in str(caught.value)
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_parent_traversal_is_rejected_before_mkdir(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="escapes"):
        share.copy_artifact_entry(tmp_path, output, "../outside/new/file")
    assert list(tmp_path.iterdir()) == []


def test_sharing_retains_unreferenced_conventional_data(tmp_path):
    expected = _pair(tmp_path, inline=True)
    retained = {name + ".data": b"unrelated retained bytes" for name in share._decoder_filenames()}
    for name, data in retained.items():
        (tmp_path / name).write_bytes(data)
    assert share.share_external_models(tmp_path)
    assert share.share_external_models(tmp_path)
    _assert_pair(tmp_path, expected)
    for name, data in retained.items():
        assert (tmp_path / name).read_bytes() == data


def test_sharing_protects_independent_graph_storage(tmp_path):
    _pair(tmp_path, inline=True)
    data = (tmp_path / "decoder_init.onnx").read_bytes()
    offset = data.index(np.arange(512, dtype=np.float32).tobytes())
    _write_external_identity(tmp_path, "encoder.onnx", "decoder_init.onnx", offset)
    np.testing.assert_array_equal(_ort_result(tmp_path / "encoder.onnx"), [0])
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="destination conflict"):
        share.share_external_models(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    np.testing.assert_array_equal(_ort_result(tmp_path / "encoder.onnx"), [0])


@pytest.mark.parametrize("auxiliary", ["opaque.onnx", "nested/opaque.onnx"])
@pytest.mark.parametrize("already_shared", [False, True])
def test_opaque_files_are_preserved_during_sharing_and_retry(tmp_path, auxiliary, already_shared):
    expected = _pair(tmp_path)
    if already_shared:
        assert share.share_external_models(tmp_path)
    opaque = tmp_path / auxiliary
    opaque.parent.mkdir(parents=True, exist_ok=True)
    data = b"opaque bytes with no ONNX graph identity"
    opaque.write_bytes(data)
    for _ in range(2):
        assert share.share_external_models(tmp_path)
        assert opaque.read_bytes() == data
        _assert_pair(tmp_path, expected)
