import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

import src.decoder_wrapper as decoders


def _write_external(path, value, location="onnx__MatMul_shared", offset=0):
    path = Path(path)
    tensor = numpy_helper.from_array(np.array([value], dtype=np.float32), "weight")
    onnx.external_data_helper.set_external_data(tensor, location, offset=offset, length=4)
    tensor.ClearField("raw_data")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["weight"], ["output"])],
        "fixture",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        [tensor],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    (path.parent / location).write_bytes(b"x" * offset + np.array([value], dtype=np.float32).tobytes())
    path.write_bytes(model.SerializeToString())


def _value(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return float(ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]).run(None, {})[0][0])


def _export_pair(tmp_path, monkeypatch, *, failure=False):
    text = SimpleNamespace(hidden_size=4, num_hidden_layers=1, num_key_value_heads=1, head_dim=4)
    model = SimpleNamespace(
        config=SimpleNamespace(thinker_config=SimpleNamespace(text_config=text)),
        thinker=SimpleNamespace(model=None, lm_head=None),
    )
    monkeypatch.setattr(decoders, "DecoderInitWrapper", lambda *args: torch.nn.Identity())
    monkeypatch.setattr(decoders, "DecoderStepWrapper", lambda *args: torch.nn.Identity())
    calls = []

    def external_writer(module, inputs, target, **kwargs):
        target = Path(target)
        calls.append(target)
        _write_external(target, 3.25 if len(calls) == 1 else 7.5, offset=7)
        if failure and len(calls) == 2:
            raise RuntimeError("tracing failed after writing external data")

    monkeypatch.setattr(torch.onnx, "export", external_writer)
    first, second = tmp_path / "decoder_init.onnx", tmp_path / "decoder_step.onnx"
    decoders.export_decoder_init(model, str(first))
    if failure:
        with pytest.raises(RuntimeError, match="tracing failed"):
            decoders.export_decoder_step(model, str(second))
    else:
        decoders.export_decoder_step(model, str(second))
    return first, second, calls


@pytest.mark.parametrize("failure", [False, True])
def test_decoder_exports_cannot_share_foreign_writer_namespace(tmp_path, monkeypatch, failure):
    first, second, calls = _export_pair(tmp_path, monkeypatch, failure=failure)
    assert _value(first) == 3.25
    if not failure:
        assert _value(second) == 7.5
    assert len({path.parent for path in calls}) == 2
    assert all(path.parent != tmp_path for path in calls)
    assert not (tmp_path / "onnx__MatMul_shared").exists()


def test_export_refuses_data_name_used_by_retained_graph(tmp_path, monkeypatch):
    first = tmp_path / "decoder_init.onnx"
    _write_external(tmp_path / "aux.onnx", 42.0, location=first.name + ".data")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    from src.onnx_export import export_onnx

    monkeypatch.setattr(torch.onnx, "export", lambda module, inputs, path, **kwargs: _write_external(path, 7.5))
    with pytest.raises(ValueError, match="retained graph destination conflict"):
        export_onnx(torch.nn.Identity(), (), str(first), opset_version=17)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert _value(tmp_path / "aux.onnx") == 42.0


def test_export_publication_failure_restores_graph_and_data(tmp_path, monkeypatch):
    import src.artifact_publication as publication
    from src.onnx_export import export_onnx

    target = tmp_path / "decoder_init.onnx"
    _write_external(target, 3.25, location=target.name + ".data")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(torch.onnx, "export", lambda module, inputs, path, **kwargs: _write_external(path, 7.5))
    replace = publication.os.replace
    failed = False

    def interrupt(source, destination):
        nonlocal failed
        if Path(destination) == target and Path(source).name == "model.proto" and not failed:
            failed = True
            raise OSError("injected graph installation failure")
        return replace(source, destination)

    monkeypatch.setattr(publication.os, "replace", interrupt)
    with pytest.raises(OSError, match="injected graph installation"):
        export_onnx(torch.nn.Identity(), (), str(target), opset_version=17)
    assert failed
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert _value(target) == 3.25


def test_invalid_exported_range_preserves_prior_graph(tmp_path, monkeypatch):
    from src.onnx_export import export_onnx

    target = tmp_path / "encoder.onnx"
    _write_external(target, 3.25, location=target.name + ".data")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def incomplete_writer(module, inputs, path, **kwargs):
        _write_external(path, 7.5)
        (Path(path).parent / "onnx__MatMul_shared").write_bytes(b"xx")

    monkeypatch.setattr(torch.onnx, "export", incomplete_writer)
    with pytest.raises(ValueError, match="Invalid external data range"):
        export_onnx(torch.nn.Identity(), (), str(target), opset_version=17)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert _value(target) == 3.25


@pytest.mark.parametrize("entry", ["encoder.onnx", "encoder.onnx.data"])
def test_nonregular_generated_destinations_preserve_directory_contents(tmp_path, monkeypatch, entry):
    from src.onnx_export import export_onnx

    target = tmp_path / "encoder.onnx"
    directory = tmp_path / entry
    directory.mkdir()
    sentinel = directory / "user-content.txt"
    sentinel.write_text("preserve this directory")
    original_stat = Path.stat

    def single_link_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == directory:
            # FAT reports one link for directories too; link count cannot prove file type.
            fields = list(result)
            fields[3] = 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", single_link_stat)
    monkeypatch.setattr(torch.onnx, "export", lambda module, inputs, path, **kwargs: _write_external(path, 7.5))
    with pytest.raises(ValueError, match="not a regular file"):
        export_onnx(torch.nn.Identity(), (), str(target), opset_version=17)
    assert directory.is_dir()
    assert sentinel.read_text() == "preserve this directory"
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == [
        entry,
        entry + "/user-content.txt",
    ]


def test_regular_generated_artifacts_can_be_replaced(tmp_path, monkeypatch):
    from src.onnx_export import export_onnx

    target = tmp_path / "encoder.onnx"
    _write_external(target, 3.25, location=target.name + ".data")
    monkeypatch.setattr(torch.onnx, "export", lambda module, inputs, path, **kwargs: _write_external(path, 7.5))
    export_onnx(torch.nn.Identity(), (), str(target), opset_version=17)
    assert _value(target) == 7.5
