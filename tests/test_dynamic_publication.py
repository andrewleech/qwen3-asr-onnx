import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import quantize
from src import artifact_publication
from src.onnx_artifacts import external_data_files
from tests.artifact_fixtures import _ort_result, _write_external_identity, _write_model
from tests.converter_fixtures import ROOT, bytes_under


def _source(root):
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"quantization": {}}))
    weights = np.arange(4096, dtype=np.float32).reshape(64, 64) / 4096
    for name in ("encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"):
        _write_model(root, name, weight=weights)
    return root


@pytest.mark.parametrize("preexisting", [False, True])
def test_dynamic_cli_publishes_only_required_storage(tmp_path, preexisting):
    source = _source(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    names = ["encoder.int8.onnx.data", "decoder_step.int8.onnx.data"]
    previous = {}
    if preexisting:
        for name in names:
            (target / name).write_bytes(b"preexisting unrelated data")
            previous[name] = (target / name).read_bytes()
    before = bytes_under(source)
    result = subprocess.run(
        [sys.executable, str(ROOT / "quantize.py"), "--input", str(source), "--output", str(target)],
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert bytes_under(source) == before
    for name in names:
        if preexisting:
            assert (target / name).read_bytes() == previous[name]
        else:
            assert not (target / name).exists()
    for name in ("encoder", "decoder_step"):
        assert not external_data_files(target / f"{name}.int8.onnx")
    dependencies = external_data_files(target / "decoder_init.int8.onnx")
    assert dependencies == {target / "decoder_init.int8.onnx.data"}
    for name in ("encoder", "decoder_init", "decoder_step"):
        np.testing.assert_allclose(
            _ort_result(target / f"{name}.int8.onnx"), _ort_result(source / f"{name}.onnx"), rtol=0.03, atol=0.25
        )
    assert not [path for path in target.iterdir() if path.is_dir()]


def test_failed_quantizer_keeps_previous_graph_and_data(tmp_path, monkeypatch):
    source = _source(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    _write_model(target, "encoder.int8.onnx")
    (target / "encoder.int8.onnx.data").write_bytes(b"retained unrelated bytes")
    before_source, before_target = bytes_under(source), bytes_under(target)
    monkeypatch.setattr(sys, "argv", ["quantize.py", "--input", str(source), "--output", str(target)])

    def fail_after_writing(_source, output, **_kwargs):
        Path(output).write_bytes(b"incomplete graph")
        Path(str(output) + ".data").write_bytes(b"incomplete new weights")
        raise RuntimeError("injected quantizer failure")

    monkeypatch.setattr(quantize, "quantize_dynamic", fail_after_writing)
    with pytest.raises(RuntimeError, match="quantizer failure"):
        quantize.main()
    assert bytes_under(source) == before_source
    assert bytes_under(target) == before_target


def test_external_graph_publication_failure_restores_previous_pair(tmp_path, monkeypatch):
    source = _source(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    previous_data = target / "decoder_init.int8.onnx.data"
    previous_data.write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(target, "decoder_init.int8.onnx", previous_data.name)
    previous_graph = (target / "decoder_init.int8.onnx").read_bytes()
    before = previous_data.read_bytes()
    monkeypatch.setattr(sys, "argv", ["quantize.py", "--input", str(source), "--output", str(target)])
    replace = artifact_publication.os.replace
    interrupted = False

    def fail_graph_install(original, destination):
        nonlocal interrupted
        if not interrupted and Path(destination) == target / "decoder_init.int8.onnx":
            interrupted = True
            raise OSError("injected graph install failure")
        return replace(original, destination)

    monkeypatch.setattr(artifact_publication.os, "replace", fail_graph_install)
    with pytest.raises(OSError, match="graph install failure"):
        quantize.main()
    assert interrupted
    assert (target / "decoder_init.int8.onnx").read_bytes() == previous_graph
    assert previous_data.read_bytes() == before
    np.testing.assert_array_equal(_ort_result(target / "decoder_init.int8.onnx"), [42])


@pytest.mark.parametrize("graph", ["encoder", "decoder_step"])
def test_dynamic_preserves_retained_dependency_of_inlined_graph(tmp_path, monkeypatch, graph):
    source = _source(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    data = target / f"{graph}.int8.onnx.data"
    data.write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", data.name)
    before_source = bytes_under(source)
    before_data, before_aux = data.read_bytes(), (target / "aux.onnx").read_bytes()
    monkeypatch.setattr(sys, "argv", ["quantize.py", "--input", str(source), "--output", str(target)])
    quantize.main()
    assert bytes_under(source) == before_source
    assert data.read_bytes() == before_data
    assert (target / "aux.onnx").read_bytes() == before_aux
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])
    assert not external_data_files(target / f"{graph}.int8.onnx")


def test_final_external_sidecar_collision_precedes_all_publication(tmp_path, monkeypatch):
    source = _source(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    data = target / "decoder_init.int8.onnx.data"
    data.write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", data.name)
    before_source, before_target = bytes_under(source), bytes_under(target)
    monkeypatch.setattr(sys, "argv", ["quantize.py", "--input", str(source), "--output", str(target)])
    with pytest.raises(ValueError, match="conflict"):
        quantize.main()
    assert bytes_under(source) == before_source
    assert bytes_under(target) == before_target
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])
