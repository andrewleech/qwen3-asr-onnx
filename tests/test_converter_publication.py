import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest

from src import artifact_publication
from src.onnx_artifacts import external_data_files
from tests.artifact_fixtures import _ort_result, _write_external_identity, _write_model
from tests.converter_fixtures import bytes_under, command, input_for

SCRIPTS = ["convert_fp16.py", "quantize_static.py", "quantize_nbits.py"]


def _args(root, script, source, target):
    args = command(root, script, source, target)
    if script == "quantize_static.py":
        from quantize_static import calibration_identity
        from src.calibration_cache import cache_key, pack_samples, save_cache

        width = onnx.load(source / "decoder_init.onnx").graph.input[0].type.tensor_type.shape.dim[1].dim_value
        identity = calibration_identity(source, 1, 1)
        samples = [{"x": np.ones((1, width), np.float32)}]
        save_cache(
            root / "cache" / f"calib_{cache_key(identity)}.npz",
            identity,
            pack_samples({"init": samples, "step": samples}),
        )
    return args


def _previous_graphs(target, script):
    target.mkdir()
    suffix = ".int4" if script == "quantize_nbits.py" else ""
    for stem in ("decoder_init", "decoder_step"):
        _write_model(target, f"{stem}{suffix}.onnx", weight=np.full((4, 3), 42, np.float32))
    return suffix


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py"])
@pytest.mark.parametrize("canonical_state", ["absent", "identical", "incompatible", "unused"])
def test_converter_sharing_preserves_canonical_policy(tmp_path, script, canonical_state):
    source = input_for(tmp_path, script)
    if canonical_state != "unused":
        for name in ("decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, weight=np.arange(4096, dtype=np.float32).reshape(64, 64) / 4096)
    target = tmp_path / "target"
    _previous_graphs(target, script)
    canonical = target / "decoder_weights.data"
    if canonical_state == "identical":
        seed_root = tmp_path / "seed"
        seed_root.mkdir()
        seed = seed_root / "output"
        args = [arg for arg in _args(seed_root, script, source, seed) if arg != "--no-share-weights"]
        result = subprocess.run(args, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        canonical.write_bytes((seed / canonical.name).read_bytes())
        _write_external_identity(target, "aux.onnx", canonical.name, dtype=np.uint8)
        expected = _ort_result(target / "aux.onnx")
        inode = canonical.stat().st_ino
    elif canonical_state in {"incompatible", "unused"}:
        canonical.write_bytes(b"unowned unused canonical bytes")
    before_source, before_target = bytes_under(source), bytes_under(target)
    args = [arg for arg in _args(tmp_path, script, source, target) if arg != "--no-share-weights"]
    result = subprocess.run(args, capture_output=True, text=True, timeout=90)
    assert bytes_under(source) == before_source
    if canonical_state == "incompatible":
        assert result.returncode != 0
        assert "incompatible bytes" in result.stdout + result.stderr
        assert bytes_under(target) == before_target
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        for stem in ("decoder_init", "decoder_step"):
            assert external_data_files(target / f"{stem}.onnx") == (
                set() if canonical_state == "unused" else {canonical}
            )
            np.testing.assert_allclose(
                _ort_result(target / f"{stem}.onnx"), _ort_result(source / f"{stem}.onnx"), atol=0.1, rtol=0.05
            )
        if canonical_state in {"identical", "unused"}:
            assert canonical.read_bytes() == before_target[Path(canonical.name)]
        if canonical_state == "identical":
            assert canonical.stat().st_ino == inode
            np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), expected)


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("fault", ["second-producer", "second-graph-install"])
def test_converter_failures_preserve_previous_outputs(tmp_path, monkeypatch, script, fault):
    source = input_for(tmp_path, script)
    target = tmp_path / "target"
    suffix = _previous_graphs(target, script)
    before_source, before_target = bytes_under(source), bytes_under(target)
    expected = _ort_result(target / f"decoder_init{suffix}.onnx")
    module = importlib.import_module(script.removesuffix(".py"))
    args = _args(tmp_path, script, source, target)
    monkeypatch.setattr(sys, "argv", args[1:])
    if fault == "second-producer":
        name = "convert_model_fp16" if script == "convert_fp16.py" else "quantize_decoder"
        original = getattr(module, name)
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            original(*args, **kwargs)
            calls += 1
            if calls == 2:
                raise RuntimeError("injected second producer failure")

        monkeypatch.setattr(module, name, fail_second)
        error = RuntimeError
    else:
        original = artifact_publication.os.replace
        failed = False

        def fail_install(source, destination):
            nonlocal failed
            if not failed and Path(destination) == target / f"decoder_step{suffix}.onnx":
                failed = True
                raise OSError("injected graph install failure")
            return original(source, destination)

        monkeypatch.setattr(artifact_publication.os, "replace", fail_install)
        error = OSError
    with pytest.raises(error, match="injected"):
        module.main()
    assert bytes_under(source) == before_source
    assert bytes_under(target) == before_target
    np.testing.assert_array_equal(_ort_result(target / f"decoder_init{suffix}.onnx"), expected)


@pytest.mark.parametrize("script", SCRIPTS)
def test_actual_later_external_collision_precedes_all_publication(tmp_path, script):
    source = input_for(tmp_path, script)
    width = 128 if script == "quantize_nbits.py" else 64
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        _write_model(
            source, name, weight=np.arange(width * width, dtype=np.float32).reshape(width, width) / (width * width)
        )
    target = tmp_path / "target"
    suffix = _previous_graphs(target, script)
    data = target / f"decoder_step{suffix}.onnx.data"
    data.write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", data.name)
    before_source, before_target = bytes_under(source), bytes_under(target)
    result = subprocess.run(_args(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "conflict" in result.stdout + result.stderr
    assert bytes_under(source) == before_source
    assert bytes_under(target) == before_target
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])
