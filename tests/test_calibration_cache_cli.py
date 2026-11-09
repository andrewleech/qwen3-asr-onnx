import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import awq_smooth
import quantize_static
from src.calibration_cache import model_fingerprint, recipe, save_cache, tokenizer_fingerprint
from src.onnx_export import export_onnx
from tests.artifact_fixtures import _ort_result, _write_external_identity
from tests.converter_fixtures import _source
from tests.tokenizer_fixtures import save_tokenizer_assets


def _awq_model():
    model = torch.nn.Module()
    model.thinker = torch.nn.Module()
    model.thinker.model = torch.nn.Module()
    model.thinker.model.embed_tokens = torch.nn.Embedding(8, 4)
    model.thinker.model.norm = torch.nn.LayerNorm(4)
    model.thinker.model.layers = torch.nn.ModuleList()
    model.config = SimpleNamespace(to_dict=lambda: {})
    return model


@pytest.mark.parametrize("retained", [False, True])
def test_awq_metadata_publication_preserves_release_and_retained_tensors(tmp_path, monkeypatch, retained):
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig

    from src.release_validation import graph_names, validate_release

    source, output = tmp_path / "checkpoint", tmp_path / "output"
    save_tokenizer_assets(source)
    if retained:
        output.mkdir()
        (output / "preprocessor_config.json").write_bytes(np.array([42], np.float32).tobytes())
        _write_external_identity(output, "aux.onnx", "preprocessor_config.json")
    model = _awq_model()
    model.config = Qwen3ASRConfig()
    model.config.thinker_config.text_config.vocab_size = 8
    model.config.thinker_config.text_config.hidden_size = 4
    model.config.thinker_config.text_config.rope_scaling = {}
    monkeypatch.setattr(awq_smooth, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(awq_smooth, "collect_activations", lambda *args, **kwargs: {"norm": torch.ones(4)})
    monkeypatch.setattr(awq_smooth, "apply_smoothing", lambda *args: None)

    def tiny_export(model, path, **kwargs):
        export_onnx(
            torch.nn.Identity(), (torch.ones(1, 4),), path, input_names=["x"], output_names=["y"], opset_version=17
        )

    for name in ("export_encoder", "export_decoder_init", "export_decoder_step"):
        monkeypatch.setattr(awq_smooth, name, tiny_export)
    monkeypatch.setattr(
        sys, "argv", ["awq_smooth.py", "--model", str(source), "--output", str(output), "--n-samples", "1"]
    )
    if retained:
        with pytest.raises(ValueError, match="retained graph"):
            awq_smooth.main()
        assert (output / "preprocessor_config.json").read_bytes() == np.array([42], np.float32).tobytes()
        np.testing.assert_array_equal(_ort_result(output / "aux.onnx"), [42])
        return
    awq_smooth.main()
    members = validate_release(output, graph_names())
    assert "preprocessor_config.json" in members
    preprocessing = json.loads((output / "preprocessor_config.json").read_text())
    assert preprocessing["sampling_rate"] == 16000
    assert preprocessing["feature_size"] == 128
    for name in graph_names():
        np.testing.assert_array_equal(_ort_result(output / name), np.ones((1, 4), np.float32))


@pytest.mark.parametrize("entry", ["hardlink", "symlink"])
def test_preprocessing_writer_replaces_entry_without_truncating_backing_file(tmp_path, entry):
    from export import write_preprocessor_config

    backing = tmp_path / "retained.data"
    backing.write_bytes(np.array([42], np.float32).tobytes())
    destination = tmp_path / "preprocessor_config.json"
    if entry == "hardlink":
        destination.hardlink_to(backing)
    else:
        destination.symlink_to(backing.name)
    _write_external_identity(tmp_path, "aux.onnx", backing.name)
    write_preprocessor_config(str(tmp_path))
    assert backing.read_bytes() == np.array([42], np.float32).tobytes()
    np.testing.assert_array_equal(_ort_result(tmp_path / "aux.onnx"), [42])
    assert not destination.is_symlink()
    assert json.loads(destination.read_text())["sampling_rate"] == 16000


def test_static_legacy_cache_is_preserved_and_only_matching_cache_is_reused(tmp_path, monkeypatch):
    source = _source(tmp_path / "source", "jinja")
    cache = tmp_path / "cache"
    cache.mkdir()
    legacy = cache / "calib_n1_s1.npz"
    previous = np.array([{"x": np.full((1, 4), 99, dtype=np.float32)}], dtype=object)
    np.savez(legacy, init_inputs=previous, step_inputs=previous)
    legacy_bytes = legacy.read_bytes()
    collected = []

    def collect(*args, **kwargs):
        collected.append(kwargs)
        samples = [{"x": np.ones((1, 4), dtype=np.float32)}]
        return samples, samples

    monkeypatch.setattr(quantize_static, "collect_calibration_data", collect)

    def run(name):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "quantize_static.py",
                "--input",
                str(source),
                "--output",
                str(tmp_path / name),
                "--cache-dir",
                str(cache),
                "--n-samples",
                "1",
                "--n-steps",
                "1",
                "--no-share-weights",
            ],
        )
        quantize_static.main()

    run("first")
    assert len(collected) == 1
    run("second")
    assert len(collected) == 1
    (source / "preprocessor_config.json").write_text('{"changed": true}')
    run("third")
    assert len(collected) == 2
    assert legacy.read_bytes() == legacy_bytes
    assert len(list(cache.glob("*.npz"))) == 3


def test_awq_legacy_cache_is_preserved_and_checkpoint_identity_controls_reuse(tmp_path, monkeypatch):
    source, output = tmp_path / "checkpoint", tmp_path / "output"
    save_tokenizer_assets(source)
    output.mkdir()
    legacy = output / "calibration_activations.npz"
    np.savez(legacy, norm=np.full(4, 99, dtype=np.float32))
    legacy_bytes = legacy.read_bytes()
    data = output / "encoder.onnx.data"
    np.array([42], dtype=np.float32).tofile(data)
    _write_external_identity(output, "aux.onnx", data.name)
    model = _awq_model()
    monkeypatch.setattr(awq_smooth, "load_model", lambda *args, **kwargs: copy.deepcopy(model))
    collected = []
    scales = []

    def collect(*args, **kwargs):
        collected.append(kwargs)
        return {"norm": torch.arange(1, 5, dtype=torch.float32)}

    def compute(activations, **kwargs):
        scales.append(activations["norm"].clone())
        return activations

    def tiny_export(model, path, **kwargs):
        export_onnx(
            torch.nn.Identity(), (torch.ones(1, 4),), path, input_names=["x"], output_names=["y"], opset_version=17
        )

    monkeypatch.setattr(awq_smooth, "collect_activations", collect)
    monkeypatch.setattr(awq_smooth, "compute_scales", compute)
    monkeypatch.setattr(awq_smooth, "apply_smoothing", lambda *args: None)
    for name in ("export_encoder", "export_decoder_init", "export_decoder_step"):
        monkeypatch.setattr(awq_smooth, name, tiny_export)
    monkeypatch.setattr(
        awq_smooth, "write_config", lambda model, directory: (Path(directory) / "config.json").write_text("{}")
    )
    monkeypatch.setattr(
        sys, "argv", ["awq_smooth.py", "--model", str(source), "--output", str(output), "--n-samples", "1"]
    )
    awq_smooth.main()
    awq_smooth.main()
    assert len(collected) == 1
    for value in scales:
        torch.testing.assert_close(value, torch.arange(1, 5, dtype=torch.float32))
    with torch.no_grad():
        model.thinker.model.embed_tokens.weight[0, 0] += 1
    awq_smooth.main()
    assert len(collected) == 2
    assert legacy.read_bytes() == legacy_bytes
    assert data.read_bytes() == np.array([42], dtype=np.float32).tobytes()
    np.testing.assert_array_equal(_ort_result(output / "aux.onnx"), [42])


@pytest.mark.parametrize("origin", ["cached", "collected"])
@pytest.mark.parametrize("problem", ["keys", "shape", "negative", "integer"])
def test_awq_rejects_malformed_activations_before_publication(tmp_path, monkeypatch, origin, problem):
    source, output = tmp_path / "checkpoint", tmp_path / "output"
    tokenizer = save_tokenizer_assets(source)
    output.mkdir()
    scales = output / "smoothing_scales.npz"
    scales.write_bytes(b"prior scales")
    model = _awq_model()
    activations = {"norm": torch.ones(4)}
    if problem == "keys":
        activations = {"unknown": torch.ones(4)}
    elif problem == "shape":
        activations = {"norm": torch.ones(1)}
    elif problem == "negative":
        activations = {"norm": -torch.ones(4)}
    elif problem == "integer":
        activations = {"norm": torch.ones(4, dtype=torch.int64)}
    identity = recipe(
        "awq",
        {"checkpoint": model_fingerprint(model), "tokenizer": tokenizer_fingerprint(tokenizer)},
        {"n_samples": 1, "device": "cpu"},
        Path(awq_smooth.__file__),
    )
    cache = output / "explicit.npz"
    if origin == "cached":
        save_cache(cache, identity, {k: v.numpy() for k, v in activations.items()})
    before = cache.read_bytes() if cache.exists() else None
    monkeypatch.setattr(awq_smooth, "load_model", lambda *args, **kwargs: copy.deepcopy(model))
    monkeypatch.setattr(awq_smooth, "collect_activations", lambda *args, **kwargs: activations)
    monkeypatch.setattr(
        awq_smooth, "apply_smoothing", lambda *args: pytest.fail("Invalid activations reached smoothing")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "awq_smooth.py",
            "--model",
            str(source),
            "--output",
            str(output),
            "--n-samples",
            "1",
            "--activations-cache",
            str(cache),
        ],
    )
    with pytest.raises(ValueError, match="activation|Unverifiable"):
        awq_smooth.main()
    assert scales.read_bytes() == b"prior scales"
    assert (cache.read_bytes() if cache.exists() else None) == before


def test_awq_preserves_explicit_dangling_cache_before_collection(tmp_path, monkeypatch):
    source, output = tmp_path / "checkpoint", tmp_path / "output"
    save_tokenizer_assets(source)
    output.mkdir()
    cache = output / "explicit.npz"
    cache.symlink_to(tmp_path / "unknown.npz")
    scales = output / "smoothing_scales.npz"
    scales.write_bytes(b"prior scales")
    monkeypatch.setattr(awq_smooth, "load_model", lambda *args, **kwargs: _awq_model())
    monkeypatch.setattr(
        awq_smooth, "collect_activations", lambda *args, **kwargs: pytest.fail("Unknown cache reached collection")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["awq_smooth.py", "--model", str(source), "--output", str(output), "--activations-cache", str(cache)],
    )
    with pytest.raises(ValueError, match="Unverifiable"):
        awq_smooth.main()
    assert cache.is_symlink() and cache.readlink() == tmp_path / "unknown.npz"
    assert scales.read_bytes() == b"prior scales"
