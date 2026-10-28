import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import export
from src.onnx_export import export_onnx
from tests.test_export_ownership import _value, _write_external
from tests.tokenizer_fixtures import save_tokenizer_assets


@pytest.mark.parametrize(
    "role,dtype", [("encoder", "fp32"), ("decoder_step", "fp32"), ("encoder", "fp16"), ("decoder_step", "fp16")]
)
def test_export_cli_preserves_borrowed_conventional_storage(tmp_path, monkeypatch, role, dtype):
    checkpoint, output = tmp_path / "checkpoint", tmp_path / "output"
    save_tokenizer_assets(checkpoint)
    output.mkdir()
    data = output / f"{role}.onnx.data"
    np.array([42], dtype=np.float32).tofile(data)
    _write_external(output / "aux.onnx", 42.0, location=data.name)
    expected = _value(output / "aux.onnx")
    model = SimpleNamespace(thinker=SimpleNamespace(model=SimpleNamespace(embed_tokens=torch.nn.Embedding(8, 4))))
    monkeypatch.setattr(export, "load_model", lambda *args, **kwargs: model)

    def tiny_export(model, path, **kwargs):
        export_onnx(
            torch.nn.Identity(), (torch.ones(1, 4),), path, input_names=["x"], output_names=["y"], opset_version=17
        )

    for name in ("export_encoder", "export_decoder_init", "export_decoder_step"):
        monkeypatch.setattr(export, name, tiny_export)
    monkeypatch.setattr(export, "write_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["export.py", "--model", str(checkpoint), "--output", str(output), "--dtype", dtype, "--no-share-weights"],
    )
    export.main()
    assert data.read_bytes() == np.array([42], dtype=np.float32).tobytes()
    np.testing.assert_array_equal(_value(output / "aux.onnx"), expected)
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    actual = ort.InferenceSession(str(output / f"{role}.onnx"), options).run(
        None, {"x": np.ones((1, 4), dtype=np.float32)}
    )[0]
    np.testing.assert_array_equal(actual, np.ones((1, 4)))


def _awq_setup(tmp_path, monkeypatch):
    import awq_smooth

    checkpoint, output = tmp_path / "checkpoint", tmp_path / "output"
    save_tokenizer_assets(checkpoint)
    output.mkdir()
    model = SimpleNamespace(thinker=SimpleNamespace(model=SimpleNamespace(embed_tokens=torch.nn.Embedding(8, 4))))
    monkeypatch.setattr(awq_smooth, "load_model", lambda *a, **k: model)
    monkeypatch.setattr(awq_smooth, "collect_activations", lambda *a, **k: {"norm": torch.ones(4)})
    monkeypatch.setattr(awq_smooth, "apply_smoothing", lambda *a, **k: None)
    monkeypatch.setattr(awq_smooth, "write_config", lambda *a, **k: None)

    def tiny_export(model, path, **kwargs):
        export_onnx(
            torch.nn.Identity(), (torch.ones(1, 4),), path, input_names=["x"], output_names=["y"], opset_version=17
        )

    for name in ("export_encoder", "export_decoder_init", "export_decoder_step"):
        monkeypatch.setattr(awq_smooth, name, tiny_export)
    return awq_smooth, checkpoint, output


def test_awq_cli_preserves_borrowed_encoder_storage(tmp_path, monkeypatch):
    module, checkpoint, output = _awq_setup(tmp_path, monkeypatch)
    raw = output / "encoder.onnx.data"
    _write_external(output / "aux.onnx", 42.0, location=raw.name)
    before = raw.read_bytes()
    monkeypatch.setattr(sys, "argv", ["awq_smooth.py", "--model", str(checkpoint), "--output", str(output)])
    module.main()
    assert raw.read_bytes() == before
    assert _value(output / "aux.onnx") == 42.0


def test_awq_fresh_cache_stem_writes_effective_npz(tmp_path, monkeypatch):
    module, checkpoint, output = _awq_setup(tmp_path, monkeypatch)
    stem = output / "fresh-activations"
    monkeypatch.setattr(
        sys,
        "argv",
        ["awq_smooth.py", "--model", str(checkpoint), "--output", str(output), "--activations-cache", str(stem)],
    )
    module.main()
    assert not stem.exists()
    with np.load(str(stem) + ".npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["norm"], np.ones(4))


def test_graph_symlink_keeps_lexical_external_data_base(tmp_path):
    from src.onnx_artifacts import external_data_files

    physical, lexical = tmp_path / "physical", tmp_path / "lexical"
    physical.mkdir()
    lexical.mkdir()
    _write_external(physical / "source.onnx", 99.0, location="weights.data")
    graph = lexical / "source.onnx"
    graph.symlink_to(physical / "source.onnx")
    (lexical / "weights.data").write_bytes(np.array([42], dtype=np.float32).tobytes())
    before = (physical / "weights.data").read_bytes(), (lexical / "weights.data").read_bytes()
    assert _value(graph) == 42.0
    assert external_data_files(graph) == {lexical / "weights.data"}
    assert ((physical / "weights.data").read_bytes(), (lexical / "weights.data").read_bytes()) == before
