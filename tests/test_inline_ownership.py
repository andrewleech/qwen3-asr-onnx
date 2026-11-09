import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import export
from src.onnx_export import export_onnx
from tests.artifact_fixtures import _ort_result, _write_external_identity
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
    _write_external_identity(output, "aux.onnx", data.name)
    expected = _ort_result(output / "aux.onnx")
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
    np.testing.assert_array_equal(_ort_result(output / "aux.onnx"), expected)
    np.testing.assert_array_equal(_ort_result(output / f"{role}.onnx"), np.ones((1, 4)))
