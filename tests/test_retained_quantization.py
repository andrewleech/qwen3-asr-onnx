import json
import os
import subprocess
import sys
from pathlib import Path

from src.onnx_artifacts import retained_quantization
from tests.artifact_fixtures import _write_model


def test_retained_tags_come_from_decoded_graph_roles(tmp_path):
    _write_model(tmp_path, "retained.int8.onnx")
    (tmp_path / "opaque.int8.onnx").write_bytes(b"raw tensor storage")
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization": {"retained": "int8_dynamic", "opaque": "int8_dynamic", "selected": "old"}})
    )
    retained, tags = retained_quantization(tmp_path, ".int8", {"selected.int8.onnx"})
    assert retained == {"retained.int8.onnx"}
    assert tags == {"retained": "int8_dynamic"}


def test_dynamic_cli_does_not_republish_opaque_file_as_quantized_model(tmp_path):
    root = Path(__file__).resolve().parents[1]
    first, second, output = (tmp_path / name for name in ("first", "second", "output"))
    for source in (first, second):
        source.mkdir()
        (source / "config.json").write_text("{}")
    _write_model(first, "encoder.onnx")
    _write_model(second, "decoder_init.onnx")

    def run(source):
        result = subprocess.run(
            [
                sys.executable,
                str(root / "quantize.py"),
                "--input",
                str(source),
                "--output",
                str(output),
                "--no-share-weights",
            ],
            capture_output=True,
            text=True,
            timeout=90,
            env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run(first)
    opaque = output / "opaque.int8.onnx"
    opaque.write_bytes(b"raw tensor storage")
    config = json.loads((output / "config.json").read_text())
    config["quantization"]["opaque"] = "int8_dynamic"
    (output / "config.json").write_text(json.dumps(config))
    run(second)
    tags = json.loads((output / "config.json").read_text())["quantization"]
    assert tags == {"encoder": "int8_dynamic", "decoder_init": "int8_dynamic"}
    assert opaque.read_bytes() == b"raw tensor storage"
