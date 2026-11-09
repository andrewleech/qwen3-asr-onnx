import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.artifact_fixtures import _ort_result, _release_input, _write_model

ROOT = Path(__file__).resolve().parents[1]


def test_quantization_trial_keeps_preexisting_source_dependencies(tmp_path):
    (tmp_path / "output").mkdir()
    model = _release_input(tmp_path / "output/qwen3-asr-0.6b")
    _write_model(model, "encoder.onnx", external=True, data_name="11111111-2222-3333-4444-555555555555.data")
    _write_model(model, "existing_augment.onnx")
    before = {path.name: path.read_bytes() for path in model.iterdir() if path.is_file()}
    expected = _ort_result(model / "encoder.onnx")
    tools = tmp_path / "bin"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        + """
import pathlib
import sys
args = sys.argv
output = pathlib.Path(args[args.index('--output') + 1])
if args[3] == 'quantize_nbits.py':
    output.mkdir(parents=True, exist_ok=True)
else:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{}')
"""
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "run_0.6b_quant_trials.sh")],
        cwd=tmp_path,
        env=os.environ | {"PATH": str(tools) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name: path.read_bytes() for path in model.iterdir() if path.is_file()} == before
    np.testing.assert_array_equal(_ort_result(model / "encoder.onnx"), expected)
