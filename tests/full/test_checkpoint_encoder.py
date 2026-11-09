import importlib.metadata
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from huggingface_hub import snapshot_download
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
from safetensors import safe_open

from src.encoder_wrapper import EncoderWrapper
from src.native_encoder import run_native_encoder
from tests.encoder_helpers import FRAME_LENGTHS, export_encoder_at_100

pytestmark = pytest.mark.integration

CHECKPOINTS = (
    ("Qwen/Qwen3-ASR-0.6B", "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"),
    ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
)


@pytest.fixture
def encoder_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def load_audio_tower(model_id, revision):
    snapshot = Path(
        snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=["config.json", "*.safetensors", "*.index.json"],
        )
    )
    raw = json.loads((snapshot / "config.json").read_text())
    config = Qwen3ASRAudioEncoderConfig(**raw["thinker_config"]["audio_config"])
    config._attn_implementation = "eager"
    tower = Qwen3ASRAudioEncoder(config).eval()
    prefix = "thinker.audio_tower."
    state = {}
    for checkpoint in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
            tensor_keys = tensors.keys()
            for key in tensor_keys:
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = tensors.get_tensor(key).float()
    tower.load_state_dict(state, strict=True)
    return tower


def feature_error(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert np.isfinite(expected).all()
    delta = actual - expected
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(expected)),
    }


@pytest.mark.parametrize("model_id,revision", CHECKPOINTS, ids=["0.6b", "1.7b"])
def test_pinned_checkpoint_encoder(model_id, revision, tmp_path, encoder_threads):
    tower = load_audio_tower(model_id, revision)
    wrapper = EncoderWrapper(tower).eval()
    path = tmp_path / "encoder.onnx"
    export_encoder_at_100(wrapper, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    rows = []
    with torch.no_grad():
        for frames in FRAME_LENGTHS:
            mel = torch.randn(1, 128, frames, generator=torch.Generator().manual_seed(frames))
            native = run_native_encoder(tower, mel).numpy()
            eager = wrapper(mel).numpy()
            exported = session.run(["audio_features"], {"mel": mel.numpy()})[0]
            expected_tokens = (frames // 100) * 13 + ((frames % 100) + 7) // 8
            assert native.shape == (1, expected_tokens, tower.config.output_dim)
            row = {
                "model": model_id,
                "revision": revision,
                "frames": frames,
                "shape": list(native.shape),
                "tokens": expected_tokens,
                "finite": bool(np.isfinite(exported).all()),
                "wrapper_vs_native": feature_error(eager, native),
                "onnx_vs_native": feature_error(exported, native),
                "onnx_vs_wrapper": feature_error(exported, eager),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            for actual, reference in [(eager, native), (exported, native), (exported, eager)]:
                np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-4)
                error = feature_error(actual, reference)
                assert error["max_abs"] <= 1e-4
                assert error["relative_l2"] < 1e-4
    report = {
        "model": model_id,
        "revision": revision,
        "traced_frames": 100,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "qwen-asr", "onnx", "onnxruntime", "transformers"]
        },
        "results": rows,
    }
    (tmp_path / "encoder-report.json").write_text(json.dumps(report, indent=2) + "\n")
