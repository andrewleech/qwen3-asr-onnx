from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

from src.encoder_wrapper import EncoderWrapper, export_encoder
from src.native_encoder import run_native_encoder
from tests.encoder_helpers import FRAME_LENGTHS, export_encoder_at_100


def assert_features_match(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(expected).all()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_l2 < 1e-5


@pytest.fixture(scope="module", autouse=True)
def encoder_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def native_tower():
    with torch.random.fork_rng():
        torch.manual_seed(4342)
        config = Qwen3ASRAudioEncoderConfig(
            d_model=32,
            encoder_attention_heads=4,
            encoder_layers=2,
            encoder_ffn_dim=64,
            downsample_hidden_size=4,
            output_dim=16,
            n_window=50,
            n_window_infer=800,
            initializer_range=0.2,
        )
        config._attn_implementation = "eager"
        return Qwen3ASRAudioEncoder(config).eval()


@pytest.fixture(scope="module")
def encoder_wrapper(native_tower):
    return EncoderWrapper(native_tower).eval()


@pytest.fixture(scope="module", params=["trace100", "export_helper997"])
def dynamic_onnx(encoder_wrapper, native_tower, tmp_path_factory, request):
    path = tmp_path_factory.mktemp("native-encoder") / "encoder.onnx"
    if request.param == "export_helper997":
        model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=native_tower))
        export_encoder(model, str(path))
    else:
        export_encoder_at_100(encoder_wrapper, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("frames", FRAME_LENGTHS)
def test_wrapper_matches_native(native_tower, encoder_wrapper, frames):
    mel = torch.randn(1, 128, frames, generator=torch.Generator().manual_seed(frames))
    with torch.no_grad():
        reference = run_native_encoder(native_tower, mel).numpy()
        eager = encoder_wrapper(mel).numpy()
    assert reference.shape == (1, (frames // 100) * 13 + ((frames % 100) + 7) // 8, 16)
    assert_features_match(eager, reference)


@pytest.mark.parametrize("frames", FRAME_LENGTHS)
def test_dynamic_onnx_matches_native(native_tower, encoder_wrapper, dynamic_onnx, frames):
    mel = torch.randn(1, 128, frames, generator=torch.Generator().manual_seed(frames))
    with torch.no_grad():
        reference = run_native_encoder(native_tower, mel).numpy()
        eager = encoder_wrapper(mel).numpy()
    actual = dynamic_onnx.run(["audio_features"], {"mel": mel.numpy()})[0]
    assert_features_match(actual, reference)
    assert_features_match(actual, eager)


def test_native_reference_preserves_attention_windows(native_tower):
    mel = torch.randn(1, 128, 801, generator=torch.Generator().manual_seed(801))
    with torch.no_grad():
        windowed = run_native_encoder(native_tower, mel)
        global_attention = native_tower(mel[0], feature_lens=torch.tensor([801])).last_hidden_state.unsqueeze(0)
    assert torch.max(torch.abs(windowed - global_attention)).item() > 1e-3


def test_native_reference_cleans_up_hooks_after_error(native_tower, monkeypatch):
    before = [dict(layer.self_attn._forward_pre_hooks) for layer in native_tower.layers]

    def fail_forward(*args, **kwargs):
        raise RuntimeError("native forward failed")

    monkeypatch.setattr(native_tower, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="native forward failed"):
        run_native_encoder(native_tower, torch.zeros(1, 128, 50))
    assert [dict(layer.self_attn._forward_pre_hooks) for layer in native_tower.layers] == before


def test_native_reference_rejects_non_tensor_output(native_tower, monkeypatch):
    before = [dict(layer.self_attn._forward_pre_hooks) for layer in native_tower.layers]
    monkeypatch.setattr(native_tower, "forward", lambda *args, **kwargs: SimpleNamespace(last_hidden_state=None))
    with pytest.raises(TypeError, match="must return a tensor"):
        run_native_encoder(native_tower, torch.zeros(1, 128, 50))
    assert [dict(layer.self_attn._forward_pre_hooks) for layer in native_tower.layers] == before


def test_wrapper_rejects_multiple_batches(encoder_wrapper):
    with pytest.raises(AssertionError, match="Expected batch=1"):
        encoder_wrapper(torch.zeros(2, 128, 50))


def test_onnx_rejects_multiple_batches(dynamic_onnx):
    with pytest.raises(InvalidArgument, match="Got invalid dimensions"):
        dynamic_onnx.run(["audio_features"], {"mel": np.zeros((2, 128, 50), dtype=np.float32)})
