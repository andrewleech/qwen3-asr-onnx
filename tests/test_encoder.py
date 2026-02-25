"""
Test encoder ONNX export correctness against PyTorch reference.

Requires:
    - Qwen3-ASR-0.6B model accessible (downloads on first run)
    - encoder.onnx in output directory
    - Test audio in tests/fixtures/test_audio.wav
"""

import os

import numpy as np
import onnxruntime as ort
import pytest
import torch

# These tests require the model and ONNX files to exist
pytestmark = pytest.mark.skipif(
    not os.path.exists("output/qwen3-asr-0.6b/encoder.onnx"),
    reason="encoder.onnx not found - run export.py first",
)


@pytest.fixture(scope="module")
def pytorch_model():
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def onnx_session():
    return ort.InferenceSession(
        "output/qwen3-asr-0.6b/encoder.onnx",
        providers=["CPUExecutionProvider"],
    )


@pytest.fixture(scope="module")
def test_mel():
    """Generate a test mel spectrogram from fixture audio or random data."""
    audio_path = "tests/fixtures/test_audio.wav"
    if os.path.exists(audio_path):
        import soundfile as sf
        from src.mel import log_mel_spectrogram

        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return log_mel_spectrogram(audio)
    else:
        # Random mel for shape/dtype testing
        return torch.randn(1, 128, 1600)


class TestEncoderExport:
    def test_output_shape(self, onnx_session, test_mel):
        """Encoder output shape should be [batch, time/8, 1024]."""
        mel_np = test_mel.numpy()
        result = onnx_session.run(["audio_features"], {"mel": mel_np})
        features = result[0]

        assert features.ndim == 3
        assert features.shape[0] == 1  # batch
        assert features.shape[2] == 1024  # output_dim

    def test_pytorch_onnx_match(self, pytorch_model, onnx_session, test_mel):
        """ONNX output should match PyTorch output within tolerance."""
        from src.encoder_wrapper import EncoderWrapper

        mel_np = test_mel.numpy()

        # PyTorch reference (using the same wrapper as ONNX export)
        wrapper = EncoderWrapper(pytorch_model.thinker.audio_tower).eval()
        with torch.no_grad():
            pt_output = wrapper(test_mel).numpy()

        # ONNX
        onnx_output = onnx_session.run(["audio_features"], {"mel": mel_np})[0]

        assert pt_output.shape == onnx_output.shape, (
            f"Shape mismatch: PT={pt_output.shape} ONNX={onnx_output.shape}"
        )

        max_diff = np.max(np.abs(pt_output - onnx_output))
        assert max_diff < 1e-4, f"Max diff {max_diff:.6e} exceeds threshold 1e-4"

    def test_dynamic_time_axis(self, onnx_session):
        """Encoder should accept variable-length mel inputs."""
        for n_frames in [800, 1600, 3000]:
            mel = np.random.randn(1, 128, n_frames).astype(np.float32)
            result = onnx_session.run(["audio_features"], {"mel": mel})
            features = result[0]
            assert features.shape[0] == 1
            assert features.shape[2] == 1024

    def test_single_batch(self, onnx_session):
        """Encoder processes batch=1 (original model doesn't support batch>1)."""
        mel = np.random.randn(1, 128, 1600).astype(np.float32)
        result = onnx_session.run(["audio_features"], {"mel": mel})
        features = result[0]
        assert features.shape[0] == 1
        assert features.shape[2] == 1024
