"""
End-to-end pipeline test: audio -> text, comparing ONNX vs PyTorch.

Requires:
    - All ONNX files exported
    - Test audio file
    - Qwen3-ASR-0.6B model accessible
"""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer

from src.encoder_wrapper import EncoderWrapper
from src.inference import greedy_decode_onnx
from src.mel import log_mel_spectrogram
from src.prompt import build_prompt_ids, get_audio_pad_range, EOS_TOKEN_IDS

OUTPUT_DIR = "output/qwen3-asr-0.6b"
AUDIO_PATH = "tests/fixtures/test_audio.wav"

pytestmark = pytest.mark.skipif(
    not (
        os.path.exists(os.path.join(OUTPUT_DIR, "encoder.onnx"))
        and os.path.exists(os.path.join(OUTPUT_DIR, "decoder_init.onnx"))
        and os.path.exists(os.path.join(OUTPUT_DIR, "decoder_step.onnx"))
        and os.path.exists(AUDIO_PATH)
    ),
    reason="ONNX files or test audio not found",
)


@pytest.fixture(scope="module")
def audio():
    """Load test audio as float32 numpy array at 16kHz."""
    data, sr = sf.read(AUDIO_PATH, dtype="float32")
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


@pytest.fixture(scope="module")
def mel(audio):
    """Compute mel spectrogram from test audio."""
    return log_mel_spectrogram(audio)


@pytest.fixture(scope="module")
def pytorch_model():
    model = AutoModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B", trust_remote_code=True
    )


@pytest.fixture(scope="module")
def onnx_sessions():
    sessions = {}
    providers = ["CPUExecutionProvider"]
    for name in ["encoder", "decoder_init", "decoder_step"]:
        sessions[name] = ort.InferenceSession(
            os.path.join(OUTPUT_DIR, f"{name}.onnx"),
            providers=providers,
        )
    return sessions


@pytest.fixture(scope="module")
def embed_tokens():
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    shape = config["embed_tokens_shape"]
    return np.fromfile(
        os.path.join(OUTPUT_DIR, "embed_tokens.bin"),
        dtype=np.float32,
    ).reshape(shape)


def decode_pytorch(model, audio_features, prompt_ids, max_tokens=256):
    """Greedy decode using PyTorch model."""
    with torch.no_grad():
        embed_weight = model.thinker.model.embed_tokens.weight.data
        ids_tensor = torch.tensor(prompt_ids, dtype=torch.long)
        input_embeds = embed_weight[ids_tensor].clone()

        audio_start, audio_end = get_audio_pad_range(prompt_ids)
        input_embeds[audio_start:audio_end] = audio_features[0]

        input_embeds = input_embeds.unsqueeze(0)
        seq_len = len(prompt_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)
        cache_position = torch.arange(seq_len)

        outputs = model.thinker.model(
            inputs_embeds=input_embeds,
            position_ids=pos_3d,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        logits = model.thinker.lm_head(outputs.last_hidden_state)
        past_kv = outputs.past_key_values

        next_token = int(logits[0, -1, :].argmax())
        tokens = [next_token]

        pos = len(prompt_ids)
        for _ in range(max_tokens - 1):
            if next_token in EOS_TOKEN_IDS:
                break

            token_embed = embed_weight[next_token].unsqueeze(0).unsqueeze(0)
            step_pos = torch.tensor([[pos]], dtype=torch.long)
            step_pos_3d = step_pos.unsqueeze(0).expand(3, -1, -1)
            step_cache_pos = torch.tensor([pos])

            outputs = model.thinker.model(
                inputs_embeds=token_embed,
                position_ids=step_pos_3d,
                past_key_values=past_kv,
                cache_position=step_cache_pos,
                use_cache=True,
                return_dict=True,
            )
            logits = model.thinker.lm_head(outputs.last_hidden_state)
            past_kv = outputs.past_key_values

            next_token = int(logits[0, -1, :].argmax())
            tokens.append(next_token)
            pos += 1

    return tokens


class TestPipeline:
    def test_encoder_produces_features(self, onnx_sessions, mel):
        """Encoder should produce non-zero features."""
        features = onnx_sessions["encoder"].run(
            ["audio_features"], {"mel": mel.numpy()}
        )[0]
        assert features.shape[0] == 1
        assert features.shape[2] == 1024
        assert not np.allclose(features, 0), "Encoder produced all zeros"

    def test_transcription_matches(
        self, pytorch_model, onnx_sessions, embed_tokens, mel, tokenizer
    ):
        """ONNX transcription should match PyTorch transcription."""
        mel_np = mel.numpy()

        # Get audio features from both (using the same wrapper as ONNX export)
        wrapper = EncoderWrapper(pytorch_model.thinker.audio_tower).eval()
        with torch.no_grad():
            pt_features = wrapper(mel)
        onnx_features = onnx_sessions["encoder"].run(
            ["audio_features"], {"mel": mel_np}
        )[0]

        audio_token_count = pt_features.shape[1]
        prompt_ids = build_prompt_ids(audio_token_count)

        # Decode both
        pt_tokens = decode_pytorch(pytorch_model, pt_features, prompt_ids)
        onnx_tokens = greedy_decode_onnx(onnx_sessions, embed_tokens, onnx_features, prompt_ids)

        pt_text = tokenizer.decode(pt_tokens, skip_special_tokens=True)
        onnx_text = tokenizer.decode(onnx_tokens, skip_special_tokens=True)

        print(f"\nPyTorch: {pt_text}")
        print(f"ONNX:    {onnx_text}")

        # Token-level comparison
        match = sum(1 for a, b in zip(pt_tokens, onnx_tokens) if a == b)
        total = max(len(pt_tokens), len(onnx_tokens))
        match_rate = match / total if total > 0 else 1.0

        assert match_rate >= 0.95, (
            f"Token match rate {match_rate:.1%} below 95% threshold. "
            f"PT: {pt_text!r}, ONNX: {onnx_text!r}"
        )

    def test_greedy_deterministic(self, onnx_sessions, embed_tokens, mel):
        """Running ONNX decode twice should produce identical output."""
        mel_np = mel.numpy()
        features = onnx_sessions["encoder"].run(
            ["audio_features"], {"mel": mel_np}
        )[0]

        audio_token_count = features.shape[1]
        prompt_ids = build_prompt_ids(audio_token_count)

        tokens_1 = greedy_decode_onnx(onnx_sessions, embed_tokens, features, prompt_ids)
        tokens_2 = greedy_decode_onnx(onnx_sessions, embed_tokens, features, prompt_ids)

        assert tokens_1 == tokens_2, "Greedy decode not deterministic"
