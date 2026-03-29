"""
4-path ASR comparison tests: native, wrapper PyTorch, FP32 ONNX, INT8 ONNX.

Reuses inference functions from compare.py. Parametrized over LibriSpeech
test fixtures of varying duration.

Requires:
    - Qwen3-ASR-0.6B model accessible (downloads on first run)
    - FP32 ONNX files in output/qwen3-asr-0.6b/
    - Audio fixtures in tests/fixtures/
"""

import json
import os
import re

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import AutoTokenizer

from compare import (
    load_onnx_sessions,
    run_native,
    run_onnx,
    run_wrapper_pytorch,
    strip_asr_prefix,
)

# ---------------------------------------------------------------------------
# Sample definitions
# ---------------------------------------------------------------------------

SAMPLES = [
    {
        "name": "librispeech_0",
        "path": "tests/fixtures/librispeech_0.wav",
        "ground_truth": "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL",
        "category": "short",
    },
    {
        "name": "librispeech_1",
        "path": "tests/fixtures/librispeech_1.wav",
        "ground_truth": "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER",
        "category": "short",
    },
    {
        "name": "librispeech_2",
        "path": "tests/fixtures/librispeech_2.wav",
        "ground_truth": "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND",
        "category": "short",
    },
    {
        "name": "librispeech_30s",
        "path": "tests/fixtures/librispeech_30s.wav",
        "ground_truth": "LAW SEEMED TO HIM WELL ENOUGH AS A SCIENCE BUT HE NEVER COULD DISCOVER A PRACTICAL CASE WHERE IT APPEARED TO HIM WORTH WHILE TO GO TO LAW AND ALL THE CLIENTS WHO STOPPED WITH THIS NEW CLERK IN THE ANTE ROOM OF THE LAW OFFICE WHERE HE WAS WRITING PHILIP INVARIABLY ADVISED TO SETTLE NO MATTER HOW BUT SETTLE GREATLY TO THE DISGUST OF HIS EMPLOYER WHO KNEW THAT JUSTICE BETWEEN MAN AND MAN COULD ONLY BE ATTAINED BY THE RECOGNIZED PROCESSES WITH THE ATTENDANT FEES",
        "category": "long",
    },
    {
        "name": "librispeech_32s",
        "path": "tests/fixtures/librispeech_32s.wav",
        "ground_truth": "BUT THERE IS ALWAYS A STRONGER SENSE OF LIFE WHEN THE SUN IS BRILLIANT AFTER RAIN AND NOW HE IS POURING DOWN HIS BEAMS AND MAKING SPARKLES AMONG THE WET STRAW AND LIGHTING UP EVERY PATCH OF VIVID GREEN MOSS ON THE RED TILES OF THE COW SHED AND TURNING EVEN THE MUDDY WATER THAT IS HURRYING ALONG THE CHANNEL TO THE DRAIN INTO A MIRROR FOR THE YELLOW BILLED DUCKS WHO ARE SEIZING THE OPPORTUNITY OF GETTING A DRINK WITH AS MUCH BODY IN IT AS POSSIBLE",
        "category": "long",
    },
    {
        "name": "librispeech_35s",
        "path": "tests/fixtures/librispeech_35s.wav",
        "ground_truth": "YESTERDAY YOU WERE TREMBLING FOR A HEALTH THAT IS DEAR TO YOU TO DAY YOU FEAR FOR YOUR OWN TO MORROW IT WILL BE ANXIETY ABOUT MONEY THE DAY AFTER TO MORROW THE DIATRIBE OF A SLANDERER THE DAY AFTER THAT THE MISFORTUNE OF SOME FRIEND THEN THE PREVAILING WEATHER THEN SOMETHING THAT HAS BEEN BROKEN OR LOST THEN A PLEASURE WITH WHICH YOUR CONSCIENCE AND YOUR VERTEBRAL COLUMN REPROACH YOU AGAIN THE COURSE OF PUBLIC AFFAIRS",
        "category": "long",
    },
]

FP32_DIR = "output/qwen3-asr-0.6b"
INT8_DIR = "output/qwen3-asr-0.6b-int8"

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_fp32_available = all(
    os.path.exists(os.path.join(FP32_DIR, f"{n}.onnx")) for n in ["encoder", "decoder_init", "decoder_step"]
)
_fixtures_available = all(os.path.exists(s["path"]) for s in SAMPLES)

pytestmark = pytest.mark.skipif(
    not (_fp32_available and _fixtures_available),
    reason="FP32 ONNX files or audio fixtures not found",
)

_int8_available = all(
    os.path.exists(os.path.join(INT8_DIR, f"{n}.onnx")) for n in ["encoder", "decoder_init", "decoder_step"]
)

slow = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures (module-scoped, loaded once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pytorch_model():
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
    except Exception:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )

        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    model.eval()
    return model


@pytest.fixture(scope="module")
def processor():
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import (
        Qwen3ASRProcessor,
    )

    return Qwen3ASRProcessor.from_pretrained("Qwen/Qwen3-ASR-0.6B")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def fp32_sessions():
    return load_onnx_sessions(FP32_DIR)


@pytest.fixture(scope="module")
def int8_sessions():
    if not _int8_available:
        pytest.skip("INT8 ONNX files not found")
    return load_onnx_sessions(INT8_DIR)


def _load_embed(onnx_dir):
    with open(os.path.join(onnx_dir, "config.json")) as f:
        cfg = json.load(f)
    dtype = np.dtype(cfg.get("embed_tokens_dtype", "float32"))
    embed = np.fromfile(
        os.path.join(onnx_dir, "embed_tokens.bin"),
        dtype=dtype,
    ).reshape(cfg["embed_tokens_shape"])
    return embed.astype(np.float32)


@pytest.fixture(scope="module")
def fp32_embed():
    return _load_embed(FP32_DIR)


@pytest.fixture(scope="module")
def int8_embed():
    if not _int8_available:
        pytest.skip("INT8 ONNX files not found")
    return _load_embed(INT8_DIR)


# ---------------------------------------------------------------------------
# Result cache — run each (sample, path) combination once
# ---------------------------------------------------------------------------

_result_cache: dict[str, dict] = {}


def _load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def _get_results(
    sample, pytorch_model, processor, tokenizer, fp32_sessions, fp32_embed, int8_sessions=None, int8_embed=None
):
    """Run all inference paths for a sample, caching results."""
    name = sample["name"]
    if name in _result_cache:
        return _result_cache[name]

    audio = _load_audio(sample["path"])
    results = {}

    results["native"] = run_native(pytorch_model, processor, audio)
    results["wrapper"] = run_wrapper_pytorch(pytorch_model, audio, tokenizer)
    results["fp32"] = run_onnx(fp32_sessions, fp32_embed, audio, tokenizer, "FP32")

    if int8_sessions is not None and int8_embed is not None:
        results["int8"] = run_onnx(int8_sessions, int8_embed, audio, tokenizer, "INT8")

    _result_cache[name] = results
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_words(text):
    """Normalize text to uppercase words only (strip punctuation)."""
    text = strip_asr_prefix(text)
    return re.sub(r"[^A-Za-z\s']", "", text).upper().split()


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

_short_samples = [s for s in SAMPLES if s["category"] == "short"]
_long_samples = [s for s in SAMPLES if s["category"] == "long"]

_sample_ids = [s["name"] for s in SAMPLES]
_short_ids = [s["name"] for s in _short_samples]
_long_ids = [s["name"] for s in _long_samples]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTokenCounts:
    @pytest.mark.parametrize("sample", SAMPLES, ids=_sample_ids)
    def test_audio_token_counts_consistent(
        self,
        sample,
        pytorch_model,
        processor,
        tokenizer,
        fp32_sessions,
        fp32_embed,
    ):
        """Wrapper, FP32 (and INT8 if available) produce the same audio_token_count."""
        results = _get_results(
            sample,
            pytorch_model,
            processor,
            tokenizer,
            fp32_sessions,
            fp32_embed,
        )
        wrapper_count = results["wrapper"]["audio_token_count"]
        fp32_count = results["fp32"]["audio_token_count"]
        assert wrapper_count == fp32_count, f"wrapper={wrapper_count} vs fp32={fp32_count}"

        if "int8" in results:
            int8_count = results["int8"]["audio_token_count"]
            assert wrapper_count == int8_count, f"wrapper={wrapper_count} vs int8={int8_count}"


class TestWrapperFP32:
    @pytest.mark.parametrize("sample", SAMPLES, ids=_sample_ids)
    def test_tokens_exact(
        self,
        sample,
        pytorch_model,
        processor,
        tokenizer,
        fp32_sessions,
        fp32_embed,
    ):
        """Wrapper and FP32 ONNX should produce identical generated tokens."""
        results = _get_results(
            sample,
            pytorch_model,
            processor,
            tokenizer,
            fp32_sessions,
            fp32_embed,
        )
        assert results["wrapper"]["tokens"] == results["fp32"]["tokens"], (
            f"wrapper tokens != fp32 tokens for {sample['name']}"
        )

    @pytest.mark.parametrize("sample", SAMPLES, ids=_sample_ids)
    def test_encoder_feature_tolerance(
        self,
        sample,
        pytorch_model,
        processor,
        tokenizer,
        fp32_sessions,
        fp32_embed,
    ):
        """Wrapper and FP32 encoder features should be within 1e-4."""
        results = _get_results(
            sample,
            pytorch_model,
            processor,
            tokenizer,
            fp32_sessions,
            fp32_embed,
        )
        wrapper_feat = results["wrapper"]["audio_features"]
        fp32_feat = results["fp32"]["audio_features"]
        assert wrapper_feat.shape == fp32_feat.shape
        max_diff = np.max(np.abs(wrapper_feat - fp32_feat))
        assert max_diff < 1e-4, f"max_diff={max_diff:.6e} exceeds 1e-4"


class TestNative:
    @pytest.mark.parametrize("sample", _short_samples, ids=_short_ids)
    def test_matches_fp32_short(
        self,
        sample,
        pytorch_model,
        processor,
        tokenizer,
        fp32_sessions,
        fp32_embed,
    ):
        """On short audio, native and FP32 should produce exact token match."""
        results = _get_results(
            sample,
            pytorch_model,
            processor,
            tokenizer,
            fp32_sessions,
            fp32_embed,
        )
        assert results["native"]["tokens"] == results["fp32"]["tokens"], (
            f"native tokens != fp32 tokens for {sample['name']}"
        )

    @slow
    @pytest.mark.parametrize("sample", _long_samples, ids=_long_ids)
    def test_matches_fp32_long_words(
        self,
        sample,
        pytorch_model,
        processor,
        tokenizer,
        fp32_sessions,
        fp32_embed,
    ):
        """On long audio, native and FP32 should produce nearly identical words.

        SDPA vs eager attention causes numerical divergence on longer sequences,
        so we allow up to 5% word error rate rather than requiring exact equality.
        """
        results = _get_results(
            sample,
            pytorch_model,
            processor,
            tokenizer,
            fp32_sessions,
            fp32_embed,
        )
        native_words = _extract_words(results["native"]["text"])
        fp32_words = _extract_words(results["fp32"]["text"])
        n = len(native_words)
        mismatches = sum(1 for a, b in zip(native_words, fp32_words) if a != b)
        mismatches += abs(len(native_words) - len(fp32_words))
        wer = mismatches / n if n > 0 else 0.0
        assert wer <= 0.05, (
            f"native vs fp32 WER {wer:.1%} exceeds 5% for {sample['name']}: "
            f"{' '.join(native_words)} vs {' '.join(fp32_words)}"
        )


class TestINT8:
    @pytest.mark.parametrize("sample", _short_samples, ids=_short_ids)
    def test_short_matches_fp32(
        self,
        sample,
        pytorch_model,
        processor,
        tokenizer,
        fp32_sessions,
        fp32_embed,
        int8_sessions,
        int8_embed,
    ):
        """On short audio, INT8 should produce identical tokens to FP32."""
        results = _get_results(
            sample,
            pytorch_model,
            processor,
            tokenizer,
            fp32_sessions,
            fp32_embed,
            int8_sessions,
            int8_embed,
        )
        if "int8" not in results:
            pytest.skip("INT8 results not available")
        assert results["int8"]["tokens"] == results["fp32"]["tokens"], (
            f"int8 tokens != fp32 tokens for {sample['name']}"
        )
