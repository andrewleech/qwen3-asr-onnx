import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import validate


@pytest.mark.parametrize(
    "actual", [np.ones((1, 2, 3)), np.full((1, 2, 3), np.nan), np.full((1, 2, 3), np.inf), np.zeros((1, 1, 3))]
)
def test_features_fail_closed(actual):
    with pytest.raises(ValueError):
        validate.compare_features(np.zeros((1, 2, 3)), actual, "probe")


def test_feature_tolerance_is_enforced():
    reference = np.full((1, 2, 3), 2.0, dtype=np.float32)
    validate.compare_features(reference, reference + 1e-5, "probe", atol=2e-5, rtol=0)
    with pytest.raises(ValueError, match="tolerance|exceed"):
        validate.compare_features(reference, reference + 1e-5, "probe", atol=1e-6, rtol=0)


def test_sessions_require_all_graphs(tmp_path):
    with pytest.raises(FileNotFoundError, match="encoder.onnx.*decoder_init.onnx.*decoder_step.onnx"):
        validate.load_onnx_sessions(str(tmp_path))


def test_pipeline_max_tokens_and_transcript_failure(monkeypatch):
    counts = []
    monkeypatch.setattr(validate, "build_prompt_ids", lambda count, tokenizer: [1, 2, 3])

    def pytorch(*args, **kwargs):
        counts.append(kwargs["max_tokens"])
        return [10]

    def onnx(*args, **kwargs):
        counts.append(kwargs["max_tokens"])
        return [11]

    monkeypatch.setattr(validate, "greedy_decode_pytorch", pytorch)
    monkeypatch.setattr(validate, "greedy_decode_onnx", onnx)
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens))
    with pytest.raises(ValueError, match="Transcript mismatch"):
        validate.validate_pipeline(
            None, {}, None, torch.zeros(1, 1, 3), np.zeros((1, 1, 3)), tokenizer, "cpu", max_tokens=7
        )
    assert counts == [7, 7]


@pytest.mark.parametrize(
    "fault", ["missing", "nan", "shape", "tolerance", "transcript", "decoder_nan", "native_tolerance"]
)
def test_cli_returns_nonzero_for_validation_fault(tmp_path, fault):
    script = tmp_path / "fault.py"
    script.write_text("""
import sys
from types import SimpleNamespace
import numpy as np
import torch
import validate
fault=sys.argv[1]
sys.argv=['validate.py','--onnx-dir','unused','--audio','unused','--max-tokens','7']
validate.sf.read=lambda *a,**k:(np.zeros(1600,dtype=np.float32),16000)
validate.log_mel_spectrogram=lambda *a,**k:torch.zeros(1,128,10)
validate.load_pytorch_model=lambda *a,**k:SimpleNamespace(thinker=SimpleNamespace(audio_tower=None))
if fault!='missing':
    validate.load_onnx_sessions=lambda *a,**k:{'encoder':object(),'decoder_init':object(),'decoder_step':object()}
validate.load_embed_tokens=lambda *a,**k:np.zeros((20,3),dtype=np.float32)
tokenizer=SimpleNamespace(decode=lambda x,**k:str(x))
validate.AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:tokenizer)
validate.load_prompt_tokenizer=lambda *a,**k:tokenizer
validate.validate_tokenizer_assets=lambda *a,**k:tokenizer
validate.build_prompt_ids=lambda *a,**k:[1,2,3]
validate.run_encoder_pytorch=lambda *a,**k:torch.zeros(1,1,3)
try:
    import src.native_encoder
    src.native_encoder.run_native_encoder=lambda *a,**k:torch.zeros(1,1,3)
except ModuleNotFoundError:
    pass
output=np.zeros((1,1,3),dtype=np.float32)
if fault=='nan': output[:]=np.nan
if fault=='shape': output=np.zeros((1,2,3),dtype=np.float32)
if fault=='tolerance': output[:]=1
if fault=='native_tolerance':
    validate.run_encoder_pytorch=lambda *a,**k:torch.full((1,1,3),0.000075)
    output[:]=0.000150
validate.run_encoder_onnx=lambda *a,**k:output
validate.greedy_decode_pytorch=lambda *a,**k:[10]
def decode_onnx(*a,**k):
    if fault=='decoder_nan':
        session=validate.FiniteOutputSession(SimpleNamespace(run=lambda *a:[np.array([[[np.nan,0.]]],dtype=np.float32)]),'decoder_init')
        session.run(['logits'],{})
    return [11] if fault=='transcript' else [10]
validate.greedy_decode_onnx=decode_onnx
validate.main()
""")
    env = os.environ | {
        "PYTHONPATH": str(Path(validate.__file__).parent),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = subprocess.run([sys.executable, str(script), fault], capture_output=True, text=True, env=env, timeout=45)
    assert result.returncode != 0, result.stdout + result.stderr
    expected = {
        "missing": "Missing required ONNX sessions",
        "nan": "non-finite",
        "shape": "shape mismatch",
        "tolerance": "exceed",
        "transcript": "Transcript mismatch",
        "decoder_nan": "non-finite ONNX output",
        "native_tolerance": "native vs ONNX: encoder features exceed",
    }[fault]
    assert expected in result.stdout + result.stderr


def test_cli_rejects_invalid_max_tokens(tmp_path):
    result = subprocess.run(
        [sys.executable, validate.__file__, "--onnx-dir", str(tmp_path), "--audio", "unused", "--max-tokens", "0"],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode != 0
    assert "--max-tokens must be positive" in result.stderr


def test_session_adapter_rejects_nan_logits_even_when_argmax_is_unchanged():
    logits = np.array([[[np.nan, 0.0]]], dtype=np.float32)
    assert int(logits.argmax()) == 0
    raw = SimpleNamespace(run=lambda outputs, inputs: [logits], get_inputs=lambda: ["input"])
    session = validate.FiniteOutputSession(raw, "decoder_init")
    assert session.get_inputs() == ["input"]
    with pytest.raises(ValueError, match="non-finite ONNX output"):
        session.run(["logits"], {"input": np.zeros(1)})


def test_session_adapter_forwards_arguments_and_values():
    inputs = {"x": np.zeros(1)}
    values = [np.ones((1, 2)), np.zeros((1, 1))]
    calls = []

    def run(names, feed):
        calls.append((names, feed))
        return values

    session = validate.FiniteOutputSession(SimpleNamespace(run=run), "decoder")
    assert session.run(["logits", "cache"], inputs) is values
    assert calls == [(["logits", "cache"], inputs)]


def test_pytorch_decode_rejects_nan_logits(monkeypatch):
    class TextModel:
        embed_tokens = SimpleNamespace(weight=torch.zeros(3, 1))

        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.zeros(1, 1, 1), past_key_values=None)

    thinker = SimpleNamespace(model=TextModel(), lm_head=lambda hidden: torch.full((1, 1, 3), float("nan")))
    model = SimpleNamespace(thinker=thinker)
    monkeypatch.setattr(validate, "get_audio_pad_range", lambda ids: (0, 1))
    with pytest.raises(ValueError, match="Non-finite PyTorch decoder logits"):
        validate.greedy_decode_pytorch(model, torch.zeros(1, 1, 1), [1], max_tokens=1)


def test_real_cli_missing_sessions_fails_before_checkpoint_load(tmp_path):
    env = os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, validate.__file__, "--onnx-dir", str(tmp_path), "--audio", "unused"],
        capture_output=True,
        text=True,
        timeout=45,
        env=env,
    )
    assert result.returncode != 0
    assert "Missing required ONNX sessions" in result.stderr


def test_validator_numerically_compares_native_wrapper_and_onnx(tmp_path, capsys):
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

    from src.encoder_wrapper import export_encoder

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
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
            model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=Qwen3ASRAudioEncoder(config).eval()))
            path = tmp_path / "encoder.onnx"
            export_encoder(model, str(path))
            options = validate.ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = validate.ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            for frames in (50, 801):
                mel = torch.randn(1, 128, frames)
                reference, actual = validate.validate_encoder(model, {"encoder": session}, mel.numpy(), mel, "cpu")
                assert reference.shape == actual.shape
        output = capsys.readouterr().out
        assert "PASS: native vs wrapper" in output
        assert "PASS: wrapper vs ONNX" in output
    finally:
        torch.set_num_threads(previous_threads)


def test_encoder_returns_native_features_for_transcript_reference(monkeypatch):
    import src.native_encoder

    native = torch.zeros(1, 1, 3)
    wrapper = native + 1e-5
    monkeypatch.setattr(src.native_encoder, "run_native_encoder", lambda *args: native)
    monkeypatch.setattr(validate, "run_encoder_pytorch", lambda *args: wrapper)
    monkeypatch.setattr(validate, "run_encoder_onnx", lambda *args: wrapper.numpy())
    model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=None))
    reference, actual = validate.validate_encoder(model, {"encoder": object()}, None, None, "cpu")
    np.testing.assert_array_equal(reference, native.numpy())
    np.testing.assert_array_equal(actual, wrapper.numpy())


def test_encoder_default_tolerance_is_absolute():
    reference = np.full((1, 1, 3), 100000.0, dtype=np.float32)
    with pytest.raises(ValueError, match="exceed"):
        validate.compare_features(reference, reference + 0.5, "probe")


def test_direct_native_tolerance_is_not_inferred_from_pairwise_passes(monkeypatch):
    import src.native_encoder

    native = torch.zeros(1, 1, 3)
    wrapper = native + 0.000075
    exported = np.full((1, 1, 3), 0.000150, dtype=np.float32)
    monkeypatch.setattr(src.native_encoder, "run_native_encoder", lambda *a: native)
    monkeypatch.setattr(validate, "run_encoder_pytorch", lambda *a: wrapper)
    monkeypatch.setattr(validate, "run_encoder_onnx", lambda *a: exported)
    model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=None))
    with pytest.raises(ValueError, match="native vs ONNX: encoder features exceed"):
        validate.validate_encoder(model, {"encoder": object()}, None, None, "cpu")


@pytest.mark.parametrize("problem", ["missing", "corrupt", "conflicting-template"])
def test_main_rejects_invalid_artifact_family_before_audio(tmp_path, monkeypatch, problem):
    import json

    from tests.tokenizer_fixtures import save_tokenizer_assets

    save_tokenizer_assets(tmp_path)
    if problem == "missing":
        (tmp_path / "tokenizer.json").unlink()
    elif problem == "corrupt":
        (tmp_path / "tokenizer.json").write_text("invalid JSON")
    else:
        (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "WRONG PROMPT"}))
    monkeypatch.setattr(validate, "load_onnx_sessions", lambda path: {})
    monkeypatch.setattr(sys, "argv", ["validate.py", "--onnx-dir", str(tmp_path), "--audio", "unused"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Artifact validation must precede audio and model loading")

    monkeypatch.setattr(validate.sf, "read", forbidden)
    monkeypatch.setattr(validate, "load_pytorch_model", forbidden)
    with pytest.raises(ValueError, match="tokenizer|ASR"):
        validate.main()


def artifact_pair(root, *, role_id=None, word_id=86):
    import json

    from src.tokenizer_assets import validate_tokenizer_assets
    from tests.tokenizer_fixtures import save_tokenizer_assets

    checkpoint, artifact = root / "checkpoint", root / "artifact"
    native = save_tokenizer_assets(checkpoint)
    save_tokenizer_assets(artifact)
    path = artifact / "tokenizer.json"
    data = json.loads(path.read_text())
    vocab = data["model"]["vocab"]
    if role_id is not None:
        vocab["system"] = role_id
    if word_id != 86:
        vocab["Kepler"] = word_id
        vocab["Different"] = 86
    path.write_text(json.dumps(data))
    return native, validate_tokenizer_assets(artifact)


def test_pipeline_rejects_different_artifact_prompt_before_decoding(tmp_path, monkeypatch):
    native, artifact = artifact_pair(tmp_path, role_id=99)

    def forbidden(*args, **kwargs):
        raise AssertionError("Prompt identity must be checked before decoding")

    monkeypatch.setattr(validate, "greedy_decode_pytorch", forbidden)
    monkeypatch.setattr(validate, "greedy_decode_onnx", forbidden)
    with pytest.raises(ValueError, match="Artifact prompt IDs"):
        validate.validate_pipeline(
            None, {}, None, torch.zeros(1, 1, 3), np.zeros((1, 1, 3)), native, "cpu", onnx_tokenizer=artifact
        )


@pytest.mark.parametrize("artifact_token", [86, 99])
def test_pipeline_compares_decoded_text_with_each_actual_tokenizer(tmp_path, monkeypatch, artifact_token, capsys):
    native, artifact = artifact_pair(tmp_path, word_id=99)
    monkeypatch.setattr(validate, "greedy_decode_pytorch", lambda *a, **k: [86])
    monkeypatch.setattr(validate, "greedy_decode_onnx", lambda *a, **k: [artifact_token])
    args = (None, {}, None, torch.zeros(1, 1, 3), np.zeros((1, 1, 3)), native, "cpu")
    if artifact_token == 86:
        with pytest.raises(ValueError, match="Transcript mismatch"):
            validate.validate_pipeline(*args, onnx_tokenizer=artifact)
    else:
        validate.validate_pipeline(*args, onnx_tokenizer=artifact)
        assert "PASS: Text matches" in capsys.readouterr().out


@pytest.mark.parametrize("value,accepted", [(1.11, False), (0.905, True)])
def test_relative_tolerance_scales_reference_features(value, accepted):
    reference = np.ones((1, 2, 3), dtype=np.float64)
    actual = np.full(reference.shape, value)
    if accepted:
        validate.compare_features(reference, actual, "relative tolerance", atol=0, rtol=0.1)
    else:
        with pytest.raises(ValueError, match="exceed"):
            validate.compare_features(reference, actual, "relative tolerance", atol=0, rtol=0.1)
