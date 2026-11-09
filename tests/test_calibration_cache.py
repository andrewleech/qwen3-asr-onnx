from pathlib import Path

import numpy as np
import pytest
import torch

from quantize_static import calibration_identity
from src import artifact_publication
from src.calibration_cache import (
    cache_key,
    load_cache,
    model_fingerprint,
    pack_samples,
    recipe,
    save_cache,
    tokenizer_fingerprint,
    unpack_samples,
)
from tests.artifact_fixtures import _release_input, _write_model


def _identity(tmp_path):
    producer = tmp_path / "collector.py"
    producer.write_text("collector-v1")
    return recipe("test", {"checkpoint": "fixture"}, {"n_samples": 1}, producer)


def test_numeric_cache_roundtrip_and_provenance(tmp_path):
    identity = _identity(tmp_path)
    path = tmp_path / "cache.npz"
    samples = {"init": [{"x": np.arange(4, dtype=np.float32).reshape(1, 4)}], "step": []}
    save_cache(path, identity, pack_samples(samples))
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    restored = unpack_samples(load_cache(path, identity), ("init", "step"))
    np.testing.assert_array_equal(restored["init"][0]["x"], samples["init"][0]["x"])
    assert restored["step"] == []
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unverifiable calibration cache"):
        load_cache(path, identity | {"parameters": {"n_samples": 2}})
    assert path.read_bytes() == before


def test_legacy_object_cache_is_preserved_and_not_loaded(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, init_inputs=np.array([{"x": np.ones(1)}], dtype=object))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unverifiable"):
        load_cache(path, _identity(tmp_path))
    assert path.read_bytes() == before


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_checkpoint_identity_uses_actual_weights(tmp_path, dtype):
    model = torch.nn.Linear(4, 3).to(dtype)
    before = model_fingerprint(model)
    assert model_fingerprint(model) == before
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert model_fingerprint(model) != before


@pytest.mark.parametrize("artifact", ["encoder.onnx.data", "embed_tokens.bin", "chat_template.jinja", "config.json"])
def test_static_identity_binds_actual_inputs(tmp_path, artifact):
    source = _release_input(tmp_path / "source")
    _write_model(source, "encoder.onnx", external=True)
    identity = calibration_identity(source, 1, 2)
    path = source / artifact
    value = path.read_bytes()
    path.write_bytes(value + b" " if artifact.endswith(("jinja", "json")) else bytes([value[0] ^ 1]) + value[1:])
    assert cache_key(calibration_identity(source, 1, 2)) != cache_key(identity)


def test_cache_installation_failure_restores_previous_cache(tmp_path, monkeypatch):
    identity = _identity(tmp_path)
    path = tmp_path / "cache.npz"
    save_cache(path, identity, {"x": np.ones(1, dtype=np.float32)})
    before = path.read_bytes()
    replace = artifact_publication.os.replace
    failed = False

    def fail_once(source, target):
        nonlocal failed
        result = replace(source, target)
        if not failed and Path(source).name == "cache.npz" and Path(target) == path:
            failed = True
            raise OSError("cache install failure")
        return result

    monkeypatch.setattr(artifact_publication.os, "replace", fail_once)
    with pytest.raises(OSError, match="cache install failure"):
        save_cache(path, identity, {"x": np.zeros(1, dtype=np.float32)})
    assert failed
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [np.array([object()]), np.array([np.nan])])
def test_invalid_cache_arrays_cannot_replace_previous_file(tmp_path, value):
    path = tmp_path / "cache.npz"
    path.write_bytes(b"prior cache")
    with pytest.raises(ValueError, match="finite numeric"):
        save_cache(path, _identity(tmp_path), {"x": value})
    assert path.read_bytes() == b"prior cache"


def test_checkpoint_configuration_and_execution_mode_affect_identity():
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig

    model = torch.nn.Linear(4, 3).eval()
    model.config = Qwen3ASRConfig()
    before = model_fingerprint(model)
    model.config.thinker_config.text_config.rms_norm_eps *= 2
    assert model_fingerprint(model) != before
    before = model_fingerprint(model)
    model.train()
    assert model_fingerprint(model) != before


@pytest.mark.parametrize("count", [-1, 2**60])
def test_invalid_sample_count_rejected_before_iteration(count):
    with pytest.raises(ValueError, match="sample count"):
        unpack_samples({"init_count": np.array(count, dtype=np.int64)}, ("init",))


def test_tokenizer_backend_affects_cache_identity(tmp_path):
    from tokenizers import normalizers

    from src.prompt import build_prompt_ids, validate_prompt_tokenizer
    from tests.tokenizer_fixtures import save_tokenizer_assets

    tokenizer = save_tokenizer_assets(tmp_path)
    before = tokenizer_fingerprint(tokenizer)
    original_ids = build_prompt_ids(2, tokenizer)
    tokenizer.backend_tokenizer.normalizer = normalizers.Replace("system", "assistant")
    validate_prompt_tokenizer(tokenizer)
    assert build_prompt_ids(2, tokenizer) != original_ids
    assert tokenizer_fingerprint(tokenizer) != before


def test_dangling_cache_entry_is_preserved(tmp_path):
    path = tmp_path / "cache.npz"
    path.symlink_to(tmp_path / "missing.npz")
    with pytest.raises(ValueError, match="Unverifiable"):
        load_cache(path, _identity(tmp_path))
    assert path.is_symlink()
    assert not path.exists()
