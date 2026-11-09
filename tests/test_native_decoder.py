from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRTextConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerTextModel

from src.decoder_wrapper import export_decoder_init, export_decoder_step


@pytest.fixture(scope="module", autouse=True)
def decoder_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def text_model():
    with torch.random.fork_rng():
        torch.manual_seed(4342)
        config = Qwen3ASRTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        )
        config._attn_implementation = "eager"
        text = Qwen3ASRThinkerTextModel(config).eval()
        head = torch.nn.Linear(32, 64, bias=False).eval()
    return SimpleNamespace(
        config=SimpleNamespace(thinker_config=SimpleNamespace(text_config=config)),
        thinker=SimpleNamespace(model=text, lm_head=head),
    )


@pytest.fixture(scope="module")
def decoder_exports(text_model, tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-decoder")
    paths = {}
    for kind, exporter in [("init", export_decoder_init), ("step", export_decoder_step)]:
        path = directory / f"decoder_{kind}.onnx"
        exporter(text_model, str(path), opset_version=17)
        paths[kind] = path
    return paths


@pytest.fixture(scope="module")
def decoder_sessions(decoder_exports):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return {
        kind: ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        for kind, path in decoder_exports.items()
    }


@pytest.mark.parametrize("kind", ["init", "step"])
def test_decoder_export_respects_requested_opset(decoder_exports, kind):
    graph = onnx.load(str(decoder_exports[kind]), load_external_data=False)
    assert [(item.domain, item.version) for item in graph.opset_import] == [("", 17)]


def prefill_inputs(model, batch, seq_len, audio_len, offset):
    generator = torch.Generator().manual_seed(seq_len)
    ids = torch.randint(0, 64, (batch, seq_len), generator=generator)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    audio = torch.randn(batch, audio_len, 32, generator=generator)
    embeds = model.thinker.model.embed_tokens(ids).clone()
    embeds[:, offset : offset + audio_len] = audio
    return embeds, {
        "input_ids": ids.numpy(),
        "position_ids": positions.numpy(),
        "audio_features": audio.numpy(),
        "audio_offset": np.array([offset], dtype=np.int64),
    }


def native_outputs(model, output):
    cache = output.past_key_values.to_legacy_cache()
    return [
        model.thinker.lm_head(output.last_hidden_state).numpy(),
        torch.stack([layer[0] for layer in cache]).numpy(),
        torch.stack([layer[1] for layer in cache]).numpy(),
    ]


def assert_decoder_matches(actual, expected):
    for got, reference in zip(actual, expected):
        assert got.shape == reference.shape
        assert np.isfinite(got).all()
        assert np.isfinite(reference).all()
        np.testing.assert_allclose(got, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("batch,seq_len,audio_len,offset", [(1, 7, 1, 2), (1, 19, 5, 9), (2, 11, 3, 1)])
def test_decoder_prefill_matches_native(text_model, decoder_sessions, batch, seq_len, audio_len, offset):
    with torch.no_grad():
        embeds, feed = prefill_inputs(text_model, batch, seq_len, audio_len, offset)
        output = text_model.thinker.model(inputs_embeds=embeds, use_cache=True)
        expected = native_outputs(text_model, output)
    actual = decoder_sessions["init"].run(None, feed)
    assert_decoder_matches(actual, expected)


@pytest.mark.parametrize("batch,seq_len,audio_len,offset", [(1, 7, 1, 2), (1, 19, 5, 9), (2, 11, 3, 1)])
def test_decoder_step_matches_native_growing_cache(text_model, decoder_sessions, batch, seq_len, audio_len, offset):
    with torch.no_grad():
        embeds, feed = prefill_inputs(text_model, batch, seq_len, audio_len, offset)
        output = text_model.thinker.model(inputs_embeds=embeds, use_cache=True)
        actual = decoder_sessions["init"].run(None, feed)
        for step in range(2):
            token = torch.full((batch, 1), 3 + step, dtype=torch.long)
            token_embeds = text_model.thinker.model.embed_tokens(token)
            positions = torch.full((batch, 1), seq_len + step, dtype=torch.long)
            output = text_model.thinker.model(
                inputs_embeds=token_embeds,
                position_ids=positions,
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            actual = decoder_sessions["step"].run(
                None,
                {
                    "input_embeds": token_embeds.numpy(),
                    "position_ids": positions.numpy(),
                    "past_keys": actual[1],
                    "past_values": actual[2],
                },
            )
            assert_decoder_matches(actual, native_outputs(text_model, output))
