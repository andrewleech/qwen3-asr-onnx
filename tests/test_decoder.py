"""
Test decoder ONNX export correctness against PyTorch reference.

Tests both decoder_init (prefill) and decoder_step (autoregressive).

Requires:
    - Qwen3-ASR-0.6B model accessible
    - decoder_init.onnx and decoder_step.onnx in output directory
"""

import os

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src.decoder_wrapper import NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM

pytestmark = pytest.mark.skipif(
    not os.path.exists("output/qwen3-asr-0.6b/decoder_init.onnx"),
    reason="decoder ONNX files not found - run export.py first",
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
def init_session():
    return ort.InferenceSession(
        "output/qwen3-asr-0.6b/decoder_init.onnx",
        providers=["CPUExecutionProvider"],
    )


@pytest.fixture(scope="module")
def step_session():
    return ort.InferenceSession(
        "output/qwen3-asr-0.6b/decoder_step.onnx",
        providers=["CPUExecutionProvider"],
    )


class TestDecoderInit:
    def test_output_names(self, init_session):
        """Init session should output logits + KV cache."""
        output_names = [o.name for o in init_session.get_outputs()]
        assert output_names[0] == "logits"
        assert len(output_names) == 1 + NUM_LAYERS * 2  # logits + key/value per layer

    def test_logits_shape(self, init_session):
        """Logits should have shape [batch, seq_len, vocab_size]."""
        seq_len = 50
        embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        results = init_session.run(None, {"input_embeds": embeds, "position_ids": pos})
        logits = results[0]

        assert logits.shape[0] == 1
        assert logits.shape[1] == seq_len
        assert logits.shape[2] == 151936  # vocab_size

    def test_kv_cache_shape(self, init_session):
        """KV cache tensors should have correct shapes."""
        seq_len = 50
        embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        results = init_session.run(None, {"input_embeds": embeds, "position_ids": pos})

        # Check each KV pair
        for i in range(NUM_LAYERS):
            key = results[1 + i * 2]
            value = results[2 + i * 2]
            assert key.shape == (1, NUM_KV_HEADS, seq_len, HEAD_DIM), (
                f"Layer {i} key shape mismatch: {key.shape}"
            )
            assert value.shape == (1, NUM_KV_HEADS, seq_len, HEAD_DIM), (
                f"Layer {i} value shape mismatch: {value.shape}"
            )

    def test_pytorch_onnx_match(self, pytorch_model, init_session):
        """ONNX logits should match PyTorch within tolerance."""
        seq_len = 30
        embeds_np = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos_np = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        # ONNX
        onnx_results = init_session.run(None, {
            "input_embeds": embeds_np,
            "position_ids": pos_np,
        })
        onnx_logits = onnx_results[0]

        # PyTorch
        embeds_pt = torch.from_numpy(embeds_np)
        pos_pt = torch.from_numpy(pos_np)
        pos_3d = pos_pt.unsqueeze(0).expand(3, -1, -1)

        with torch.no_grad():
            outputs = pytorch_model.thinker.model(
                inputs_embeds=embeds_pt,
                position_ids=pos_3d,
                use_cache=True,
                return_dict=True,
            )
            pt_logits = pytorch_model.thinker.lm_head(outputs.last_hidden_state).numpy()

        max_diff = np.max(np.abs(pt_logits - onnx_logits))
        assert max_diff < 1e-3, f"Logits max diff {max_diff:.6e} exceeds threshold 1e-3"


class TestDecoderStep:
    def test_step_after_init(self, init_session, step_session):
        """Step should work with KV cache from init."""
        # Run init
        seq_len = 30
        embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        init_results = init_session.run(None, {
            "input_embeds": embeds,
            "position_ids": pos,
        })

        # Build step inputs from init outputs
        step_embeds = np.random.randn(1, 1, 1024).astype(np.float32)
        step_pos = np.array([[seq_len]], dtype=np.int64)

        step_inputs = {
            "input_embeds": step_embeds,
            "position_ids": step_pos,
        }

        output_names = [o.name for o in init_session.get_outputs()]
        for i, name in enumerate(output_names[1:], 1):
            step_name = name.replace("present_", "past_")
            step_inputs[step_name] = init_results[i]

        # Run step
        step_results = step_session.run(None, step_inputs)

        logits = step_results[0]
        assert logits.shape == (1, 1, 151936)

        # KV cache should have seq_len + 1
        for i in range(NUM_LAYERS):
            key = step_results[1 + i * 2]
            assert key.shape[2] == seq_len + 1, (
                f"Layer {i} key seq_len should be {seq_len + 1}, got {key.shape[2]}"
            )

    def test_multi_step(self, init_session, step_session):
        """Multiple decode steps should accumulate KV cache correctly."""
        # Init
        seq_len = 20
        embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        results = init_session.run(None, {
            "input_embeds": embeds,
            "position_ids": pos,
        })

        output_names = [o.name for o in init_session.get_outputs()]
        kv_cache = {}
        for i, name in enumerate(output_names[1:], 1):
            kv_cache[name] = results[i]

        # Run 5 steps
        current_pos = seq_len
        for step in range(5):
            step_embeds = np.random.randn(1, 1, 1024).astype(np.float32)
            step_pos = np.array([[current_pos]], dtype=np.int64)

            step_inputs = {
                "input_embeds": step_embeds,
                "position_ids": step_pos,
            }
            for name, value in kv_cache.items():
                step_name = name.replace("present_", "past_")
                step_inputs[step_name] = value

            step_results = step_session.run(None, step_inputs)

            # Update KV cache
            kv_cache = {}
            step_output_names = [o.name for o in step_session.get_outputs()]
            for i, name in enumerate(step_output_names[1:], 1):
                kv_cache[name] = step_results[i]

            current_pos += 1

        # Final KV cache should have seq_len + 5
        for name, value in kv_cache.items():
            assert value.shape[2] == seq_len + 5, (
                f"{name} seq_len should be {seq_len + 5}, got {value.shape[2]}"
            )
