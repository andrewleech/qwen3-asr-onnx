"""
Test decoder ONNX export correctness against PyTorch reference.

Tests both decoder_init (prefill) and decoder_step (autoregressive).

The decoder uses stacked KV cache format:
    present_keys:   [num_layers, batch, kv_heads, seq_len, head_dim]
    present_values: [num_layers, batch, kv_heads, seq_len, head_dim]

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
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
    except ValueError as e:
        print(f"AutoModel.from_pretrained failed ({e}), falling back to direct import")
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B", torch_dtype=torch.float32, device_map="cpu",
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
        """Init session should output logits + stacked keys + stacked values."""
        output_names = [o.name for o in init_session.get_outputs()]
        assert output_names == ["logits", "present_keys", "present_values"]

    def test_logits_shape(self, init_session):
        """Logits should have shape [batch, seq_len, vocab_size]."""
        seq_len = 50
        embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        logits, _, _ = init_session.run(
            ["logits", "present_keys", "present_values"],
            {"input_embeds": embeds, "position_ids": pos},
        )

        assert logits.shape == (1, seq_len, 151936)

    def test_kv_cache_shape(self, init_session):
        """Stacked KV cache should have shape [num_layers, batch, kv_heads, seq, head_dim]."""
        seq_len = 50
        embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        _, keys, values = init_session.run(
            ["logits", "present_keys", "present_values"],
            {"input_embeds": embeds, "position_ids": pos},
        )

        expected = (NUM_LAYERS, 1, NUM_KV_HEADS, seq_len, HEAD_DIM)
        assert keys.shape == expected, f"Keys shape {keys.shape} != {expected}"
        assert values.shape == expected, f"Values shape {values.shape} != {expected}"

    def test_pytorch_onnx_match(self, pytorch_model, init_session):
        """ONNX logits should match PyTorch within tolerance."""
        seq_len = 30
        embeds_np = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos_np = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        # ONNX
        onnx_logits, _, _ = init_session.run(
            ["logits", "present_keys", "present_values"],
            {"input_embeds": embeds_np, "position_ids": pos_np},
        )

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
        seq_len = 30
        embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        _, present_keys, present_values = init_session.run(
            ["logits", "present_keys", "present_values"],
            {"input_embeds": embeds, "position_ids": pos},
        )

        # Run one step
        step_embeds = np.random.randn(1, 1, 1024).astype(np.float32)
        step_pos = np.array([[seq_len]], dtype=np.int64)

        logits, new_keys, new_values = step_session.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": step_embeds,
                "position_ids": step_pos,
                "past_keys": present_keys,
                "past_values": present_values,
            },
        )

        assert logits.shape == (1, 1, 151936)
        assert new_keys.shape == (NUM_LAYERS, 1, NUM_KV_HEADS, seq_len + 1, HEAD_DIM)
        assert new_values.shape == (NUM_LAYERS, 1, NUM_KV_HEADS, seq_len + 1, HEAD_DIM)

    def test_multi_step(self, init_session, step_session):
        """Multiple decode steps should accumulate KV cache correctly."""
        seq_len = 20
        embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        _, present_keys, present_values = init_session.run(
            ["logits", "present_keys", "present_values"],
            {"input_embeds": embeds, "position_ids": pos},
        )

        # Run 5 steps
        current_pos = seq_len
        for step in range(5):
            step_embeds = np.random.randn(1, 1, 1024).astype(np.float32)
            step_pos = np.array([[current_pos]], dtype=np.int64)

            _, present_keys, present_values = step_session.run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_embeds": step_embeds,
                    "position_ids": step_pos,
                    "past_keys": present_keys,
                    "past_values": present_values,
                },
            )
            current_pos += 1

        expected_seq = seq_len + 5
        assert present_keys.shape == (NUM_LAYERS, 1, NUM_KV_HEADS, expected_seq, HEAD_DIM)
        assert present_values.shape == (NUM_LAYERS, 1, NUM_KV_HEADS, expected_seq, HEAD_DIM)
