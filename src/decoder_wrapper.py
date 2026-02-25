"""
Wrapper modules for Qwen3-ASR decoder export to ONNX.

Exports two separate ONNX graphs:
- decoder_init: Prefill — processes full input_embeds, outputs logits + KV cache
- decoder_step: Autoregressive — processes single token + KV cache, outputs logits + updated KV cache

The original text model uses DynamicCache internally and has a torch.jit.is_tracing()
guard that disables cache creation during tracing. These wrappers construct the cache
explicitly before calling into the model.

The decoder is Qwen3-0.6B:
    28 layers, d=1024, 16 Q-heads / 8 KV-heads (GQA 2:1), head_dim=128
    SwiGLU MLP, per-head Q/K RMSNorm, interleaved MRoPE
"""

import torch
import torch.nn as nn


# Qwen3-0.6B decoder dimensions
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128


class DecoderInitWrapper(nn.Module):
    """
    Decoder prefill: processes full input sequence, returns logits and KV cache.

    For audio-only ASR, MRoPE degenerates to standard 1D RoPE since all three
    position dimensions (temporal, height, width) receive identical values.
    We accept 1D position_ids and tile to 3D internally.
    """

    def __init__(self, text_model, lm_head):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        """
        Args:
            input_embeds: [batch, seq_len, 1024] — combined text + audio embeddings
            position_ids: [batch, seq_len] — 1D positions (tiled to 3D for MRoPE)

        Returns:
            Tuple of (logits, *kv_cache_tensors):
                logits: [batch, seq_len, vocab_size]
                kv_cache_tensors: 56 tensors (28 layers x 2 for key/value),
                    each [batch, num_kv_heads, seq_len, head_dim]
        """
        # Tile position_ids to 3D for MRoPE: [3, batch, seq_len]
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)

        seq_len = input_embeds.shape[1]
        cache_position = torch.arange(seq_len, device=input_embeds.device)

        # Create a DynamicCache so the model stores KV pairs.
        # We import here to avoid hard dependency at module load time.
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()

        outputs = self.text_model(
            inputs_embeds=input_embeds,
            position_ids=pos_3d,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        # Extract KV cache tensors from DynamicCache
        past_kv = outputs.past_key_values
        result = [logits]
        for i in range(NUM_LAYERS):
            result.append(past_kv.key_cache[i])
            result.append(past_kv.value_cache[i])

        return tuple(result)


class DecoderStepWrapper(nn.Module):
    """
    Decoder autoregressive step: processes single new token with KV cache.
    """

    def __init__(self, text_model, lm_head):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        *past_key_values_flat,
    ):
        """
        Args:
            input_embeds: [batch, 1, 1024] — single token embedding
            position_ids: [batch, 1] — position of this token
            past_key_values_flat: 56 tensors — flattened KV cache from previous steps
                (layer0_key, layer0_value, layer1_key, layer1_value, ...)

        Returns:
            Tuple of (logits, *updated_kv_cache):
                logits: [batch, 1, vocab_size]
                updated_kv_cache: 56 tensors with seq_len extended by 1
        """
        # Tile position_ids to 3D for MRoPE
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Reconstruct DynamicCache from flat tensors
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for i in range(NUM_LAYERS):
            cache.key_cache.append(past_key_values_flat[i * 2])
            cache.value_cache.append(past_key_values_flat[i * 2 + 1])

        # cache_position tells the model where this step is in the sequence
        past_len = past_key_values_flat[0].shape[2]
        cache_position = torch.arange(
            past_len, past_len + 1, device=input_embeds.device
        )

        outputs = self.text_model(
            inputs_embeds=input_embeds,
            position_ids=pos_3d,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        # Extract updated KV cache
        past_kv = outputs.past_key_values
        result = [logits]
        for i in range(NUM_LAYERS):
            result.append(past_kv.key_cache[i])
            result.append(past_kv.value_cache[i])

        return tuple(result)


def _kv_input_names():
    """Generate ONNX input names for KV cache tensors."""
    names = []
    for i in range(NUM_LAYERS):
        names.append(f"past_key_{i}")
        names.append(f"past_value_{i}")
    return names


def _kv_output_names():
    """Generate ONNX output names for KV cache tensors."""
    names = []
    for i in range(NUM_LAYERS):
        names.append(f"present_key_{i}")
        names.append(f"present_value_{i}")
    return names


def _kv_dynamic_axes(prefix: str, seq_dim_name: str):
    """Generate dynamic axes for KV cache tensors."""
    axes = {}
    for i in range(NUM_LAYERS):
        axes[f"{prefix}_key_{i}"] = {0: "batch", 2: seq_dim_name}
        axes[f"{prefix}_value_{i}"] = {0: "batch", 2: seq_dim_name}
    return axes


def export_decoder_init(
    model,
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
):
    """
    Export the decoder prefill graph to ONNX.

    If standard tracing fails (due to DynamicCache or vmap issues), this
    function falls back to torch.onnx.export with dynamo_export=True.

    Args:
        model: Loaded Qwen3ASRForConditionalGeneration model.
        output_path: Path to save the .onnx file.
        opset_version: ONNX opset version.
        device: Device for tracing.
    """
    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderInitWrapper(text_model, lm_head).eval().to(device)

    # Dummy input: 100 tokens (typical for ~8 seconds of audio + prompt tokens)
    seq_len = 100
    dummy_embeds = torch.randn(1, seq_len, 1024, device=device, dtype=torch.float32)
    dummy_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

    input_names = ["input_embeds", "position_ids"]
    output_names = ["logits"] + _kv_output_names()

    dynamic_axes = {
        "input_embeds": {0: "batch", 1: "seq_len"},
        "position_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }
    dynamic_axes.update(_kv_dynamic_axes("present", "seq_len"))

    with torch.no_grad():
        try:
            torch.onnx.export(
                wrapper,
                (dummy_embeds, dummy_pos),
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )
        except Exception as e:
            print(f"Standard ONNX export failed: {e}")
            print("Attempting export with dynamo backend...")
            torch.onnx.export(
                wrapper,
                (dummy_embeds, dummy_pos),
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=True,
            )

    print(f"Decoder init exported to {output_path}")


def export_decoder_step(
    model,
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
):
    """
    Export the decoder autoregressive step graph to ONNX.

    Args:
        model: Loaded Qwen3ASRForConditionalGeneration model.
        output_path: Path to save the .onnx file.
        opset_version: ONNX opset version.
        device: Device for tracing.
    """
    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderStepWrapper(text_model, lm_head).eval().to(device)

    # Dummy inputs: single token + KV cache from 100-token prefill
    past_seq_len = 100
    dummy_embeds = torch.randn(1, 1, 1024, device=device, dtype=torch.float32)
    dummy_pos = torch.tensor([[past_seq_len]], device=device, dtype=torch.long)

    # Dummy KV cache: 56 tensors
    dummy_kv = []
    for _ in range(NUM_LAYERS):
        dummy_kv.append(torch.randn(1, NUM_KV_HEADS, past_seq_len, HEAD_DIM, device=device, dtype=torch.float32))
        dummy_kv.append(torch.randn(1, NUM_KV_HEADS, past_seq_len, HEAD_DIM, device=device, dtype=torch.float32))

    input_names = ["input_embeds", "position_ids"] + _kv_input_names()
    output_names = ["logits"] + _kv_output_names()

    dynamic_axes = {
        "input_embeds": {0: "batch"},
        "position_ids": {0: "batch"},
        "logits": {0: "batch"},
    }
    dynamic_axes.update(_kv_dynamic_axes("past", "past_seq_len"))
    dynamic_axes.update(_kv_dynamic_axes("present", "total_seq_len"))

    args = (dummy_embeds, dummy_pos, *dummy_kv)

    with torch.no_grad():
        try:
            torch.onnx.export(
                wrapper,
                args,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )
        except Exception as e:
            print(f"Standard ONNX export failed: {e}")
            print("Attempting export with dynamo backend...")
            torch.onnx.export(
                wrapper,
                args,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=True,
            )

    print(f"Decoder step exported to {output_path}")
