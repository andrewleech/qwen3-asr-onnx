"""
Wrapper modules for Qwen3-ASR decoder export to ONNX.

Exports two ONNX graphs (split decoder architecture):
- decoder_init: Prefill — processes input_ids + audio features, outputs logits + KV cache
- decoder_step: Autoregressive — processes single token ID + KV cache, outputs logits + updated KV cache

These wrappers manually iterate through decoder layers, computing attention
explicitly with plain tensors. This avoids DynamicCache which cannot be traced
by torch.export.

KV cache is represented as two stacked tensors:
- past_keys:   [num_layers, batch, kv_heads, seq_len, head_dim]
- past_values: [num_layers, batch, kv_heads, seq_len, head_dim]

The decoder uses Qwen3 architecture:
    SwiGLU MLP, per-head Q/K RMSNorm, interleaved MRoPE
    Dimensions vary by model size (0.6B: d=1024, 1.7B: d=2048, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Qwen3-0.6B decoder dimensions (kept for test compatibility; wrappers read from text_config)
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states, n_rep):
    """Expand KV heads: [batch, kv_heads, seq, dim] -> [batch, q_heads, seq, dim]."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _attention(query, key, value, mask, scaling, num_kv_groups):
    """Eager attention with GQA expansion."""
    key = _repeat_kv(key, num_kv_groups)
    value = _repeat_kv(value, num_kv_groups)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if mask is not None:
        attn_weights = attn_weights + mask[:, :, :, : key.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous()


def _decoder_layer_forward(layer, hidden_states, cos, sin, mask, past_key, past_value, num_kv_groups):
    """
    Run a single decoder layer, returning (hidden_states, key_states, value_states).

    Args:
        layer: Qwen3 decoder layer module
        hidden_states: [batch, seq_len, 1024]
        cos, sin: Rotary position embeddings [batch, seq_len, head_dim]
        mask: Causal attention mask [batch, 1, q_len, kv_len] or None
        past_key: [batch, kv_heads, past_seq, head_dim] or None
        past_value: [batch, kv_heads, past_seq, head_dim] or None

    Returns:
        (hidden_states, key_states, value_states)
        key_states/value_states are post-RoPE, concatenated with past if provided
    """
    attn = layer.self_attn

    # Input LayerNorm + attention
    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)

    # Q/K/V projections
    input_shape = normed.shape[:-1]  # [batch, seq_len]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

    # Apply RoPE
    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Concatenate with past KV (step only; past_key is None during prefill).
    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    # Attention
    attn_output = _attention(query_states, key_states, value_states, mask, attn.scaling, num_kv_groups)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn.o_proj(attn_output)

    hidden_states = residual + attn_output

    # MLP
    residual = hidden_states
    normed = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(normed)

    return hidden_states, key_states, value_states


class DecoderInitWrapper(nn.Module):
    """
    Decoder prefill: processes input_ids + audio features, returns logits and KV cache.

    Embedding lookup happens inside the graph. Audio features are scattered over
    the audio_pad token positions in the embedding sequence.

    Manually iterates through layers to avoid DynamicCache.
    For audio-only ASR, MRoPE degenerates to standard 1D RoPE since all three
    position dimensions (temporal, height, width) receive identical values.
    We accept 1D position_ids and tile to 3D internally.
    """

    def __init__(self, text_model, lm_head, text_config):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        audio_features: torch.Tensor,
        audio_offset: torch.Tensor,
    ):
        """
        Args:
            input_ids: [batch, seq_len] token IDs (includes audio_pad placeholders)
            position_ids: [batch, seq_len]
            audio_features: [batch, audio_len, hidden_size] encoder output (FP32 from encoder)
            audio_offset: [1] start position of audio_pad tokens in the sequence

        Returns:
            (logits, present_keys, present_values):
                logits: [batch, seq_len, vocab_size] (FP32 for argmax compatibility)
                present_keys: [num_layers, batch, kv_heads, seq_len, head_dim]
                present_values: [num_layers, batch, kv_heads, seq_len, head_dim]
        """
        # Embedding lookup
        input_embeds = self.embed_tokens(input_ids)  # [B, S, H]

        # Scatter audio features over audio_pad positions
        audio_len = audio_features.shape[1]
        indices = torch.arange(audio_len, device=input_ids.device) + audio_offset[0]
        indices = indices.unsqueeze(0).unsqueeze(-1).expand_as(audio_features)
        input_embeds = input_embeds.scatter(1, indices, audio_features)

        batch, seq_len = input_embeds.shape[:2]

        # Tile position_ids to 3D for MRoPE: [3, batch, seq_len]
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Compute rotary position embeddings (cos, sin)
        cos, sin = self.rotary_emb(input_embeds, pos_3d)

        # Create causal attention mask: [1, 1, seq_len, seq_len]
        causal_mask = torch.full(
            (seq_len, seq_len), torch.finfo(input_embeds.dtype).min,
            device=input_embeds.device, dtype=input_embeds.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        hidden_states = input_embeds
        all_keys = []
        all_values = []

        for layer in self.layers:
            hidden_states, key_states, value_states = _decoder_layer_forward(
                layer, hidden_states, cos, sin, causal_mask,
                past_key=None, past_value=None,
                num_kv_groups=self.num_kv_groups,
            )
            all_keys.append(key_states)
            all_values.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_keys = torch.stack(all_keys, dim=0)
        present_values = torch.stack(all_values, dim=0)
        return logits, present_keys, present_values


class DecoderStepWrapper(nn.Module):
    """
    Decoder autoregressive step: processes single pre-embedded token with KV cache.

    Unlike decoder_init, this wrapper does NOT contain the embedding table.
    The caller performs embedding lookup externally (from a cached FP32 copy of
    the embedding matrix) and passes input_embeds directly. This keeps the
    decoder_step session small enough to inline weights in the .onnx proto
    (~596 MB INT8) and avoids memory-mapping a large shared external data file.

    Manually iterates through layers to avoid DynamicCache.
    """

    def __init__(self, text_model, lm_head, text_config):
        super().__init__()
        # Note: no self.embed_tokens — embedding lookup is done externally
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
    ):
        """
        Args:
            input_embeds: [batch, 1, hidden_size] pre-looked-up token embedding
            position_ids: [batch, 1]
            past_keys: [num_layers, batch, kv_heads, past_seq, head_dim]
            past_values: [num_layers, batch, kv_heads, past_seq, head_dim]

        Returns:
            (logits, present_keys, present_values):
                logits: [batch, 1, vocab_size]
                present_keys: [num_layers, batch, kv_heads, past_seq+1, head_dim]
                present_values: [num_layers, batch, kv_heads, past_seq+1, head_dim]
        """

        # Tile position_ids to 3D for MRoPE
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Compute rotary embeddings for current position only
        cos, sin = self.rotary_emb(input_embeds, pos_3d)

        # No causal mask needed for single-token query (attends to all past + self)
        mask = None

        hidden_states = input_embeds
        all_keys = []
        all_values = []

        for i, layer in enumerate(self.layers):
            hidden_states, key_states, value_states = _decoder_layer_forward(
                layer, hidden_states, cos, sin, mask,
                past_key=past_keys[i], past_value=past_values[i],
                num_kv_groups=self.num_kv_groups,
            )
            all_keys.append(key_states)
            all_values.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        present_keys = torch.stack(all_keys, dim=0)
        present_values = torch.stack(all_values, dim=0)
        return logits, present_keys, present_values


def export_decoder_init(
    model,
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
):
    """
    Export the decoder prefill graph to ONNX.

    Args:
        model: Loaded Qwen3ASRForConditionalGeneration model.
        output_path: Path to save the .onnx file.
        opset_version: ONNX opset version.
        device: Device for tracing.
    """
    text_config = model.config.thinker_config.text_config
    hidden_size = text_config.hidden_size

    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderInitWrapper(text_model, lm_head, text_config).eval().to(device)

    # Dummy inputs: 100 tokens total, 80 audio tokens starting at offset 9
    seq_len = 100
    audio_len = 80
    audio_offset_val = 9  # standard ASR prompt: 9 prefix tokens before audio_pad block
    dummy_ids = torch.zeros(1, seq_len, device=device, dtype=torch.long)
    dummy_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    dummy_audio = torch.randn(1, audio_len, hidden_size, device=device, dtype=torch.float32)
    dummy_offset = torch.tensor([audio_offset_val], device=device, dtype=torch.long)

    input_names = ["input_ids", "position_ids", "audio_features", "audio_offset"]
    output_names = ["logits", "present_keys", "present_values"]

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "position_ids": {0: "batch", 1: "seq_len"},
        "audio_features": {0: "batch", 1: "audio_len"},
        "logits": {0: "batch", 1: "seq_len"},
        "present_keys": {1: "batch", 3: "seq_len"},
        "present_values": {1: "batch", 3: "seq_len"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_pos, dummy_audio, dummy_offset),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    from .onnx_fixup import fix_reshape_allowzero
    n = fix_reshape_allowzero(output_path)
    print(f"Decoder init exported to {output_path} (fixed {n} Reshape allowzero attrs)")


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
    text_config = model.config.thinker_config.text_config
    hidden_size = text_config.hidden_size
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    head_dim = text_config.head_dim

    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderStepWrapper(text_model, lm_head, text_config).eval().to(device)

    # Dummy inputs: single pre-embedded token + KV cache from 100-token prefill
    past_seq_len = 100
    dummy_embeds = torch.randn(1, 1, hidden_size, device=device, dtype=torch.float32)
    dummy_pos = torch.tensor([[past_seq_len]], device=device, dtype=torch.long)

    # Stacked KV cache: [num_layers, batch, kv_heads, past_seq, head_dim]
    dummy_past_keys = torch.randn(num_layers, 1, num_kv_heads, past_seq_len, head_dim, device=device, dtype=torch.float32)
    dummy_past_values = torch.randn(num_layers, 1, num_kv_heads, past_seq_len, head_dim, device=device, dtype=torch.float32)

    input_names = ["input_embeds", "position_ids", "past_keys", "past_values"]
    output_names = ["logits", "present_keys", "present_values"]

    dynamic_axes = {
        "input_embeds": {0: "batch"},
        "position_ids": {0: "batch"},
        "past_keys": {1: "batch", 3: "past_seq_len"},
        "past_values": {1: "batch", 3: "past_seq_len"},
        "logits": {0: "batch"},
        "present_keys": {1: "batch", 3: "total_seq_len"},
        "present_values": {1: "batch", 3: "total_seq_len"},
    }

    args = (dummy_embeds, dummy_pos, dummy_past_keys, dummy_past_values)

    with torch.no_grad():
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

    from .onnx_fixup import fix_reshape_allowzero
    n = fix_reshape_allowzero(output_path)
    print(f"Decoder step exported to {output_path} (fixed {n} Reshape allowzero attrs)")
