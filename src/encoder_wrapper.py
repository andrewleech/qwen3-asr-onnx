"""
Wrapper module for Qwen3-ASR audio encoder export to ONNX.

Reimplements the native encoder's windowed convolution and windowed attention
using trace-friendly operations (no cu_seqlens, no boolean mask indexing,
no dynamic splitting, no data-dependent branching).

Processing stages:
  1. Windowed Conv2D: pad mel to multiple of 100, batch-convolve
  2. Positional embeddings per window (positions 0..12)
  3. Flatten, pad to multiple of 104 (attention window), reshape to batched windows
  4. Batched bidirectional attention through transformer layers
  5. Flatten, trim to valid_count, project

Input: mel spectrogram [1, 128, time]
Output: audio features [1, tokens, 1024]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 0.6B encoder dimensions
NUM_HEADS = 14
HEAD_DIM = 64
D_MODEL = 896
CONV_WINDOW = 100       # n_window * 2
TOKENS_PER_WINDOW = 13  # tokens output by conv on a full 100-frame window
ATTN_WINDOW_SIZE = 104  # TOKENS_PER_WINDOW * (n_window_infer // conv_window) = 13 * 8


def _conv_out_len(t):
    """Output length after one stride-2 conv (kernel=3, padding=1).

    Equivalent to (t - 1) // 2 + 1 for t >= 1 and 0 for t = 0.
    Uses (t + 1) // 2 to avoid negative intermediate values, which
    would give wrong results in ONNX (truncation-toward-zero division).
    """
    return (t + 1) // 2


def _get_feat_extract_output_lengths(input_lengths):
    """Number of encoder tokens from mel frame count.

    Matches the native Qwen3-ASR formula. Uses _conv_out_len instead of
    (x - 1) // 2 + 1 to be safe under ONNX integer division semantics.
    """
    leave = input_lengths % CONV_WINDOW
    t = _conv_out_len(leave)
    t = _conv_out_len(t)
    t = _conv_out_len(t)
    return t + (input_lengths // CONV_WINDOW) * TOKENS_PER_WINDOW


def _encoder_attention(q, k, v, mask, scaling):
    """
    Batched bidirectional attention with additive mask.

    Args:
        q, k, v: [batch, num_heads, seq, head_dim]
        mask: [batch, 1, 1, seq] additive mask (-inf for padding)
        scaling: float
    Returns:
        [batch, seq, num_heads * head_dim]
    """
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
    attn_weights = attn_weights + mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn_weights, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def _encoder_layer_forward(layer, x, attn_mask, scaling):
    """
    Single encoder layer forward with batched 3D input.

    Args:
        layer: Qwen3ASRAudioEncoderLayer
        x: [batch, seq, d_model]
        attn_mask: [batch, 1, 1, seq] additive mask
        scaling: float

    Returns:
        [batch, seq, d_model]
    """
    sa = layer.self_attn
    batch, seq, _ = x.shape

    residual = x
    normed = layer.self_attn_layer_norm(x)

    q = sa.q_proj(normed).view(batch, seq, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    k = sa.k_proj(normed).view(batch, seq, NUM_HEADS, HEAD_DIM).transpose(1, 2)
    v = sa.v_proj(normed).view(batch, seq, NUM_HEADS, HEAD_DIM).transpose(1, 2)

    attn_out = _encoder_attention(q, k, v, attn_mask, scaling)
    attn_out = sa.out_proj(attn_out)
    x = residual + attn_out

    residual = x
    normed = layer.final_layer_norm(x)
    x = residual + layer.fc2(F.gelu(layer.fc1(normed)))

    return x


class EncoderWrapper(nn.Module):
    """
    Reimplements the Qwen3ASRAudioEncoder forward pass with windowed
    convolution and windowed attention, using ONNX-traceable operations.

    All control flow is branch-free (no data-dependent if/else) to ensure
    correct tracing under torch.export / torch.onnx.export.
    """

    def __init__(self, audio_tower):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        self.positional_embedding = audio_tower.positional_embedding
        self.layers = audio_tower.layers
        self.ln_post = audio_tower.ln_post
        self.proj1 = audio_tower.proj1
        self.proj2 = audio_tower.proj2
        self.act = audio_tower.act
        self.scaling = self.layers[0].self_attn.scaling

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [1, 128, T] log-mel spectrogram.

        Returns:
            [1, valid_count, 1024] audio features.
        """
        T = mel.shape[2]

        # --- Stage 1: Windowed Conv2D ---
        # Pad T to next multiple of CONV_WINDOW (no-op when already aligned)
        pad_amount = (CONV_WINDOW - T % CONV_WINDOW) % CONV_WINDOW
        mel = F.pad(mel, (0, pad_amount))
        T_padded = mel.shape[2]
        num_conv_windows = T_padded // CONV_WINDOW

        # [1, 128, T_padded] -> [N, 1, 128, CONV_WINDOW]
        x = mel.squeeze(0)                                     # [128, T_padded]
        x = x.reshape(128, num_conv_windows, CONV_WINDOW)      # [128, N, 100]
        x = x.permute(1, 0, 2)                                 # [N, 128, 100]
        x = x.unsqueeze(1)                                     # [N, 1, 128, 100]

        # Conv2D stem
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        # [N, 480, 16, tokens_per_window] -> [N, tokens_per_window, 7680] -> [N, tpw, 896]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.conv_out(x)

        # Per-window positional embeddings (same positions 0..tpw-1 for every window)
        pos_embed = self.positional_embedding(t)  # [tpw, 896]
        x = x + pos_embed.unsqueeze(0)

        # --- Stage 2: Flatten and compute valid_count ---
        valid_count = _get_feat_extract_output_lengths(T)
        flat = x.reshape(-1, D_MODEL)     # [N * tpw, 896]
        flat = flat[:valid_count]          # trim conv-padding tokens

        # --- Stage 3: Pad to multiple of ATTN_WINDOW_SIZE, reshape to windows ---
        attn_pad = (ATTN_WINDOW_SIZE - valid_count % ATTN_WINDOW_SIZE) % ATTN_WINDOW_SIZE
        flat = F.pad(flat, (0, 0, 0, attn_pad))  # no-op when attn_pad=0
        total_padded = valid_count + attn_pad
        num_attn_windows = total_padded // ATTN_WINDOW_SIZE
        x = flat.reshape(num_attn_windows, ATTN_WINDOW_SIZE, D_MODEL)

        # --- Stage 4: Attention mask (branch-free) ---
        # Global position for each element; positions >= valid_count are padding
        positions = torch.arange(total_padded, device=mel.device)
        positions = positions.reshape(num_attn_windows, ATTN_WINDOW_SIZE)
        pad_mask = (positions >= valid_count).to(mel.dtype) * torch.finfo(mel.dtype).min
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(1)  # [num_windows, 1, 1, ATTN_WINDOW_SIZE]

        # --- Stage 5: Transformer layers ---
        for layer in self.layers:
            x = _encoder_layer_forward(layer, x, attn_mask, self.scaling)

        # --- Stage 6: Flatten, trim, project ---
        x = x.reshape(-1, D_MODEL)[:valid_count]
        x = x.unsqueeze(0)  # [1, valid_count, 896]

        x = self.ln_post(x)
        x = self.act(self.proj1(x))
        x = self.proj2(x)

        return x  # [1, valid_count, 1024]


def export_encoder(
    model,
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
):
    """Export the audio encoder to ONNX."""
    audio_tower = model.thinker.audio_tower
    wrapper = EncoderWrapper(audio_tower).eval().to(device)

    # Use non-round frame count to exercise padding code during tracing
    dummy_mel = torch.randn(1, 128, 997, device=device, dtype=torch.float32)

    with torch.no_grad():
        test_output = wrapper(dummy_mel)
        expected_tokens = _get_feat_extract_output_lengths(997)
        assert test_output.shape == (1, expected_tokens, 1024), (
            f"Shape {test_output.shape} != expected (1, {expected_tokens}, 1024)"
        )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_mel,),
            output_path,
            input_names=["mel"],
            output_names=["audio_features"],
            dynamic_axes={
                "mel": {2: "time"},
                "audio_features": {1: "enc_time"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    print(f"Encoder exported to {output_path}")
