#!/usr/bin/env python3
"""
AWQ smoothing for Qwen3-ASR INT8 quantization.

Collects per-channel activation statistics from RMSNorm outputs during
calibration, computes smoothing scales, and migrates variance from activation
channels into preceding RMSNorm weights and following linear weights.

The smoothed model is mathematically equivalent to the original under FP32.
After smoothing, per-tensor INT8 quantization (quantize.py) produces higher
quality because outlier activation channels are absorbed into weights.

Smoothing groups (norm → linears sharing its output):
  layers[i].input_layernorm           → self_attn.{q,k,v}_proj
  layers[i].post_attention_layernorm  → mlp.{gate,up}_proj
  model.norm                          → lm_head

Not smoothed (no preceding norm): o_proj, down_proj.

AWQ math for RMSNorm (γ is norm weight, s is per-channel scale,
W is linear weight [out, in]):
  γ_new = γ / s
  W_new[j, i] = W[j, i] * s[i]
Mathematically equivalent: RMSNorm_new(x) @ W_new^T = RMSNorm(x) @ W^T

Usage:
    uv run python awq_smooth.py \\
        --model Qwen/Qwen3-ASR-0.6B \\
        --output output/qwen3-asr-0.6b-smooth \\
        --alpha 0.5 --n-samples 128 --verify
"""

import argparse
import io
import os
import time

import numpy as np
import onnx
import soundfile as sf
import torch
from datasets import Audio, load_dataset

from export import (
    copy_tokenizer,
    extract_embed_tokens,
    load_model,
    write_config,
)
from share_weights import share_external_models
from src.decoder_wrapper import export_decoder_init, export_decoder_step
from src.encoder_wrapper import export_encoder
from src.mel import log_mel_spectrogram
from src.prompt import EOS_TOKEN_IDS, build_prompt_ids, get_audio_pad_range
from validate import greedy_decode_pytorch, run_encoder_pytorch


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def collect_activations(
    model,
    n_samples: int,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Collect per-channel mean absolute activation statistics from RMSNorm outputs.

    Registers forward hooks on each target RMSNorm, runs calibration forward
    passes through the decoder, and accumulates output.abs().mean(dim=(0,1))
    per channel across all passes.

    Returns dict mapping norm name -> Tensor[hidden_size] of mean absolute
    per-channel activations.
    """
    text_model = model.thinker.model
    num_layers = len(text_model.layers)

    # Identify target norm modules and their names
    norm_modules: dict[str, torch.nn.Module] = {}
    for i in range(num_layers):
        layer = text_model.layers[i]
        norm_modules[f"layers.{i}.input_layernorm"] = layer.input_layernorm
        norm_modules[f"layers.{i}.post_attention_layernorm"] = layer.post_attention_layernorm
    norm_modules["norm"] = text_model.norm

    # Accumulators: sum of per-channel mean-abs activations, and count
    accum: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    hooks = []
    for name, module in norm_modules.items():
        def make_hook(n: str):
            def hook(mod, inp, output):
                # output: [batch, seq_len, hidden_size]
                stats = output.detach().float().abs().mean(dim=(0, 1))  # [hidden_size]
                if n not in accum:
                    accum[n] = stats.clone()
                    counts[n] = 1
                else:
                    accum[n] += stats
                    counts[n] += 1
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"Collecting activations from {n_samples} calibration samples (librispeech train.clean.100)...")

    ds = load_dataset(
        "openslr/librispeech_asr",
        "all",
        split="train.clean.100",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    sample_count = 0
    for sample in ds:
        if sample_count >= n_samples:
            break

        raw = sample["audio"].get("bytes")
        if raw is None:
            continue
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

        if len(arr) < 3200:  # < 0.2s
            continue

        try:
            mel = log_mel_spectrogram(arr, device=device)
            audio_features = run_encoder_pytorch(model, mel)
            audio_token_count = audio_features.shape[1]
            if audio_token_count == 0:
                continue

            prompt_ids = build_prompt_ids(audio_token_count)
            # Hooks accumulate on every decoder forward call inside greedy_decode_pytorch
            greedy_decode_pytorch(
                model, audio_features, prompt_ids,
                max_tokens=64, device=device,
            )

            sample_count += 1
            if sample_count % 16 == 0:
                print(f"  {sample_count}/{n_samples} samples processed")

        except Exception as e:
            print(f"  Warning: sample skipped ({e})")
            continue

    for h in hooks:
        h.remove()

    if sample_count == 0:
        raise RuntimeError("No calibration samples collected — check HF dataset access")

    total_passes = sum(counts.values())
    print(f"  Done: {sample_count} samples, {total_passes} decoder forward passes")

    # Average over all accumulated passes
    result: dict[str, torch.Tensor] = {}
    for name in norm_modules:
        if name in accum:
            result[name] = accum[name] / counts[name]
        else:
            # Dead module — uniform scale (no smoothing)
            hidden_size = model.config.thinker_config.text_config.hidden_size
            result[name] = torch.ones(hidden_size, dtype=torch.float32)

    return result


# ---------------------------------------------------------------------------
# Scale computation
# ---------------------------------------------------------------------------

def compute_scales(
    activations: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Compute per-channel smoothing scales from activation statistics.

    s[i] = (act[i] / mean(act)).pow(alpha), clamped to [1e-5, 1e5]

    alpha=0: no smoothing (s=1 everywhere)
    alpha=1: full migration of activation variance into weights
    alpha=0.5: balanced (Lin et al. AWQ default)
    """
    scales: dict[str, torch.Tensor] = {}
    for name, act in activations.items():
        mean_act = act.mean().clamp(min=1e-8)
        s = (act / mean_act).pow(alpha).clamp(min=1e-5, max=1e5)
        scales[name] = s
    return scales


# ---------------------------------------------------------------------------
# Smoothing application
# ---------------------------------------------------------------------------

def apply_smoothing(model, scales: dict[str, torch.Tensor]) -> None:
    """
    Apply AWQ smoothing in-place to all three norm groups.

    For each group:
        norm.weight.data /= s
        linear.weight.data *= s[None, :]    (scales input channels)

    Handles tied lm_head / embed_tokens: clones lm_head.weight before modifying
    so embed_tokens is not corrupted.
    """
    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    embed_tokens = text_model.embed_tokens

    # Detect and break tied weights before any modification
    if lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr():
        print("  Tied lm_head/embed_tokens detected — cloning lm_head weight")
        lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())

    num_layers = len(text_model.layers)

    for i in range(num_layers):
        layer = text_model.layers[i]

        # Group 1: input_layernorm → q_proj, k_proj, v_proj
        s = scales[f"layers.{i}.input_layernorm"].to(layer.input_layernorm.weight.device)
        layer.input_layernorm.weight.data /= s
        for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            proj.weight.data *= s[None, :]

        # Group 2: post_attention_layernorm → gate_proj, up_proj
        s = scales[f"layers.{i}.post_attention_layernorm"].to(layer.post_attention_layernorm.weight.device)
        layer.post_attention_layernorm.weight.data /= s
        for proj in [layer.mlp.gate_proj, layer.mlp.up_proj]:
            proj.weight.data *= s[None, :]

    # Group 3: final norm → lm_head
    s = scales["norm"].to(text_model.norm.weight.device)
    text_model.norm.weight.data /= s
    lm_head.weight.data *= s[None, :]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_equivalence(
    original_model,
    smoothed_model,
    test_audio_path: str,
    device: str = "cpu",
) -> bool:
    """
    Verify smoothed model produces identical output tokens as original.

    The encoder is not modified by smoothing, so run it once on either model.
    Divergence indicates a bug in the smoothing math (wrong dimension, sign, group).

    Returns True if token sequences are identical.
    """
    audio, sr = sf.read(test_audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    mel = log_mel_spectrogram(audio, device=device)
    # Encoder not modified — run on original, reuse features for both
    audio_features = run_encoder_pytorch(original_model, mel)
    prompt_ids = build_prompt_ids(audio_features.shape[1])

    print("  Original model...")
    orig_tokens = greedy_decode_pytorch(original_model, audio_features, prompt_ids, device=device)
    print("  Smoothed model...")
    smooth_tokens = greedy_decode_pytorch(smoothed_model, audio_features, prompt_ids, device=device)

    if orig_tokens == smooth_tokens:
        print(f"  PASS: Identical token output ({len(orig_tokens)} tokens)")
        return True

    match = sum(1 for a, b in zip(orig_tokens, smooth_tokens) if a == b)
    total = max(len(orig_tokens), len(smooth_tokens))
    print(f"  FAIL: Token mismatch ({match}/{total} match)")
    for i, (a, b) in enumerate(zip(orig_tokens, smooth_tokens)):
        if a != b:
            print(f"  First divergence at position {i}: original={a} smoothed={b}")
            break
    return False


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_scale_stats(scales: dict[str, torch.Tensor]) -> None:
    """Print per-group smoothing scale statistics."""
    print("\nSmoothing scale statistics:")
    print(f"  {'Norm':<52} {'min':>7} {'max':>7} {'mean':>7} {'max/min':>8}")
    print(f"  {'-'*84}")

    # Print in layer order, grouped by suffix
    for suffix in ["input_layernorm", "post_attention_layernorm", "norm"]:
        for name in sorted(scales.keys()):
            if not name.endswith(suffix):
                continue
            s = scales[name]
            s_min = s.min().item()
            s_max = s.max().item()
            s_mean = s.mean().item()
            ratio = s_max / max(s_min, 1e-8)
            print(f"  {name:<52} {s_min:>7.3f} {s_max:>7.3f} {s_mean:>7.3f} {ratio:>8.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AWQ smoothing for Qwen3-ASR ONNX export")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-ASR-0.6B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for smoothed ONNX (default: derived from model name)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Smoothing strength α (0=no smoothing, 1=full, default: 0.5)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=128,
        help="Number of calibration samples from librispeech train.clean.100",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify smoothed model produces identical tokens to original",
    )
    parser.add_argument(
        "--verify-audio", type=str, default="tests/fixtures/test_audio.wav",
        help="Audio file for equivalence verification",
    )
    parser.add_argument(
        "--activations-cache", type=str, default=None,
        help="Path to .npz file for loading/saving calibration activations. "
             "If the file exists, calibration is skipped (useful for alpha sweeps).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--skip-encoder", action="store_true",
        help="Skip encoder export",
    )
    args = parser.parse_args()

    if args.output is None:
        name = args.model.rstrip("/").rsplit("/", 1)[-1]
        args.output = os.path.join("output", name.lower() + "-smooth")

    os.makedirs(args.output, exist_ok=True)

    # Determine activations cache path
    acts_cache = args.activations_cache or os.path.join(args.output, "calibration_activations.npz")

    # Load model
    model = load_model(args.model, device=args.device, dtype=torch.float32)

    # Collect or load calibration activations
    if os.path.exists(acts_cache):
        print(f"Loading cached activations from {acts_cache}")
        data = np.load(acts_cache)
        activations = {k: torch.from_numpy(data[k]) for k in data.files}
        print(f"  Loaded {len(activations)} activation vectors")
    else:
        t0 = time.time()
        activations = collect_activations(model, n_samples=args.n_samples, device=args.device)
        print(f"  Collection done in {time.time() - t0:.1f}s")
        np.savez(acts_cache, **{k: v.numpy() for k, v in activations.items()})
        print(f"  Activations saved to {acts_cache}")

    # Compute smoothing scales
    print(f"\nComputing scales (alpha={args.alpha})...")
    scales = compute_scales(activations, alpha=args.alpha)
    scales_path = os.path.join(args.output, "smoothing_scales.npz")
    np.savez(scales_path, **{k: v.numpy() for k, v in scales.items()})
    print(f"  Scales saved to {scales_path}")

    print_scale_stats(scales)

    # Load second copy for verification before modifying original
    model_orig = None
    if args.verify:
        if os.path.exists(args.verify_audio):
            print(f"\nLoading second model copy for equivalence verification...")
            model_orig = load_model(args.model, device=args.device, dtype=torch.float32)
        else:
            print(f"Skipping verify: {args.verify_audio} not found")

    # Apply smoothing in-place
    print("\nApplying AWQ smoothing...")
    apply_smoothing(model, scales)
    print("  Done")

    # Verify equivalence
    if args.verify and model_orig is not None:
        print(f"\nVerifying equivalence on {args.verify_audio}...")
        ok = verify_equivalence(model_orig, model, args.verify_audio, device=args.device)
        if not ok:
            print("WARNING: Smoothed model produces different tokens — review implementation")

    # Export ONNX
    if not args.skip_encoder:
        print("\n=== Exporting encoder ===")
        encoder_path = os.path.join(args.output, "encoder.onnx")
        export_encoder(model, encoder_path, opset_version=args.opset, device=args.device)
        encoder_data = encoder_path + ".data"
        if os.path.exists(encoder_data):
            print("  Embedding encoder weights into .onnx proto...")
            enc = onnx.load(encoder_path, load_external_data=True)
            onnx.save(enc, encoder_path)
            os.remove(encoder_data)
    else:
        print("\n=== Encoder skipped (--skip-encoder) ===")

    print("\n=== Exporting decoder (init) ===")
    export_decoder_init(
        model,
        os.path.join(args.output, "decoder_init.onnx"),
        opset_version=args.opset,
        device=args.device,
    )

    print("\n=== Exporting decoder (step) ===")
    export_decoder_step(
        model,
        os.path.join(args.output, "decoder_step.onnx"),
        opset_version=args.opset,
        device=args.device,
    )

    print("\n=== Sharing decoder weights ===")
    share_external_models(args.output)

    print("\n=== Extracting embedding matrix ===")
    embed_shape = extract_embed_tokens(model, args.output)

    print("\n=== Copying tokenizer ===")
    copy_tokenizer(args.model, args.output)

    print("\n=== Writing config ===")
    write_config(model, args.output, embed_shape)

    print(f"\nExport complete. Output directory: {args.output}")
    print("Files:")
    for f in sorted(os.listdir(args.output)):
        path = os.path.join(args.output, f)
        size = os.path.getsize(path)
        if size > 1e6:
            print(f"  {f}: {size / 1e6:.1f} MB")
        else:
            print(f"  {f}: {size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
