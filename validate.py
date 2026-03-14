#!/usr/bin/env python3
"""
Validate ONNX exports against PyTorch reference.

Loads both the original PyTorch model and the exported ONNX files,
runs the same inputs through both, and compares outputs.

Usage:
    python validate.py --onnx-dir output/qwen3-asr-0.6b --audio tests/fixtures/test_audio.wav
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer

from src.inference import greedy_decode_onnx
from src.mel import log_mel_spectrogram
from src.prompt import (
    build_prompt_ids,
    get_audio_pad_range,
    EOS_TOKEN_IDS,
    AUDIO_PAD_TOKEN_ID,
)


def load_pytorch_model(model_id: str, device: str = "cpu"):
    """Load the PyTorch model for reference."""
    try:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
    except ValueError as e:
        print(f"AutoModel.from_pretrained failed ({e}), falling back to direct import")
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=device,
        )
    model.eval()
    return model


def load_onnx_sessions(onnx_dir: str):
    """Load all ONNX runtime sessions."""
    sessions = {}
    providers = ["CPUExecutionProvider"]

    # Check for CUDA
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    for name in ["encoder", "decoder_init", "decoder_step"]:
        path = os.path.join(onnx_dir, f"{name}.onnx")
        if os.path.exists(path):
            sessions[name] = ort.InferenceSession(path, providers=providers)
            print(f"  Loaded {name}.onnx")
        else:
            print(f"  WARNING: {path} not found")

    return sessions


def load_embed_tokens(onnx_dir: str) -> np.ndarray:
    """Load embedding matrix from raw binary file."""
    config_path = os.path.join(onnx_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    shape = config["embed_tokens_shape"]
    embed_path = os.path.join(onnx_dir, "embed_tokens.bin")
    embed = np.fromfile(embed_path, dtype=np.float32).reshape(shape)
    print(f"  Loaded embed_tokens.bin: {embed.shape}")
    return embed


def run_encoder_onnx(session, mel: np.ndarray) -> np.ndarray:
    """Run encoder through ONNX Runtime."""
    result = session.run(["audio_features"], {"mel": mel})
    return result[0]


def run_encoder_pytorch(model, mel: torch.Tensor) -> torch.Tensor:
    """Run encoder through PyTorch using the same wrapper as ONNX export."""
    from src.encoder_wrapper import EncoderWrapper

    wrapper = EncoderWrapper(model.thinker.audio_tower).eval()
    wrapper = wrapper.to(mel.device)
    with torch.no_grad():
        return wrapper(mel)


def greedy_decode_pytorch(
    model,
    audio_features: torch.Tensor,
    prompt_ids: list[int],
    max_tokens: int = 512,
    device: str = "cpu",
) -> list[int]:
    """
    Greedy decode using PyTorch model.

    Mirrors the ONNX decode logic for comparison.
    """
    with torch.no_grad():
        embed_weight = model.thinker.model.embed_tokens.weight.data

        # Build input embeddings
        ids_tensor = torch.tensor(prompt_ids, device=device, dtype=torch.long)
        input_embeds = embed_weight[ids_tensor]  # [seq_len, 1024]

        # Replace audio_pad positions
        audio_start, audio_end = get_audio_pad_range(prompt_ids)
        input_embeds[audio_start:audio_end] = audio_features[0].to(input_embeds.dtype)

        input_embeds = input_embeds.unsqueeze(0)  # [1, seq_len, 1024]
        position_ids = torch.arange(len(prompt_ids), device=device, dtype=torch.long).unsqueeze(0)

        # Tile for MRoPE
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Prefill
        seq_len = len(prompt_ids)
        cache_position = torch.arange(seq_len, device=device)

        outputs = model.thinker.model(
            inputs_embeds=input_embeds,
            position_ids=pos_3d,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        logits = model.thinker.lm_head(outputs.last_hidden_state)
        past_key_values = outputs.past_key_values

        next_token = int(logits[0, -1, :].argmax())
        output_tokens = [next_token]

        if next_token in EOS_TOKEN_IDS:
            return output_tokens

        # Autoregressive loop
        pos = len(prompt_ids)
        for _ in range(max_tokens - 1):
            token_embed = embed_weight[next_token].unsqueeze(0).unsqueeze(0)
            step_pos = torch.tensor([[pos]], device=device, dtype=torch.long)
            step_pos_3d = step_pos.unsqueeze(0).expand(3, -1, -1)
            step_cache_pos = torch.tensor([pos], device=device)

            outputs = model.thinker.model(
                inputs_embeds=token_embed,
                position_ids=step_pos_3d,
                past_key_values=past_key_values,
                cache_position=step_cache_pos,
                use_cache=True,
                return_dict=True,
            )
            logits = model.thinker.lm_head(outputs.last_hidden_state)
            past_key_values = outputs.past_key_values

            next_token = int(logits[0, -1, :].argmax())
            output_tokens.append(next_token)
            pos += 1

            if next_token in EOS_TOKEN_IDS:
                break

    return output_tokens


def validate_encoder(model, sessions, mel_np, mel_torch, device):
    """Compare encoder outputs between PyTorch and ONNX."""
    print("\n=== Encoder Validation ===")

    # PyTorch
    t0 = time.time()
    pt_features = run_encoder_pytorch(model, mel_torch).cpu().numpy()
    pt_time = time.time() - t0

    # ONNX
    t0 = time.time()
    onnx_features = run_encoder_onnx(sessions["encoder"], mel_np)
    onnx_time = time.time() - t0

    max_diff = np.max(np.abs(pt_features - onnx_features))
    mean_diff = np.mean(np.abs(pt_features - onnx_features))

    print(f"  PyTorch output shape: {pt_features.shape}")
    print(f"  ONNX output shape:    {onnx_features.shape}")
    print(f"  Max absolute diff:    {max_diff:.6e}")
    print(f"  Mean absolute diff:   {mean_diff:.6e}")
    print(f"  PyTorch time:         {pt_time:.3f}s")
    print(f"  ONNX time:            {onnx_time:.3f}s")

    if max_diff > 1e-4:
        print("  WARNING: Max diff exceeds 1e-4 threshold")
    else:
        print("  PASS: Encoder outputs match within tolerance")

    return pt_features, onnx_features


def validate_pipeline(model, sessions, embed_tokens, audio_features_pt, audio_features_onnx, tokenizer, device):
    """Compare full pipeline (encode -> decode) between PyTorch and ONNX."""
    print("\n=== Pipeline Validation ===")

    # Build prompt
    audio_token_count = audio_features_pt.shape[1]
    prompt_ids = build_prompt_ids(audio_token_count)
    print(f"  Prompt length: {len(prompt_ids)} tokens ({audio_token_count} audio tokens)")

    # PyTorch decode
    print("  Running PyTorch decode...")
    t0 = time.time()
    pt_tokens = greedy_decode_pytorch(
        model, audio_features_pt, prompt_ids, device=device
    )
    pt_time = time.time() - t0

    # ONNX decode
    print("  Running ONNX decode...")
    t0 = time.time()
    onnx_tokens = greedy_decode_onnx(
        sessions, embed_tokens, audio_features_onnx, prompt_ids
    )
    onnx_time = time.time() - t0

    # Decode to text
    pt_text = tokenizer.decode(pt_tokens, skip_special_tokens=True)
    onnx_text = tokenizer.decode(onnx_tokens, skip_special_tokens=True)

    print(f"\n  PyTorch text ({len(pt_tokens)} tokens, {pt_time:.2f}s):")
    print(f"    {pt_text}")
    print(f"\n  ONNX text ({len(onnx_tokens)} tokens, {onnx_time:.2f}s):")
    print(f"    {onnx_text}")

    # Compare token-by-token
    match_count = sum(
        1 for a, b in zip(pt_tokens, onnx_tokens) if a == b
    )
    total = max(len(pt_tokens), len(onnx_tokens))
    print(f"\n  Token match: {match_count}/{total}")

    if pt_tokens == onnx_tokens:
        print("  PASS: Exact token-for-token match")
    elif pt_text == onnx_text:
        print("  PASS: Text matches (minor token differences)")
    else:
        print("  WARN: Text differs between PyTorch and ONNX")
        # Show first divergence
        for i, (a, b) in enumerate(zip(pt_tokens, onnx_tokens)):
            if a != b:
                print(f"  First divergence at position {i}: PT={a} ONNX={b}")
                break


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX export")
    parser.add_argument("--onnx-dir", required=True, help="Directory with ONNX files")
    parser.add_argument("--audio", required=True, help="Path to test audio file (16kHz WAV)")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B", help="HuggingFace model ID")
    parser.add_argument("--device", default="cpu", help="Device for PyTorch model")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max decode tokens")
    args = parser.parse_args()

    # Load audio
    print("Loading audio...")
    audio, sr = sf.read(args.audio, dtype="float32")
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"  Audio: {len(audio)/16000:.1f}s, {len(audio)} samples")

    # Compute mel spectrogram
    print("Computing mel spectrogram...")
    mel_torch = log_mel_spectrogram(audio, device=args.device)
    mel_np = mel_torch.cpu().numpy()
    print(f"  Mel shape: {mel_np.shape}")

    # Load models
    print("\nLoading PyTorch model...")
    model = load_pytorch_model(args.model, device=args.device)

    print("\nLoading ONNX sessions...")
    sessions = load_onnx_sessions(args.onnx_dir)

    print("\nLoading embedding matrix...")
    embed_tokens = load_embed_tokens(args.onnx_dir)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Validate encoder
    if "encoder" in sessions:
        pt_features, onnx_features = validate_encoder(
            model, sessions, mel_np, mel_torch, args.device
        )
    else:
        print("\nSkipping encoder validation (no encoder.onnx)")
        pt_features = run_encoder_pytorch(model, mel_torch).cpu().numpy()
        onnx_features = pt_features

    # Validate pipeline
    if "decoder_init" in sessions and "decoder_step" in sessions:
        pt_features_torch = torch.from_numpy(pt_features).to(args.device)
        validate_pipeline(
            model, sessions, embed_tokens,
            pt_features_torch, onnx_features,
            tokenizer, args.device,
        )
    else:
        print("\nSkipping pipeline validation (missing decoder ONNX files)")


if __name__ == "__main__":
    main()
