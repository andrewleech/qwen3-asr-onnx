#!/usr/bin/env python3
"""
Validate ONNX exports against PyTorch reference.

Loads both the original PyTorch model and the exported ONNX files,
runs the same inputs through both, and compares outputs.

Usage:
    python validate.py --onnx-dir output/qwen3-asr-0.6b --audio tests/fixtures/test_audio.wav
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoModel

from src.embeddings import load_embed_tokens
from src.inference import greedy_decode_onnx
from src.mel import log_mel_spectrogram
from src.prompt import (
    EOS_TOKEN_IDS,
    build_prompt_ids,
    get_audio_pad_range,
    load_prompt_tokenizer,
)
from src.tokenizer_assets import validate_tokenizer_assets


def load_pytorch_model(model_id: str, device: str = "cpu"):
    """Load the PyTorch model for reference."""
    try:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
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
            attn_implementation="eager",
        )
    model.eval()
    return model


class FiniteOutputSession:
    def __init__(self, session, name):
        self.session = session
        self.name = name

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, output_names, input_feed):
        outputs = self.session.run(output_names, input_feed)
        for index, output in enumerate(outputs):
            if not np.isfinite(output).all():
                raise ValueError(f"{self.name}: non-finite ONNX output {index}")
        return outputs


def load_onnx_sessions(onnx_dir: str):
    names = ("encoder", "decoder_init", "decoder_step")
    missing = [f"{name}.onnx" for name in names if not os.path.isfile(os.path.join(onnx_dir, f"{name}.onnx"))]
    if missing:
        raise FileNotFoundError(f"Missing required ONNX sessions: {', '.join(missing)}")
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return {
        name: FiniteOutputSession(
            ort.InferenceSession(os.path.join(onnx_dir, f"{name}.onnx"), providers=providers), name
        )
        for name in names
    }


def run_encoder_onnx(session, mel: np.ndarray) -> np.ndarray:
    """Run encoder through ONNX Runtime."""
    result = session.run(["audio_features"], {"mel": mel})
    return result[0]  # type: ignore[no-any-return]


def run_encoder_pytorch(model, mel: torch.Tensor) -> torch.Tensor:
    """Run encoder through PyTorch using the same wrapper as ONNX export."""
    from src.encoder_wrapper import EncoderWrapper

    wrapper = EncoderWrapper(model.thinker.audio_tower).eval()
    wrapper = wrapper.to(mel.device)
    with torch.no_grad():
        return wrapper(mel)  # type: ignore[no-any-return]


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
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
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
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite PyTorch decoder logits")
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
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite PyTorch decoder logits")
            past_key_values = outputs.past_key_values

            next_token = int(logits[0, -1, :].argmax())
            output_tokens.append(next_token)
            pos += 1

            if next_token in EOS_TOKEN_IDS:
                break

    return output_tokens


def compare_features(reference, actual, label, *, atol=1e-4, rtol=0.0):
    if reference.shape != actual.shape or reference.ndim != 3 or not reference.size:
        raise ValueError(f"{label}: feature shape mismatch: {reference.shape} vs {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError(f"{label}: non-finite encoder features")
    error = np.abs(reference - actual)
    denominator = np.linalg.norm(reference.ravel())
    relative_l2 = (
        np.linalg.norm(error.ravel()) / denominator if denominator else (0.0 if not error.any() else float("inf"))
    )
    print(
        f"{label}: max absolute diff {error.max():.6e}, mean {error.mean():.6e}, relative L2 {relative_l2:.6e}, atol={atol}, rtol={rtol}"
    )
    if not np.allclose(actual, reference, atol=atol, rtol=rtol):
        raise ValueError(f"{label}: encoder features exceed atol={atol}, rtol={rtol}")
    print(f"PASS: {label}")


def validate_encoder(model, sessions, mel_np, mel_torch, device, *, atol=1e-4, rtol=0.0):
    from src.native_encoder import run_native_encoder

    with torch.no_grad():
        native_features = run_native_encoder(model.thinker.audio_tower, mel_torch).cpu().numpy()
    pt_features = run_encoder_pytorch(model, mel_torch).cpu().numpy()
    onnx_features = run_encoder_onnx(sessions["encoder"], mel_np)
    compare_features(native_features, pt_features, "native vs wrapper", atol=atol, rtol=rtol)
    compare_features(pt_features, onnx_features, "wrapper vs ONNX", atol=atol, rtol=rtol)
    compare_features(native_features, onnx_features, "native vs ONNX", atol=atol, rtol=rtol)
    return native_features, onnx_features


def validate_pipeline(
    model,
    sessions,
    embed_tokens,
    audio_features_pt,
    audio_features_onnx,
    tokenizer,
    device,
    max_tokens=256,
    onnx_tokenizer=None,
):
    """Compare full pipeline (encode -> decode) between PyTorch and ONNX."""
    print("\n=== Pipeline Validation ===")

    # Build prompt
    audio_token_count = audio_features_pt.shape[1]
    prompt_ids = build_prompt_ids(audio_token_count, tokenizer)
    artifact_tokenizer = tokenizer if onnx_tokenizer is None else onnx_tokenizer
    artifact_prompt = build_prompt_ids(audio_token_count, artifact_tokenizer)
    if artifact_prompt != prompt_ids:
        raise ValueError("Artifact prompt IDs do not match the checkpoint")
    print(f"  Prompt length: {len(prompt_ids)} tokens ({audio_token_count} audio tokens)")

    # PyTorch decode
    print("  Running PyTorch decode...")
    t0 = time.time()
    pt_tokens = greedy_decode_pytorch(model, audio_features_pt, prompt_ids, max_tokens=max_tokens, device=device)
    pt_time = time.time() - t0

    # ONNX decode
    print("  Running ONNX decode...")
    t0 = time.time()
    onnx_tokens = greedy_decode_onnx(
        sessions, embed_tokens, audio_features_onnx, artifact_prompt, max_tokens=max_tokens
    )
    onnx_time = time.time() - t0

    # Decode to text
    pt_text = tokenizer.decode(pt_tokens, skip_special_tokens=True)
    onnx_text = artifact_tokenizer.decode(onnx_tokens, skip_special_tokens=True)

    print(f"\n  PyTorch text ({len(pt_tokens)} tokens, {pt_time:.2f}s):")
    print(f"    {pt_text}")
    print(f"\n  ONNX text ({len(onnx_tokens)} tokens, {onnx_time:.2f}s):")
    print(f"    {onnx_text}")

    # Compare token-by-token
    match_count = sum(1 for a, b in zip(pt_tokens, onnx_tokens) if a == b)
    total = max(len(pt_tokens), len(onnx_tokens))
    print(f"\n  Token match: {match_count}/{total}")

    if pt_text != onnx_text:
        raise ValueError(f"Transcript mismatch: PyTorch={pt_text!r}, ONNX={onnx_text!r}")
    if pt_tokens == onnx_tokens:
        print("  PASS: Exact token-for-token match")
    else:
        print("  PASS: Text matches (minor token differences)")


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX export")
    parser.add_argument("--onnx-dir", required=True, help="Directory with ONNX files")
    parser.add_argument("--audio", required=True, help="Path to test audio file (16kHz WAV)")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B", help="HuggingFace model ID")
    parser.add_argument("--device", default="cpu", help="Device for PyTorch model")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max decode tokens")
    parser.add_argument("--atol", type=float, default=1e-4, help="Encoder absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.0, help="Encoder relative tolerance")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if not np.isfinite(args.atol) or not np.isfinite(args.rtol) or args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be finite and nonnegative")

    print("\nLoading ONNX sessions...")
    sessions = load_onnx_sessions(args.onnx_dir)
    artifact_tokenizer = validate_tokenizer_assets(args.onnx_dir)

    # Load audio
    print("Loading audio...")
    audio, sr = sf.read(args.audio, dtype="float32")
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"  Audio: {len(audio) / 16000:.1f}s, {len(audio)} samples")

    # Compute mel spectrogram
    print("Computing mel spectrogram...")
    mel_torch = log_mel_spectrogram(audio, device=args.device)
    mel_np = mel_torch.cpu().numpy()
    print(f"  Mel shape: {mel_np.shape}")

    # Load models
    print("\nLoading PyTorch model...")
    model = load_pytorch_model(args.model, device=args.device)

    print("\nLoading embedding matrix...")
    embed_tokens = load_embed_tokens(args.onnx_dir)

    print("\nLoading tokenizer...")
    tokenizer = load_prompt_tokenizer(args.model)

    pt_features, onnx_features = validate_encoder(
        model,
        sessions,
        mel_np,
        mel_torch,
        args.device,
        atol=args.atol,
        rtol=args.rtol,
    )
    validate_pipeline(
        model,
        sessions,
        embed_tokens,
        torch.from_numpy(pt_features).to(args.device),
        onnx_features,
        tokenizer,
        args.device,
        max_tokens=args.max_tokens,
        onnx_tokenizer=artifact_tokenizer,
    )


if __name__ == "__main__":
    main()
