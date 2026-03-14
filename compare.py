#!/usr/bin/env python3
"""
Compare ASR transcription across four inference paths:

1. Native  — qwen-asr processor + model.generate() (the reference implementation)
2. Wrapper — our EncoderWrapper + greedy_decode_pytorch (PyTorch, same arch as ONNX)
3. FP32    — ONNX Runtime with FP32 models
4. INT8    — ONNX Runtime with INT8-quantized models

Runs each path on one or more audio files and reports:
- Transcription text from each path
- Token-level agreement between paths
- Encoder feature numerical differences (where applicable)
"""

import argparse
import json
import os
import time
from textwrap import indent

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Path 1: Native qwen-asr inference
# ---------------------------------------------------------------------------

def run_native(model, processor, audio: np.ndarray, sr: int = 16000) -> dict:
    """Run native qwen-asr inference (processor + model.generate)."""
    from src.prompt import EOS_TOKEN_IDS

    # Build the text prompt using the chat template
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    text_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(
        text=[text_prompt],
        audio=[audio],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    elapsed = time.time() - t0

    # output is GenerateDecoderOnlyOutput; get the sequence tensor
    sequences = output.sequences if hasattr(output, "sequences") else output
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = sequences[0][prompt_len:].tolist()

    # Remove EOS tokens from the end
    while gen_ids and gen_ids[-1] in EOS_TOKEN_IDS:
        gen_ids.pop()

    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Count audio tokens the native processor allocated
    audio_pad_count = int((inputs["input_ids"] == 151676).sum().item())
    mel_frames = int(inputs["feature_attention_mask"].sum(-1).item())

    return {
        "text": text,
        "tokens": gen_ids,
        "time": elapsed,
        "mel_frames": mel_frames,
        "audio_token_count": audio_pad_count,
    }


# ---------------------------------------------------------------------------
# Path 2: PyTorch wrapper (same architecture as ONNX export)
# ---------------------------------------------------------------------------

def run_wrapper_pytorch(model, audio: np.ndarray, tokenizer, device: str = "cpu") -> dict:
    """Run our PyTorch wrappers (EncoderWrapper + greedy_decode_pytorch)."""
    from src.encoder_wrapper import EncoderWrapper
    from src.mel import log_mel_spectrogram
    from src.prompt import build_prompt_ids, EOS_TOKEN_IDS

    # Mel spectrogram
    mel_torch = log_mel_spectrogram(audio, device=device)

    # Encoder
    wrapper = EncoderWrapper(model.thinker.audio_tower).eval().to(device)
    with torch.no_grad():
        audio_features = wrapper(mel_torch)

    # Build prompt and decode
    audio_token_count = audio_features.shape[1]
    prompt_ids = build_prompt_ids(audio_token_count)

    from validate import greedy_decode_pytorch
    t0 = time.time()
    tokens = greedy_decode_pytorch(model, audio_features, prompt_ids, max_tokens=256, device=device)
    elapsed = time.time() - t0

    # Strip EOS
    while tokens and tokens[-1] in EOS_TOKEN_IDS:
        tokens.pop()

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return {
        "text": text,
        "tokens": tokens,
        "time": elapsed,
        "audio_features": audio_features.cpu().numpy(),
        "mel_frames": mel_torch.shape[2],
        "audio_token_count": audio_token_count,
    }


# ---------------------------------------------------------------------------
# Paths 3 & 4: ONNX Runtime (FP32 or INT8)
# ---------------------------------------------------------------------------

def load_onnx_sessions(onnx_dir: str) -> dict:
    """Load ONNX runtime sessions for split decoder architecture."""
    sessions = {}
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    for name in ["encoder", "decoder_init", "decoder_step"]:
        path = os.path.join(onnx_dir, f"{name}.onnx")
        if os.path.exists(path):
            sessions[name] = ort.InferenceSession(path, providers=providers)
    return sessions


def run_onnx(sessions, embed_tokens, audio: np.ndarray, tokenizer, label: str) -> dict:
    """Run full ONNX pipeline with split decoder (decoder_init + decoder_step)."""
    from src.inference import greedy_decode_onnx
    from src.mel import log_mel_spectrogram
    from src.prompt import build_prompt_ids, EOS_TOKEN_IDS

    # Mel
    mel_torch = log_mel_spectrogram(audio)
    mel_np = mel_torch.cpu().numpy()

    # Encoder
    t0 = time.time()
    audio_features = sessions["encoder"].run(["audio_features"], {"mel": mel_np})[0]
    enc_time = time.time() - t0

    # Prompt + decode
    audio_token_count = audio_features.shape[1]
    prompt_ids = build_prompt_ids(audio_token_count)

    t0 = time.time()
    tokens = greedy_decode_onnx(sessions, embed_tokens, audio_features, prompt_ids, max_tokens=256)
    dec_time = time.time() - t0

    # Strip EOS
    while tokens and tokens[-1] in EOS_TOKEN_IDS:
        tokens.pop()

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return {
        "text": text,
        "tokens": tokens,
        "time": enc_time + dec_time,
        "audio_features": audio_features,
        "mel_frames": mel_torch.shape[2],
        "audio_token_count": audio_token_count,
    }


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def strip_asr_prefix(text: str) -> str:
    """Strip 'language English<asr_text>' prefix from model output."""
    import re
    # Remove "language X<asr_text>" or just "<asr_text>" prefix
    text = re.sub(r"^language\s+\w+<asr_text>", "", text)
    text = re.sub(r"^<asr_text>", "", text)
    return text.strip()


def compare_texts(results: dict[str, dict], ground_truth: str | None):
    """Print text comparison table."""
    if ground_truth:
        print(f"  Ground truth: {ground_truth}")
    for name, r in results.items():
        raw = r["text"]
        clean = strip_asr_prefix(raw)
        marker = ""
        if ground_truth and clean.upper().replace(",", "").replace(".", "").strip() == ground_truth.upper().strip():
            marker = " [EXACT MATCH]"
        print(f"  {name:12s}: {clean}{marker}")


def compare_tokens(results: dict[str, dict]):
    """Compare token sequences pairwise."""
    names = list(results.keys())
    ref_name = names[0]
    ref_tokens = results[ref_name]["tokens"]

    for name in names[1:]:
        other = results[name]["tokens"]
        match = sum(1 for a, b in zip(ref_tokens, other) if a == b)
        total = max(len(ref_tokens), len(other))
        exact = ref_tokens == other
        status = "EXACT" if exact else f"{match}/{total} match"
        print(f"  {ref_name} vs {name}: {status} (len {len(ref_tokens)} vs {len(other)})")


def compare_encoder_features(results: dict[str, dict]):
    """Compare encoder feature tensors where available."""
    pairs = []
    names_with_features = [n for n, r in results.items() if "audio_features" in r]
    if len(names_with_features) < 2:
        return

    for i in range(len(names_with_features)):
        for j in range(i + 1, len(names_with_features)):
            a_name = names_with_features[i]
            b_name = names_with_features[j]
            a = results[a_name]["audio_features"]
            b = results[b_name]["audio_features"]
            if a.shape != b.shape:
                print(f"  {a_name} vs {b_name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            else:
                max_diff = np.max(np.abs(a - b))
                mean_diff = np.mean(np.abs(a - b))
                print(f"  {a_name} vs {b_name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare ASR inference across 4 paths")
    parser.add_argument("--audio", nargs="+", required=True, help="Audio WAV files")
    parser.add_argument("--ground-truth", nargs="*", default=None,
                        help="Ground truth transcriptions (one per audio file)")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--fp32-dir", default="output/qwen3-asr-0.6b")
    parser.add_argument("--int8-dir", default="output/qwen3-asr-0.6b-int8")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ground_truths = args.ground_truth or [None] * len(args.audio)
    if len(ground_truths) != len(args.audio):
        parser.error("Number of --ground-truth must match number of --audio files")

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print("Loading PyTorch model...")
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            args.model, torch_dtype=torch.float32,
            device_map=args.device, trust_remote_code=True,
        )
    except Exception:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration,
        )
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.float32, device_map=args.device,
        )
    model.eval()

    print("Loading processor...")
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor
    processor = Qwen3ASRProcessor.from_pretrained(args.model)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading FP32 ONNX sessions...")
    fp32_sessions = load_onnx_sessions(args.fp32_dir)

    print("Loading INT8 ONNX sessions...")
    int8_sessions = load_onnx_sessions(args.int8_dir)

    # Load embedding matrices
    def load_embed(onnx_dir):
        with open(os.path.join(onnx_dir, "config.json")) as f:
            cfg = json.load(f)
        dtype = np.dtype(cfg.get("embed_tokens_dtype", "float32"))
        embed = np.fromfile(
            os.path.join(onnx_dir, "embed_tokens.bin"), dtype=dtype
        ).reshape(cfg["embed_tokens_shape"])
        # Always work in float32
        return embed.astype(np.float32)

    fp32_embed = load_embed(args.fp32_dir)
    int8_embed = load_embed(args.int8_dir)

    # -----------------------------------------------------------------------
    # Run comparisons
    # -----------------------------------------------------------------------
    for audio_path, gt in zip(args.audio, ground_truths):
        print(f"\n{'='*72}")
        print(f"Audio: {audio_path}")
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        print(f"Duration: {len(audio)/16000:.1f}s  ({len(audio)} samples)")
        print(f"{'='*72}")

        results = {}

        # Path 1: Native
        print("\n  Running native inference...")
        results["native"] = run_native(model, processor, audio, sr=16000)

        # Path 2: Wrapper PyTorch
        print("  Running wrapper PyTorch...")
        results["wrapper"] = run_wrapper_pytorch(model, audio, tokenizer, device=args.device)

        # Path 3: FP32 ONNX
        print("  Running FP32 ONNX...")
        results["fp32"] = run_onnx(fp32_sessions, fp32_embed, audio, tokenizer, "FP32")

        # Path 4: INT8 ONNX
        print("  Running INT8 ONNX...")
        results["int8"] = run_onnx(int8_sessions, int8_embed, audio, tokenizer, "INT8")

        # --- Report ---
        print(f"\n--- Transcriptions ---")
        compare_texts(results, gt)

        print(f"\n--- Timing ---")
        for name, r in results.items():
            print(f"  {name:12s}: {r['time']:.2f}s")

        print(f"\n--- Token Comparison ---")
        compare_tokens(results)

        # Also compare non-native paths pairwise
        non_native = {k: v for k, v in results.items() if k != "native"}
        if len(non_native) >= 2:
            names = list(non_native.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    ta, tb = non_native[a]["tokens"], non_native[b]["tokens"]
                    match = sum(1 for x, y in zip(ta, tb) if x == y)
                    total = max(len(ta), len(tb))
                    exact = ta == tb
                    status = "EXACT" if exact else f"{match}/{total} match"
                    print(f"  {a} vs {b}: {status}")

        print(f"\n--- Encoder Features ---")
        compare_encoder_features(results)

        # Shapes info
        print(f"\n--- Audio Token Counts ---")
        for name, r in results.items():
            if "mel_frames" in r:
                print(f"  {name:12s}: mel_frames={r['mel_frames']}, audio_tokens={r['audio_token_count']}")

    print()


if __name__ == "__main__":
    main()
