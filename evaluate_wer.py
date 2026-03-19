#!/usr/bin/env python3
"""
WER evaluation of Qwen3-ASR ONNX models across benchmark datasets.

Streams samples from HuggingFace datasets (no full download), runs ONNX
inference for each specified model directory, and reports WER per dataset
per model.

Datasets:
  librispeech-other  openslr/librispeech_asr test.other  (clean, harder speakers)
  ami-sdm            edinburghcstr/ami SDM test           (far-field meeting speech)

Usage:
    uv run python evaluate_wer.py \\
        --models "0.6B FP32:output/qwen3-asr-0.6b" \\
                 "0.6B INT8:output/qwen3-asr-0.6b:int8" \\
                 "0.6B int4:output/qwen3-asr-0.6b:int4" \\
                 "1.7B FP32:output/qwen3-asr-1.7b" \\
                 "1.7B int4:output/qwen3-asr-1.7b:int4" \\
        --datasets librispeech-other ami-sdm \\
        --n-samples 200
"""

import argparse
import io
import json
import os
import re
import string
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import onnxruntime as ort
import soundfile as sf
from datasets import load_dataset, Audio


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Normalize text for WER computation.

    - Strip Qwen3-ASR language prefix ("language English<asr_text>")
    - Lowercase
    - Remove punctuation except apostrophes (preserves contractions)
    - Collapse whitespace
    """
    # Strip Qwen3-ASR output prefix: "language <Name><asr_text>" (normal) or
    # "language <name> asr text" (degraded INT8 variant without the special token)
    text = re.sub(r"^language\s+\w+(?:<asr_text>|[\s\n]+asr\s+text\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^<asr_text>", "", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation except apostrophes
    keep = set(string.ascii_lowercase + string.digits + "' ")
    text = "".join(c if c in keep else " " for c in text)
    # Collapse whitespace
    return " ".join(text.split())


def wer(references: list[str], hypotheses: list[str]) -> float:
    """
    Compute word error rate.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=reference words.
    Uses dynamic programming (Wagner-Fischer).
    """
    total_errors = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        n = len(ref_words)
        m = len(hyp_words)
        total_words += n

        if n == 0:
            total_errors += m
            continue

        # DP table
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            new_dp = [i] + [0] * m
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    new_dp[j] = dp[j - 1]
                else:
                    new_dp[j] = 1 + min(dp[j - 1], dp[j], new_dp[j - 1])
            dp = new_dp
        total_errors += dp[m]

    if total_words == 0:
        return 0.0
    return total_errors / total_words


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def _resolve_model_path(model_dir: str, name: str, quant: str | None = None) -> str:
    """Resolve model file with quantization suffix, falling back to FP32.

    Looks for {name}.{quant}.onnx, falls back to {name}.onnx if not found.
    Matches the Rust resolve_model_path logic in transcribe-rs.
    """
    if quant:
        path = os.path.join(model_dir, f"{name}.{quant}.onnx")
        if os.path.exists(path):
            return path
    # FP32 fallback
    path = os.path.join(model_dir, f"{name}.onnx")
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"No model found for {name} (quant={quant}) in {model_dir}")


def load_sessions(model_dir: str, quant: str | None = None) -> dict:
    """Load encoder, decoder_init, decoder_step from a model directory.

    quant: optional quantization suffix ("int4", "int8", "fp16") for suffixed files.
    Falls back through int8 → FP32 if the requested variant isn't found.
    """
    sessions = {}
    providers = ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    for name in ["encoder", "decoder_init", "decoder_step"]:
        path = _resolve_model_path(model_dir, name, quant)
        print(f"    {name}: {os.path.basename(path)}")
        sessions[name] = ort.InferenceSession(path, opts, providers=providers)
    return sessions


def load_embed(model_dir: str) -> np.ndarray:
    """Load embed_tokens.bin, converting FP16 → FP32 if needed."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    # v2+ format: embed_tokens_shape removed, derive from decoder config
    if "embed_tokens_shape" in cfg:
        shape = cfg["embed_tokens_shape"]
    else:
        shape = [cfg["decoder"]["vocab_size"], cfg["decoder"]["hidden_size"]]
    # embed_tokens_dtype applies regardless of whether embed_tokens_shape is present
    dtype_str = cfg.get("embed_tokens_dtype", "float32")
    dtype = np.float16 if dtype_str == "float16" else np.float32
    embed = np.fromfile(os.path.join(model_dir, "embed_tokens.bin"), dtype=dtype).reshape(shape)
    return embed.astype(np.float32)


# Tokenizer cache — load once per model dir
_tokenizer_cache: dict = {}


def get_tokenizer(model_dir: str):
    if model_dir not in _tokenizer_cache:
        from transformers import AutoTokenizer
        _tokenizer_cache[model_dir] = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
    return _tokenizer_cache[model_dir]


# ---------------------------------------------------------------------------
# Dataset streaming
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "librispeech-other": {
        "hf_id": "openslr/librispeech_asr",
        "config": "all",
        "split": "test.other",
        "text_field": "text",
        "label": "LibriSpeech test-other",
    },
    "ami-sdm": {
        "hf_id": "edinburghcstr/ami",
        "config": "sdm",
        "split": "test",
        "text_field": "text",
        "label": "AMI SDM test",
    },
}


def stream_samples(dataset_key: str, n: int) -> Iterator[tuple[np.ndarray, str]]:
    """Yield (audio_f32_16khz, reference_text) tuples."""
    cfg = DATASET_CONFIGS[dataset_key]
    ds = load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"], streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    text_field = cfg["text_field"]
    count = 0
    for sample in ds:
        if count >= n:
            break

        # Decode audio bytes → float32 numpy at native sample rate
        raw = sample["audio"]["bytes"]
        if raw is None:
            continue
        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            continue

        # Stereo → mono
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

        ref = sample[text_field]
        if not ref or not ref.strip():
            continue

        yield arr.astype(np.float32), ref.strip()
        count += 1


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    name: str
    model_dir: str
    # per-dataset: list of (ref_normalized, hyp_normalized)
    pairs: dict = field(default_factory=dict)


def evaluate(model_dirs: list[tuple[str, str, str | None]], dataset_keys: list[str], n_samples: int, output_path: str | None):
    """
    Main evaluation loop.

    model_dirs: list of (name, path, quant) tuples
    """
    # Load all models upfront
    print("Loading models...")
    models = []
    for name, model_dir, quant in model_dirs:
        quant_label = f" ({quant})" if quant else ""
        print(f"  {name}: {model_dir}{quant_label}")
        sessions = load_sessions(model_dir, quant)
        embed = load_embed(model_dir)
        tokenizer = get_tokenizer(model_dir)
        models.append((name, model_dir, sessions, embed, tokenizer))
    print()

    results: dict[str, dict[str, list]] = {name: {} for name, *_ in models}
    infer_times: dict[str, float] = {name: 0.0 for name, *_ in models}
    total_audio_secs: float = 0.0

    for dataset_key in dataset_keys:
        cfg = DATASET_CONFIGS[dataset_key]
        label = cfg["label"]
        print(f"{'='*72}")
        print(f"Dataset: {label}  (n={n_samples})")
        print(f"{'='*72}")

        for name, *_ in models:
            results[name][dataset_key] = []  # list of (ref, hyp) normalized pairs

        sample_count = 0
        t_start = time.time()

        for audio, ref_raw in stream_samples(dataset_key, n_samples):
            ref_norm = normalize(ref_raw)
            if not ref_norm:
                continue

            sample_count += 1
            audio_secs = len(audio) / 16000.0
            total_audio_secs += audio_secs
            elapsed = time.time() - t_start
            print(f"  [{sample_count:4d}/{n_samples}] {elapsed:6.0f}s  ref: {ref_norm[:60]}")

            for name, model_dir, sessions, embed, tokenizer in models:
                try:
                    t0 = time.time()
                    hyp_raw = _run(sessions, embed, tokenizer, audio)
                    infer_times[name] += time.time() - t0
                    hyp_norm = normalize(hyp_raw)
                except Exception as e:
                    print(f"    {name}: ERROR {e}")
                    hyp_norm = ""
                results[name][dataset_key].append((ref_norm, hyp_norm))
                # Print current running WER
                refs = [r for r, _ in results[name][dataset_key]]
                hyps = [h for _, h in results[name][dataset_key]]
                current_wer = wer(refs, hyps) * 100
                print(f"    {name}: {hyp_norm[:50]}  [WER so far: {current_wer:.1f}%]")

        print()

    # Summary table
    print(f"\n{'='*72}")
    print(f"{'SUMMARY':^72}")
    print(f"{'='*72}")
    col_w = max(len(n) for n, *_ in models) + 2
    header = f"{'Dataset':<28} {'Model':<{col_w}} {'Samples':>8} {'WER%':>7} {'Time':>8} {'RTF':>7}"
    print(header)
    print("-" * len(header))

    summary = []
    for dataset_key in dataset_keys:
        label = DATASET_CONFIGS[dataset_key]["label"]
        for name, *_ in models:
            pairs = results[name].get(dataset_key, [])
            if not pairs:
                continue
            refs = [r for r, _ in pairs]
            hyps = [h for _, h in pairs]
            w = wer(refs, hyps) * 100
            n = len(pairs)
            t = infer_times[name]
            rtf = t / total_audio_secs if total_audio_secs > 0 else 0
            print(f"{label:<28} {name:<{col_w}} {n:>8} {w:>7.2f} {t:>7.1f}s {rtf:>6.2f}x")
            summary.append({"dataset": label, "model": name, "n": n, "wer": round(w, 4),
                            "infer_secs": round(t, 1), "audio_secs": round(total_audio_secs, 1),
                            "rtf": round(rtf, 3)})

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {output_path}")


def _run(sessions: dict, embed_tokens: np.ndarray, tokenizer, audio: np.ndarray) -> str:
    """Run full ONNX pipeline, return raw decoded text."""
    from src.inference import greedy_decode_onnx
    from src.mel import log_mel_spectrogram
    from src.prompt import build_prompt_ids, EOS_TOKEN_IDS

    mel = log_mel_spectrogram(audio)
    mel_np = mel.cpu().numpy()

    audio_features = sessions["encoder"].run(["audio_features"], {"mel": mel_np})[0]
    audio_token_count = audio_features.shape[1]

    if audio_token_count == 0:
        return ""

    prompt_ids = build_prompt_ids(audio_token_count)
    tokens = greedy_decode_onnx(sessions, embed_tokens, audio_features, prompt_ids, max_tokens=256)

    while tokens and tokens[-1] in EOS_TOKEN_IDS:
        tokens.pop()

    return tokenizer.decode(tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WER of Qwen3-ASR ONNX models on benchmark datasets"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME:DIR",
        help="Model variants as 'name:path' pairs, e.g. '0.6B FP32:output/qwen3-asr-0.6b'",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["librispeech-other", "ami-sdm"],
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate on (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of samples per dataset (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results (optional)",
    )
    args = parser.parse_args()

    model_dirs = []
    for spec in args.models:
        if ":" not in spec:
            parser.error(f"Model spec must be 'name:path[:quant]', got: {spec!r}")
        parts = spec.split(":")
        name, path = parts[0], parts[1]
        quant = parts[2] if len(parts) > 2 else None
        if not os.path.isdir(path):
            parser.error(f"Model directory not found: {path}")
        model_dirs.append((name, path, quant))

    evaluate(model_dirs, args.datasets, args.n_samples, args.output)


if __name__ == "__main__":
    main()
