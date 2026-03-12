#!/usr/bin/env python3
"""
WER evaluation of Parakeet-TDT ONNX models on the same benchmark datasets
used for Qwen3-ASR, for cross-engine comparison.

Parakeet-TDT uses a transducer architecture:
  nemo128.onnx (mel preprocessor) → encoder → decoder_joint (greedy RNN-T)

Usage:
    uv run python evaluate_wer_parakeet.py \
        --model-dir /path/to/parakeet-tdt-0.6b-v3-int8 \
        --datasets librispeech-other \
        --n-samples 200 --output parakeet_wer.json
"""

import argparse
import io
import json
import os
import re
import string
import sys
import time
from typing import Iterator

import numpy as np
import onnxruntime as ort
import soundfile as sf
from datasets import load_dataset, Audio


# ---------------------------------------------------------------------------
# Text normalization (shared with evaluate_wer.py)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Normalize text for WER computation.

    - Lowercase
    - Remove punctuation except apostrophes (preserves contractions)
    - Collapse whitespace
    """
    text = text.lower()
    keep = set(string.ascii_lowercase + string.digits + "' ")
    text = "".join(c if c in keep else " " for c in text)
    return " ".join(text.split())


def wer(references: list[str], hypotheses: list[str]) -> float:
    """
    Compute word error rate using Wagner-Fischer DP.
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
# Parakeet ONNX model
# ---------------------------------------------------------------------------

SUBSAMPLING_FACTOR = 8
WINDOW_SIZE = 0.01
MAX_TOKENS_PER_STEP = 10

# SentencePiece word boundary
SP_SPACE = "\u2581"

# Regex matching Rust's: r"\A\s|\s\B|(\s)\b"
DECODE_SPACE_RE = re.compile(r"\A\s|\s\B|(\s)\b")


def load_vocab(model_dir: str) -> tuple[list[str], int]:
    """Load vocab.txt, return (vocab_list, blank_idx)."""
    vocab_path = os.path.join(model_dir, "vocab.txt")
    tokens_with_ids = []
    blank_idx = None
    max_id = 0

    with open(vocab_path) as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) >= 2:
                token = parts[0]
                try:
                    tid = int(parts[1])
                except ValueError:
                    continue
                if token == "<blk>":
                    blank_idx = tid
                tokens_with_ids.append((token, tid))
                max_id = max(max_id, tid)

    vocab = [""] * (max_id + 1)
    for token, tid in tokens_with_ids:
        vocab[tid] = token.replace(SP_SPACE, " ")

    if blank_idx is None:
        raise ValueError("Missing <blk> token in vocabulary")
    return vocab, blank_idx


def load_parakeet(model_dir: str) -> dict:
    """Load Parakeet ONNX sessions + vocab."""
    providers = ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sessions = {}
    for name, try_int8 in [("nemo128", False), ("encoder-model", True), ("decoder_joint-model", True)]:
        if try_int8:
            int8_path = os.path.join(model_dir, f"{name}.int8.onnx")
            fp32_path = os.path.join(model_dir, f"{name}.onnx")
            path = int8_path if os.path.exists(int8_path) else fp32_path
        else:
            path = os.path.join(model_dir, f"{name}.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        sessions[name] = ort.InferenceSession(path, opts, providers=providers)

    vocab, blank_idx = load_vocab(model_dir)

    # Get decoder state shapes from model inputs
    decoder = sessions["decoder_joint-model"]
    state_shapes = {}
    for inp in decoder.get_inputs():
        if inp.name.startswith("input_states_"):
            # Shape like [2, -1, 640] — replace -1 with 1 for batch
            shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]
            state_shapes[inp.name] = shape

    return {
        "preprocessor": sessions["nemo128"],
        "encoder": sessions["encoder-model"],
        "decoder": sessions["decoder_joint-model"],
        "vocab": vocab,
        "blank_idx": blank_idx,
        "vocab_size": len(vocab),
        "state_shapes": state_shapes,
    }


def decode_tokens(vocab: list[str], token_ids: list[int]) -> str:
    """Convert token IDs to text, matching Rust decode_tokens logic."""
    tokens = []
    for tid in token_ids:
        if 0 <= tid < len(vocab):
            tokens.append(vocab[tid])

    raw = "".join(tokens)

    # Apply the same regex as Rust: strip leading space, handle word boundaries
    def replacer(m):
        if m.group(1) is not None:
            return " "
        return ""

    return DECODE_SPACE_RE.sub(replacer, raw)


def transcribe_parakeet(model: dict, audio: np.ndarray) -> str:
    """Run full Parakeet pipeline on float32 16kHz audio, return text."""
    # Preprocessor: audio → mel features
    waveforms = audio.reshape(1, -1).astype(np.float32)
    waveforms_lens = np.array([len(audio)], dtype=np.int64)

    features, features_lens = model["preprocessor"].run(
        ["features", "features_lens"],
        {"waveforms": waveforms, "waveforms_lens": waveforms_lens},
    )

    # Encoder: features → encoded
    encoder_out, encoded_lengths = model["encoder"].run(
        ["outputs", "encoded_lengths"],
        {"audio_signal": features, "length": features_lens},
    )

    # Encoder output is [batch, features, time] → transpose to [batch, time, features]
    encoder_out = np.transpose(encoder_out, (0, 2, 1))

    encodings_len = int(encoded_lengths[0])
    encodings = encoder_out[0]  # [time, 1024]

    # Greedy TDT decode
    blank_idx = model["blank_idx"]
    vocab_size = model["vocab_size"]

    # Initialize decoder states
    state1 = np.zeros(model["state_shapes"]["input_states_1"], dtype=np.float32)
    state2 = np.zeros(model["state_shapes"]["input_states_2"], dtype=np.float32)

    tokens = []
    t = 0
    emitted_tokens = 0

    while t < encodings_len:
        encoder_step = encodings[t]  # [1024]
        # Shape: [1, time_steps, 1] — single step
        enc_input = encoder_step.reshape(1, -1, 1)

        target_token = tokens[-1] if tokens else blank_idx
        targets = np.array([[target_token]], dtype=np.int32)
        target_length = np.array([1], dtype=np.int32)

        logits, new_state1, new_state2 = model["decoder"].run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": enc_input,
                "targets": targets,
                "target_length": target_length,
                "input_states_1": state1,
                "input_states_2": state2,
            },
        )

        # Flatten all batch dims: logits may be (1, 1, 1, N) → (N,)
        logits_1d = logits.flatten()

        # TDT: split vocab logits from duration logits
        if len(logits_1d) > vocab_size:
            vocab_logits = logits_1d[:vocab_size]
        else:
            vocab_logits = logits_1d

        token = int(np.argmax(vocab_logits))

        if token != blank_idx:
            # Only update RNN state on non-blank emissions (matches Rust impl)
            state1 = new_state1
            state2 = new_state2
            tokens.append(token)
            emitted_tokens += 1

        if token == blank_idx or emitted_tokens >= MAX_TOKENS_PER_STEP:
            t += 1
            emitted_tokens = 0

    return decode_tokens(model["vocab"], tokens)


# ---------------------------------------------------------------------------
# Dataset streaming (shared with evaluate_wer.py)
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

        raw = sample["audio"]["bytes"]
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

        ref = sample[text_field]
        if not ref or not ref.strip():
            continue

        yield arr.astype(np.float32), ref.strip()
        count += 1


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model_dirs: list[tuple[str, str]], dataset_keys: list[str],
             n_samples: int, output_path: str | None):
    print("Loading models...")
    models = []
    for name, model_dir in model_dirs:
        print(f"  {name}: {model_dir}")
        model = load_parakeet(model_dir)
        models.append((name, model_dir, model))
    print()

    results: dict[str, dict[str, list]] = {name: {} for name, *_ in models}

    for dataset_key in dataset_keys:
        cfg = DATASET_CONFIGS[dataset_key]
        label = cfg["label"]
        print(f"{'='*72}")
        print(f"Dataset: {label}  (n={n_samples})")
        print(f"{'='*72}")

        for name, *_ in models:
            results[name][dataset_key] = []

        sample_count = 0
        t_start = time.time()

        for audio, ref_raw in stream_samples(dataset_key, n_samples):
            ref_norm = normalize(ref_raw)
            if not ref_norm:
                continue

            sample_count += 1
            elapsed = time.time() - t_start
            print(f"  [{sample_count:4d}/{n_samples}] {elapsed:6.0f}s  ref: {ref_norm[:60]}")

            for name, model_dir, model in models:
                try:
                    hyp_raw = transcribe_parakeet(model, audio)
                    hyp_norm = normalize(hyp_raw)
                except Exception as e:
                    print(f"    {name}: ERROR {e}")
                    hyp_norm = ""
                results[name][dataset_key].append((ref_norm, hyp_norm))
                refs = [r for r, _ in results[name][dataset_key]]
                hyps = [h for _, h in results[name][dataset_key]]
                current_wer = wer(refs, hyps) * 100
                print(f"    {name}: {hyp_norm[:50]}  [WER so far: {current_wer:.1f}%]")

        print()

    # Summary
    print(f"\n{'='*72}")
    print(f"{'SUMMARY':^72}")
    print(f"{'='*72}")
    col_w = max(len(n) for n, *_ in models) + 2
    header = f"{'Dataset':<28} {'Model':<{col_w}} {'Samples':>8} {'WER%':>7}"
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
            print(f"{label:<28} {name:<{col_w}} {n:>8} {w:>7.2f}")
            summary.append({"dataset": label, "model": name, "n": n, "wer": round(w, 4)})

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WER of Parakeet-TDT ONNX models on benchmark datasets"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME:DIR",
        help="Model variants as 'name:path' pairs, e.g. 'Parakeet INT8:/path/to/model'",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["librispeech-other"],
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model_dirs = []
    for spec in args.models:
        if ":" not in spec:
            parser.error(f"Model spec must be 'name:path', got: {spec!r}")
        name, _, path = spec.partition(":")
        if not os.path.isdir(path):
            parser.error(f"Model directory not found: {path}")
        model_dirs.append((name, path))

    evaluate(model_dirs, args.datasets, args.n_samples, args.output)


if __name__ == "__main__":
    main()
