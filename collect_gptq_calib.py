#!/usr/bin/env python3
"""
collect_gptq_calib.py — Collect decoder input tensors for GPTQ calibration.

Runs the actual Qwen3-ASR inference pipeline on audio samples and captures
decoder_init and decoder_step inputs for use with GPTQ quantization.

Input data is streamed from HuggingFace LibriSpeech test-other (same source
as evaluate_wer.py), ensuring calibration distribution matches evaluation.

Output: pickle files containing lists of input dicts compatible with ORT's
CalibrationDataReader interface.

Usage:
    # Collect both init and step inputs (decoder_init and decoder_step PKLs):
    uv run python collect_gptq_calib.py \
        --model output/qwen3-asr-1.7b \
        --n-samples 32 \
        --decoder-steps 8 \
        --output calibration_cache/1.7b_gptq_calib.pkl \
        --target decoder_init

    uv run python collect_gptq_calib.py \
        --model output/qwen3-asr-1.7b \
        --n-samples 32 \
        --decoder-steps 8 \
        --output calibration_cache/1.7b_gptq_calib_step.pkl \
        --target decoder_step
"""

import argparse
import os
import pickle
import sys

import numpy as np
import onnxruntime as ort

# Add src/ to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_sessions(model_dir: str, threads: int) -> dict:
    """Load encoder, decoder_init, and decoder_step ORT sessions."""
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = threads
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    sessions = {}
    for name in ["encoder", "decoder_init", "decoder_step"]:
        path = os.path.join(model_dir, f"{name}.onnx")
        if os.path.exists(path):
            print(f"  Loading {name}.onnx ...")
            sessions[name] = ort.InferenceSession(path, sess_options=sess_opts)
    return sessions


def load_embed(model_dir: str) -> np.ndarray:
    """Load embed_tokens.bin as float32 [vocab, hidden]."""
    path = os.path.join(model_dir, "embed_tokens.bin")
    data = np.fromfile(path, dtype=np.float16)
    for hidden in [2048, 1024]:
        if len(data) % hidden == 0:
            return data.reshape(-1, hidden).astype(np.float32)
    raise ValueError(f"Cannot determine embed_tokens shape from {len(data)} fp16 values")


def stream_audio(n: int):
    """Yield float32 16kHz audio arrays from LibriSpeech test-other (streaming)."""
    from datasets import Audio, load_dataset
    import soundfile as sf
    import io as _io
    import librosa

    ds = load_dataset(
        "openslr/librispeech_asr",
        "other",
        split="test",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    yielded = 0
    for sample in ds:
        if yielded >= n:
            break
        raw = sample["audio"]["bytes"]
        arr, sr = sf.read(_io.BytesIO(raw))
        arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        yield arr
        yielded += 1


def collect_init_inputs(sessions, embed_tokens, n_samples, verbose=True):
    """Collect decoder_init input dicts from n_samples audio files."""
    from src.mel import log_mel_spectrogram
    from src.prompt import build_prompt_ids, get_audio_pad_range

    inputs = []
    for i, audio in enumerate(stream_audio(n_samples)):
        if verbose:
            print(f"  [{i+1}/{n_samples}] {len(audio)/16000:.1f}s audio", flush=True)

        mel = log_mel_spectrogram(audio)
        mel_np = mel.cpu().numpy()

        audio_features = sessions["encoder"].run(["audio_features"], {"mel": mel_np})[0]
        audio_token_count = audio_features.shape[1]
        if audio_token_count == 0:
            continue

        prompt_ids = build_prompt_ids(audio_token_count)
        input_embeds = embed_tokens[prompt_ids].copy()
        audio_start, audio_end = get_audio_pad_range(prompt_ids)
        input_embeds[audio_start:audio_end] = audio_features[0]
        input_embeds = input_embeds[np.newaxis, :, :]
        position_ids = np.arange(len(prompt_ids), dtype=np.int64)[np.newaxis, :]

        inputs.append({
            "input_embeds": input_embeds,
            "position_ids": position_ids,
        })

    return inputs


def collect_step_inputs(sessions, embed_tokens, n_samples, decoder_steps, verbose=True):
    """Collect decoder_step input dicts (with real KV cache) from n_samples audio files."""
    from src.mel import log_mel_spectrogram
    from src.prompt import build_prompt_ids, get_audio_pad_range, EOS_TOKEN_IDS

    inputs = []
    for i, audio in enumerate(stream_audio(n_samples)):
        if verbose:
            print(f"  [{i+1}/{n_samples}] {len(audio)/16000:.1f}s audio", flush=True)

        mel = log_mel_spectrogram(audio)
        mel_np = mel.cpu().numpy()

        audio_features = sessions["encoder"].run(["audio_features"], {"mel": mel_np})[0]
        audio_token_count = audio_features.shape[1]
        if audio_token_count == 0:
            continue

        prompt_ids = build_prompt_ids(audio_token_count)
        input_embeds = embed_tokens[prompt_ids].copy()
        audio_start, audio_end = get_audio_pad_range(prompt_ids)
        input_embeds[audio_start:audio_end] = audio_features[0]
        input_embeds = input_embeds[np.newaxis, :, :]
        position_ids = np.arange(len(prompt_ids), dtype=np.int64)[np.newaxis, :]

        # Run decoder_init to get initial KV cache
        logits, keys, values = sessions["decoder_init"].run(
            ["logits", "present_keys", "present_values"],
            {"input_embeds": input_embeds, "position_ids": position_ids},
        )

        next_token = int(np.argmax(logits[0, -1, :]))
        pos = len(prompt_ids)

        # Collect decoder_step inputs for first `decoder_steps` steps
        for step in range(decoder_steps):
            if next_token in EOS_TOKEN_IDS:
                break
            step_embed = embed_tokens[next_token][np.newaxis, np.newaxis, :]
            step_pos = np.array([[pos]], dtype=np.int64)

            inputs.append({
                "input_embeds": step_embed,
                "position_ids": step_pos,
                "past_keys": keys,
                "past_values": values,
            })

            if step < decoder_steps - 1:
                logits, keys, values = sessions["decoder_step"].run(
                    ["logits", "present_keys", "present_values"],
                    {
                        "input_embeds": step_embed,
                        "position_ids": step_pos,
                        "past_keys": keys,
                        "past_values": values,
                    },
                )
                next_token = int(np.argmax(logits[0, -1, :]))
                pos += 1

    print(f"  Collected {len(inputs)} step input samples")
    return inputs


def main():
    parser = argparse.ArgumentParser(description="Collect GPTQ calibration data for Qwen3-ASR decoders")
    parser.add_argument("--model", required=True, help="Model directory (FP32 ONNX)")
    parser.add_argument("--n-samples", type=int, default=32, help="Number of audio samples to use")
    parser.add_argument(
        "--target",
        choices=["decoder_init", "decoder_step"],
        required=True,
        help="Which decoder to collect calibration data for",
    )
    parser.add_argument(
        "--decoder-steps",
        type=int,
        default=8,
        help="Number of decode steps to collect per sample (for decoder_step target)",
    )
    parser.add_argument("--output", required=True, help="Output pickle file path")
    parser.add_argument("--threads", type=int, default=8, help="ORT intra-op threads")
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    sessions = load_sessions(args.model, args.threads)
    embed_tokens = load_embed(args.model)
    print(f"  embed_tokens: {embed_tokens.shape}")

    if args.target == "decoder_init":
        print(f"\nCollecting {args.n_samples} decoder_init samples ...")
        data = collect_init_inputs(sessions, embed_tokens, args.n_samples)
    else:
        print(f"\nCollecting {args.n_samples} × {args.decoder_steps} decoder_step samples ...")
        data = collect_step_inputs(sessions, embed_tokens, args.n_samples, args.decoder_steps)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)
    size_mb = os.path.getsize(args.output) / 1024**2
    print(f"\nSaved {len(data)} samples to {args.output} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
