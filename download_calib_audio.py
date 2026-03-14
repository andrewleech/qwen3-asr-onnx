#!/usr/bin/env python3
"""
Download N audio samples from LibriSpeech test-other (HuggingFace streaming)
and save as WAV files for GPTQ calibration.

Usage:
    uv run python download_calib_audio.py \
        --output calibration_cache/audio \
        --n-samples 32
"""

import argparse
import io
import os

import numpy as np
import soundfile as sf
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="calibration_cache/audio", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Streaming {args.n_samples} samples from LibriSpeech test-other ...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        "other",
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    saved = 0
    for sample in ds:
        if saved >= args.n_samples:
            break
        audio = sample["audio"]
        arr = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        out_path = os.path.join(args.output, f"sample_{saved:04d}.wav")
        sf.write(out_path, arr, sr)
        duration = len(arr) / sr
        print(f"  [{saved+1}/{args.n_samples}] {os.path.basename(out_path)} ({duration:.1f}s)")
        saved += 1

    print(f"Done. {saved} files saved to {args.output}/")


if __name__ == "__main__":
    main()
