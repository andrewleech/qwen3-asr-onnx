# Benchmark Results

Date: 2026-03-12
Platform: WSL2 Linux, CPU only, all models and audio on WSL ext4
Parameters: 2 warmup + 5 timed runs per configuration

## JFK Audio (11.0s)

| Engine | Load | Mean | Min | Max | RTF |
|---|---|---|---|---|---|
| Qwen3-ASR 0.6B FP32 | 17.0s | 3.14s | 3.01s | 3.32s | 0.29x |
| Qwen3-ASR 0.6B Smooth INT8 | 4.9s | 1.49s | 1.38s | 1.62s | 0.14x |
| Parakeet-TDT 0.6B INT8 | 1.2s | 1.74s | 1.56s | 1.96s | 0.16x |

## LibriSpeech (34.95s)

| Engine | Load | Mean | Min | Max | RTF |
|---|---|---|---|---|---|
| Qwen3-ASR 0.6B FP32 | 9.8s | 11.10s | 10.88s | 11.31s | 0.32x |
| Qwen3-ASR 0.6B Smooth INT8 | 3.2s | 5.82s | 5.72s | 6.04s | 0.17x |
| Parakeet-TDT 0.6B INT8 | 0.9s | 4.71s | 4.25s | 5.11s | 0.13x |

## Notes

- RTF < 1 = faster than real-time
- Qwen3 FP32 dir has no .int8.onnx files, so pure FP32 encoder+decoder is loaded
- Qwen3 Smooth INT8 files are named without .int8. suffix but contain INT8-quantized weights
- Qwen3 Smooth INT8 is ~2x faster than FP32 due to INT8 weight quantization
- Load time difference: FP32 model is ~3.8 GB vs ~1.1 GB for smooth INT8
- Parakeet has the fastest load time (smallest model, ~670 MB)

## Transcription Output Comparison (JFK 11s)

- **Qwen3 FP32**: "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country."
- **Qwen3 Smooth INT8**: "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
- **Parakeet INT8**: "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country."

Qwen3 Smooth INT8 uses a comma instead of semicolon; Parakeet uses a period + new sentence.

## Transcription Output Comparison (LibriSpeech 35s)

- **Qwen3 FP32**: "...your conscience and your vertible column reproach you..."
- **Qwen3 Smooth INT8**: "...your conscience and your vertible calm reproach you..."
- **Parakeet INT8**: "...your conscience and your vertebral column reproach you..."

Parakeet gets "vertebral column" correct. Both Qwen3 variants produce "vertible" (FP32) or "vertible calm" (smooth INT8).

## WER Results (LibriSpeech test-other, 200 samples)

| Engine | WER |
|---|---|
| Qwen3-ASR 1.7B FP32 | 3.79% |
| Qwen3-ASR 1.7B int4 GPTQ+RTN al4 | 4.25% |
| Qwen3-ASR 0.6B FP32 | 4.42% |
| Qwen3-ASR 0.6B int4 RTN al4 | 5.08% |
| Qwen3-ASR 0.6B AWQ INT8 α=0.2 | 5.21% |
| Parakeet-TDT 0.6B INT8 | 5.45% |
| Qwen3-ASR 0.6B AWQ INT8 α=0.5 | 5.62% |
| Qwen3-ASR 0.6B Naive INT8 | 6.75% |
| Qwen3-ASR 1.7B AWQ INT8 α=0.2 | 9.04% |

## JFK RTF Results (1.7B, 2026-03-14)

| Engine | RTF (11s) | RTF (35s) |
|---|---|---|
| Qwen3-ASR 1.7B FP32 | ~0.70x | ~0.70x |
| Qwen3-ASR 1.7B int4 GPTQ+RTN al4 | 0.34x | — |
| Qwen3-ASR 1.7B int4 RTN (block_size=64) | 0.56x | 0.65x |

## Combined Summary

| Engine | WER | RTF (11s) | RTF (35s) | vs Parakeet |
|---|---|---|---|---|
| Qwen3-ASR 1.7B FP32 | 3.79% | ~0.70x | ~0.70x | 4.4× slower |
| Qwen3-ASR 1.7B int4 GPTQ+RTN al4 | 4.25% | 0.34x | — | 2.1× slower |
| Qwen3-ASR 0.6B FP32 | 4.42% | 0.29x | 0.32x | 1.8× slower |
| Qwen3-ASR 0.6B AWQ INT8 α=0.2 | 5.21% | 0.14x | 0.17x | 1.1× faster |
| Parakeet-TDT 0.6B INT8 | 5.45% | 0.16x | 0.13x | baseline |
| Qwen3-ASR 0.6B AWQ INT8 α=0.5 | 5.62% | 0.14x | 0.17x | 1.1× faster |
