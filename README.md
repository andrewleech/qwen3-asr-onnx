# Qwen3-ASR ONNX Export Tool

Export [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) to ONNX format for use with ONNX Runtime.

Pre-exported models are available on HuggingFace:

- [andrewleech/qwen3-asr-0.6b-onnx](https://huggingface.co/andrewleech/qwen3-asr-0.6b-onnx) — FP32 + INT8 (AWQ α=0.2)
- [andrewleech/qwen3-asr-1.7b-onnx](https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx) — FP32 + int4 (GPTQ+RTN al4)

## Output Files

Each model directory contains FP32 files plus quantized variants with suffixed names:

| File pattern | Description |
|---|---|
| `encoder.onnx` | FP32 audio encoder (weights inlined) |
| `encoder.{int8,fp16}.onnx` | Quantized encoder variants |
| `decoder_init.onnx` + `.data` | FP32 decoder prefill (full sequence, outputs KV cache) |
| `decoder_init.{int4,int8,fp16}.onnx` + `.data` | Quantized decoder_init variants |
| `decoder_step.onnx` + `.data` | FP32 decoder autoregressive step (single token + KV cache) |
| `decoder_step.{int4,int8,fp16}.onnx` + `.data` | Quantized decoder_step variants (may be inlined when < 2 GB) |
| `embed_tokens.bin` | Token embedding matrix `[vocab_size, hidden_size]` as raw float32 |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config + special tokens + mel params |

The decoder `.onnx` files are small graph protos (~2 MB each) with weights in separate `.data` files. For smaller quantized variants (e.g. 0.6B INT8 decoder_step at 571 MB), weights are inlined in the `.onnx` file. Encoder weights are always inlined.

Multiple quantization variants coexist in a single directory. The consumer selects a variant at load time (e.g. `Quantization::Int4` in [transcribe-rs](https://github.com/andrewleech/transcribe-rs)), which resolves `decoder_init.int4.onnx` with automatic fallback to `decoder_init.onnx` (FP32) for files without a quantized variant (such as the encoder for int4 models).

### Model Sizes

| Model | FP32 | Best quantized | Quantized size |
|---|---|---|---|
| **0.6B** | ~5.8 GB | INT8 AWQ α=0.2 | ~2.1 GB |
| **1.7B** | ~16 GB | int4 GPTQ+RTN al4 | ~5.2 GB |

INT8 models (0.6B) are produced via AWQ smoothing followed by dynamic quantization. AWQ smoothing collects per-channel activation statistics from calibration audio, then migrates outlier activation variance into RMSNorm and linear weights. The smoothed FP32 model is mathematically equivalent to the original (verified by token-for-token comparison).

int4 models (1.7B) use ORT's `MatMulNBitsQuantizer` with per-group scales (block size 64). GPTQ calibration is used for `decoder_init` (better layer-wise reconstruction); RTN with `accuracy_level=4` is used for `decoder_step` (no quality difference, avoids impractical KV-cache calibration data).

### WER and speed vs Parakeet (LibriSpeech test-other, 200 samples, WSL/Linux CPU)

| Engine | WER | RTF (11s) | vs Parakeet speed |
|---|---|---|---|
| 1.7B FP32 | 3.79% | ~0.70x | 4.4× slower |
| 1.7B int4 GPTQ+RTN al4 | 4.25% | 0.34x | 2.1× slower |
| 0.6B FP32 | 4.42% | 0.29x | 1.8× slower |
| **0.6B AWQ INT8 α=0.2** | **5.21%** | **0.14x** | **1.1× faster** |
| Parakeet-TDT 0.6B INT8 | 5.45% | 0.16x | baseline |
| 0.6B int4 RTN al4 | 5.08% | 0.19x | 1.2× slower |

RTF < 1.0 = faster than real-time. Lower is better for both WER and RTF.

**0.6B AWQ INT8 α=0.2** is the recommended configuration: it beats Parakeet on both WER and speed at 1.1 GB model size. Qwen3 also produces full punctuation; Parakeet outputs minimal punctuation.

**1.7B int4 GPTQ+RTN al4** is the accuracy option: 4.25% WER at 2× the RTF of Parakeet. AWQ INT8 for 1.7B is not recommended — it causes degraded special token prediction (9% WER).

## Setup

```bash
pip install -r requirements.txt
# or with uv:
uv sync
```

## Pipeline

The export pipeline has three stages. Each stage is a standalone script. Quantized files are written into the same directory as the FP32 source using suffixed filenames.

### 1. Export (FP32 ONNX)

```bash
python export.py --model Qwen/Qwen3-ASR-0.6B
python export.py --model Qwen/Qwen3-ASR-1.7B
```

Produces `output/qwen3-asr-0.6b/` (or `1.7b`) with FP32 ONNX files. Encoder weights are embedded in the proto; decoder init and step use separate external weight files.

Options: `--skip-encoder`, `--skip-decoder`, `--device cuda`, `--opset 18`.

### 2. Validate (FP32 vs PyTorch)

```bash
python validate.py --onnx-dir output/qwen3-asr-0.6b --audio tests/fixtures/test_audio.wav
```

Loads the PyTorch model and the ONNX export, runs the same audio through both, and compares encoder features and decoded tokens. Expects exact token-for-token match.

### 3. AWQ Smooth + Quantize (INT8, recommended for 0.6B)

```bash
# Collect calibration activations, smooth weights, re-export FP32 ONNX
python awq_smooth.py --model Qwen/Qwen3-ASR-0.6B \
    --output output/qwen3-asr-0.6b-smooth \
    --alpha 0.2 --n-samples 128 --verify

# Quantize the smoothed model to INT8 (adds .int8 files alongside FP32)
python quantize.py \
    --input output/qwen3-asr-0.6b-smooth \
    --output output/qwen3-asr-0.6b
```

α=0.2 is the optimal smoothing strength for 0.6B INT8. Calibration activations are cached in the smooth output directory (`calibration_activations.npz`). To sweep alpha values without re-collecting:

```bash
python awq_smooth.py --model Qwen/Qwen3-ASR-0.6B \
    --output output/qwen3-asr-0.6b-smooth \
    --alpha 0.2 \
    --activations-cache output/qwen3-asr-0.6b-smooth/calibration_activations.npz
```

### 4. int4 MatMulNBits Quantization (recommended for 1.7B)

`quantize_nbits.py` applies ORT's `MatMulNBitsQuantizer` to the decoder files. The encoder is not quantized (int4 degrades encoder quality).

```bash
# RTN (no calibration data, fast — adds .int4 files alongside FP32):
python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b \
    --bits 4 --block-size 64 --accuracy-level 4
```

`--accuracy-level 4` activates a higher-precision accumulation kernel in ORT that is both faster and more accurate than the default on x86.

For improved load time via GPTQ calibration on decoder_init, see the next section.

### 5. GPTQ Calibration + int4 (1.7B, best load time)

GPTQ minimises layer-wise reconstruction error using real decoder inputs, producing a smaller decoder_init with ~40% faster load time. WER and inference speed are identical to RTN.

Collecting calibration data for `decoder_init` (~22 MB output):

```bash
python collect_gptq_calib.py \
    --model output/qwen3-asr-1.7b \
    --n-samples 32 --decoder-steps 8 \
    --output calibration_cache/1.7b_gptq_init.npz \
    --target decoder_init
```

The recommended hybrid uses GPTQ for `decoder_init` and RTN al4 for `decoder_step`:

```bash
# GPTQ decoder_init (adds decoder_init.int4.onnx)
python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --algo gptq --calib-data calibration_cache/1.7b_gptq_init.npz \
    --decoders decoder_init

# RTN al4 decoder_step (adds decoder_step.int4.onnx)
python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --decoders decoder_step

# Clean up GPTQ temp files
rm -f output/qwen3-asr-1.7b/*-*-*-*-*.data output/qwen3-asr-1.7b/*_augment.onnx
```

The script handles `config.json` hiding during GPTQ (ORT's neural_compressor calls `AutoConfig.from_pretrained()` which fails on the custom `qwen3_asr` model type) and adds the `com.microsoft` opset import for `MatMulNBits` nodes.

GPTQ writes large UUID-named temporary `.data` files (~6.5 GB each) in the source directory — the cleanup command removes them.

### Evaluate WER

```bash
python evaluate_wer.py \
    --models "0.6B FP32:output/qwen3-asr-0.6b" \
             "0.6B INT8:output/qwen3-asr-0.6b" \
    --datasets librispeech-other \
    --n-samples 200 --output results.json
```

Streams samples from HuggingFace datasets (no full download), runs ONNX inference for each model, and reports WER. Supports `librispeech-other` and `ami-sdm` datasets.

### Other Tools

```bash
# Compare inference paths (native PyTorch, wrapper, FP32 ONNX, INT8 ONNX)
python compare.py --audio tests/fixtures/librispeech_0.wav

# Upload to HuggingFace
python upload.py --input output/qwen3-asr-0.6b --repo andrewleech/qwen3-asr-0.6b-onnx
```

## Full Reproduction

Complete sequence to produce all output variants from the upstream HuggingFace models. All commands assume `uv run python` (or activate the venv and use `python` directly).

The upstream models are downloaded automatically from HuggingFace on first use and cached in `~/.cache/huggingface/hub/`.

### 0.6B (FP32 + AWQ INT8)

```bash
# FP32 export (~5 min, ~4 GB RAM)
uv run python export.py --model Qwen/Qwen3-ASR-0.6B

# Validate FP32 ONNX matches PyTorch
uv run python validate.py \
    --onnx-dir output/qwen3-asr-0.6b \
    --audio tests/fixtures/test_audio.wav

# AWQ smooth + re-export (~25 min, ~6 GB RAM — calibration is the bottleneck)
uv run python awq_smooth.py \
    --model Qwen/Qwen3-ASR-0.6B \
    --output output/qwen3-asr-0.6b-smooth \
    --alpha 0.2 --n-samples 128 --verify

# Quantize smoothed model to INT8 (writes .int8 files into 0.6b dir)
uv run python quantize.py \
    --input output/qwen3-asr-0.6b-smooth \
    --output output/qwen3-asr-0.6b
```

### 1.7B (FP32 + int4 GPTQ hybrid)

```bash
# FP32 export (~15 min, ~10 GB RAM)
uv run python export.py --model Qwen/Qwen3-ASR-1.7B

# Validate
uv run python validate.py \
    --onnx-dir output/qwen3-asr-1.7b \
    --audio tests/fixtures/test_audio.wav

# Collect GPTQ calibration data for decoder_init (~10 min on 24-core, ~22 MB output)
uv run python collect_gptq_calib.py \
    --model output/qwen3-asr-1.7b \
    --n-samples 32 --decoder-steps 8 \
    --output calibration_cache/1.7b_gptq_init.npz \
    --target decoder_init

# GPTQ decoder_init (writes decoder_init.int4.onnx into 1.7b dir)
uv run python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --algo gptq --calib-data calibration_cache/1.7b_gptq_init.npz \
    --decoders decoder_init

# RTN al4 decoder_step (writes decoder_step.int4.onnx into 1.7b dir)
uv run python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --decoders decoder_step

# Clean up GPTQ temp files
rm -f output/qwen3-asr-1.7b/*-*-*-*-*.data output/qwen3-asr-1.7b/*_augment.onnx
```

### WER evaluation

```bash
uv run python evaluate_wer.py \
    --models "0.6B FP32:output/qwen3-asr-0.6b" \
             "0.6B INT8:output/qwen3-asr-0.6b" \
             "1.7B FP32:output/qwen3-asr-1.7b" \
             "1.7B int4:output/qwen3-asr-1.7b" \
    --datasets librispeech-other \
    --n-samples 200 --output wer_results.json
```

### Output structure

After full reproduction:

```
output/
├── qwen3-asr-0.6b/              # All 0.6B variants in one directory
│   ├── encoder.onnx              # FP32 (shared — no int4 encoder needed)
│   ├── encoder.int8.onnx         # INT8 AWQ
│   ├── decoder_init.onnx         # FP32
│   ├── decoder_init.onnx.data
│   ├── decoder_init.int8.onnx    # INT8 AWQ
│   ├── decoder_init.int8.onnx.data
│   ├── decoder_step.onnx         # FP32
│   ├── decoder_step.onnx.data
│   ├── decoder_step.int8.onnx    # INT8 AWQ (inlined, no .data)
│   ├── embed_tokens.bin
│   ├── config.json
│   └── tokenizer.json
├── qwen3-asr-0.6b-smooth/       # AWQ intermediate (+ calibration_activations.npz)
├── qwen3-asr-1.7b/              # All 1.7B variants in one directory
│   ├── encoder.onnx              # FP32 (shared)
│   ├── decoder_init.onnx         # FP32
│   ├── decoder_init.onnx.data
│   ├── decoder_init.int4.onnx    # int4 GPTQ
│   ├── decoder_init.int4.onnx.data
│   ├── decoder_step.onnx         # FP32
│   ├── decoder_step.onnx.data
│   ├── decoder_step.int4.onnx    # int4 RTN al4
│   ├── decoder_step.int4.onnx.data
│   ├── embed_tokens.bin
│   ├── config.json
│   └── tokenizer.json
└── calibration_cache/            # GPTQ calibration data (reusable)
```

Each model directory is self-contained and can be used directly by ONNX Runtime consumers. The `-smooth` directory is an intermediate; it can be deleted after INT8 quantization. For distribution, quantized files can be packaged separately from FP32 files while sharing the common files (encoder, tokenizer, config, embed_tokens).

## Architecture

- **Encoder**: mel `[1, 128, T]` → windowed Conv2D (100-frame windows, 3x stride-2 convs producing 13 tokens per window) → windowed attention (104-token windows across transformer layers) → MLP projection → `[1, N, output_dim]` where `N = get_feat_extract_output_lengths(T)`. 0.6B: 18 layers, d=896, output_dim=1024. 1.7B: 24 layers, d=1024, output_dim=2048.
- **Decoder**: Qwen3 (28 layers, 16 Q-heads / 8 KV-heads, GQA, SwiGLU, MRoPE). 0.6B: d=1024. 1.7B: d=2048.
- **Integration**: Prefix LM — encoder output replaces `<|audio_pad|>` token embeddings. Decoder uses self-attention only.

The encoder uses windowed processing from the native Qwen3-ASR architecture: mel frames are split into 100-frame convolution windows, each producing 13 tokens via a 3-layer stride-2 Conv2D stack. Tokens are then grouped into 104-token attention windows for the transformer layers. Padding tokens in the final window are masked out.

## Mel Spectrogram Parameters

Identical to Whisper: 16kHz, 128 bins, 25ms window (n_fft=400), 10ms hop (hop_length=160), Hann window, Slaney mel scale, 0-8kHz.

## Comparison Results

### FP32 ONNX vs PyTorch

Tested against the native Qwen3-ASR model on LibriSpeech samples (5s to 35s).

| Duration | Native vs FP32 | Wrapper vs FP32 | Encoder max_diff |
|----------|---------------|-----------------|-----------------|
| 5-12s | Exact token match | Exact token match | < 6e-6 |
| 30s | Exact token match | Exact token match | < 6e-6 |
| 32-35s | Same words, punctuation differs | Exact token match | < 6e-6 |

- **Wrapper vs FP32**: Always exact token match — the ONNX export faithfully reproduces the PyTorch wrapper.
- **Native vs FP32**: Identical on short audio. On long audio (>30s), SDPA vs eager attention causes punctuation-level divergence; word content matches.

### Quantization Quality

WER measured on LibriSpeech test-other (200 samples).

| Model | FP32 WER | INT8 naive | INT8 AWQ α=0.2 | int4 RTN al4 | int4 GPTQ+RTN al4 |
|---|---|---|---|---|---|
| 0.6B | 4.4% | 6.7% | 5.2% | 5.1% | — |
| 1.7B | 3.8% | — | 9.0% | 4.3% | 4.25% |

AWQ smoothing reduces the 0.6B INT8 WER penalty from +2.3pp (naive) to +0.8pp (α=0.2). 1.7B AWQ INT8 is not recommended — outlier weights in the 1.7B decoder cause special token prediction failures under per-tensor INT8 quantization regardless of smoothing. int4 MatMulNBits with per-group scales handles these outliers locally; the 1.7B int4 WER penalty is only +0.46pp vs FP32.

## DirectML Compatibility

The exported ONNX graphs are patched post-export to remove `allowzero=1` from all `Reshape` nodes. DirectML rejects this attribute, and no shape tensor in the graphs contains a literal zero dimension, so removal is safe. This is done automatically during export.

## Tests

```bash
python -m pytest tests/
```

Audio fixtures in `tests/fixtures/` are tracked with Git LFS. Run `git lfs pull` after cloning to fetch them. See `tests/fixtures/README.md` for provenance.

## License

Apache 2.0, matching the upstream Qwen3-ASR models.
