# Qwen3-ASR ONNX Export Tool

Export [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) to ONNX format for use with ONNX Runtime.

Pre-exported models are available on HuggingFace:

- [andrewleech/qwen3-asr-0.6b-onnx](https://huggingface.co/andrewleech/qwen3-asr-0.6b-onnx) — FP32 + INT8 (AWQ α=0.2)
- [andrewleech/qwen3-asr-1.7b-onnx](https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx) — FP32 + int4 (GPTQ+RTN al4)

## Output Files

Each export produces a directory containing:

| File | Description |
|---|---|
| `encoder.onnx` | Audio encoder (Conv2D stem + transformer + MLP projection), weights inlined |
| `decoder_init.onnx` | Decoder prefill (full sequence, outputs KV cache) |
| `decoder_init.onnx.data` | External weights for decoder_init |
| `decoder_step.onnx` | Decoder autoregressive step (single token + KV cache) |
| `decoder_step.onnx.data` | External weights for decoder_step (may be inlined for smaller variants) |
| `embed_tokens.bin` | Token embedding matrix `[vocab_size, hidden_size]` as raw float32 |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config + special tokens + mel params |

The decoder `.onnx` files are small graph protos (~2 MB each) with weights stored in separate `.data` files via ONNX external data pointers. For smaller quantized variants (e.g. 0.6B INT8 decoder_step at 571 MB), weights may be inlined in the `.onnx` file when under the 2 GB protobuf limit. Encoder weights are always inlined.

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

The export pipeline has three stages. Each stage is a standalone script.

### 1. Export (FP32 ONNX)

```bash
python export.py --model Qwen/Qwen3-ASR-0.6B
python export.py --model Qwen/Qwen3-ASR-1.7B
```

Produces `output/qwen3-asr-0.6b/` (or `1.7b`) with FP32 ONNX files. Encoder weights are embedded in the proto; decoder init and step share a single external weights file via `share_weights.py`.

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

# Quantize the smoothed model to INT8
python quantize.py --input output/qwen3-asr-0.6b-smooth \
    --output output/qwen3-asr-0.6b-int8
```

α=0.2 is the optimal smoothing strength for 0.6B INT8. Calibration activations are cached in the output directory (`calibration_activations.npz`). To sweep alpha values without re-collecting:

```bash
python awq_smooth.py --model Qwen/Qwen3-ASR-0.6B \
    --output output/qwen3-asr-0.6b-smooth-a02 \
    --alpha 0.2 \
    --activations-cache output/qwen3-asr-0.6b-smooth/calibration_activations.npz
```

### 4. int4 MatMulNBits Quantization (recommended for 1.7B)

`quantize_nbits.py` applies ORT's `MatMulNBitsQuantizer` to the decoder files. The encoder is copied unchanged (int4 quantization degrades encoder quality).

```bash
# RTN (no calibration data, fast):
python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b-int4 \
    --bits 4 --block-size 64 --accuracy-level 4
```

`--accuracy-level 4` activates a higher-precision accumulation kernel in ORT that is both faster and more accurate than the default on x86 (0.25x vs 0.56x RTF, 4.33% vs 4.33% WER for RTN).

For improved WER via GPTQ calibration, see the next section.

### 5. GPTQ Calibration + int4 (1.7B, best accuracy)

GPTQ minimises layer-wise reconstruction error using real decoder inputs. Calibration data is collected by running the full inference pipeline on LibriSpeech audio samples.

Collecting calibration data for `decoder_init` (~22 MB, practical):

```bash
python collect_gptq_calib.py \
    --model output/qwen3-asr-1.7b \
    --n-samples 32 \
    --decoder-steps 8 \
    --output calibration_cache/1.7b_gptq_init.npz \
    --target decoder_init
```

`decoder_step` calibration (32 samples × 8 steps × KV cache ~20 MB each ≈ 5 GB .npz) is impractical to load alongside an 8.8 GB model. The recommended hybrid uses GPTQ for `decoder_init` and RTN al4 for `decoder_step`:

```bash
# GPTQ decoder_init
python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b-int4 \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --algo gptq --calib-data calibration_cache/1.7b_gptq_init.npz \
    --decoders decoder_init

# RTN al4 decoder_step (quantize into same output dir)
python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b-int4 \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --decoders decoder_step

# Clean up GPTQ temp files in source dir
rm -f output/qwen3-asr-1.7b/*-*-*-*-*.data output/qwen3-asr-1.7b/*_augment.onnx
```

The script automatically handles `config.json` hiding during GPTQ (ORT's neural_compressor calls `AutoConfig.from_pretrained()` which fails on the custom `qwen3_asr` model type), file renaming to standard `{name}.onnx` names, external data reference fixup, and `com.microsoft` opset import for `MatMulNBits` nodes.

GPTQ writes large UUID-named temporary `.data` files (~6.5 GB each) in the *source* model directory — the cleanup command above removes them.

### Evaluate WER

```bash
python evaluate_wer.py \
    --models "0.6B FP32:output/qwen3-asr-0.6b" \
             "0.6B INT8:output/qwen3-asr-0.6b-int8" \
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

# Quantize smoothed model to INT8
uv run python quantize.py \
    --input output/qwen3-asr-0.6b-smooth \
    --output output/qwen3-asr-0.6b-int8
```

### 1.7B (FP32 + int4 GPTQ hybrid)

```bash
# FP32 export (~15 min, ~10 GB RAM)
uv run python export.py --model Qwen/Qwen3-ASR-1.7B

# Validate
uv run python validate.py \
    --onnx-dir output/qwen3-asr-1.7b \
    --audio tests/fixtures/test_audio.wav

# Collect GPTQ calibration data for decoder_init (~10 min, ~22 MB output)
uv run python collect_gptq_calib.py \
    --model output/qwen3-asr-1.7b \
    --n-samples 32 --decoder-steps 8 \
    --output calibration_cache/1.7b_gptq_init.npz \
    --target decoder_init

# GPTQ decoder_init
uv run python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b-int4 \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --algo gptq --calib-data calibration_cache/1.7b_gptq_init.npz \
    --decoders decoder_init

# RTN al4 decoder_step (no calibration data needed)
uv run python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b-int4 \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --decoders decoder_step

# Clean up GPTQ temp files in source dir
rm -f output/qwen3-asr-1.7b/*-*-*-*-*.data output/qwen3-asr-1.7b/*_augment.onnx
```

Disk peak during 1.7B int4: ~13 GB (8.8 GB FP32 source + 4.3 GB output). GPTQ writes ~6.5 GB temporary `.data` files to the source directory during the `decoder_init` step — clean them up as shown above.

### WER evaluation

```bash
uv run python evaluate_wer.py \
    --models "0.6B FP32:output/qwen3-asr-0.6b" \
             "0.6B INT8:output/qwen3-asr-0.6b-int8" \
             "1.7B FP32:output/qwen3-asr-1.7b" \
             "1.7B int4:output/qwen3-asr-1.7b-int4" \
    --datasets librispeech-other \
    --n-samples 200 --output wer_results.json
```

### Output directories

After full reproduction:

```
output/
├── qwen3-asr-0.6b/         # FP32
├── qwen3-asr-0.6b-smooth/  # AWQ-smoothed FP32 (intermediate, + calibration_activations.npz)
├── qwen3-asr-0.6b-int8/    # INT8 AWQ α=0.2 (distributable)
├── qwen3-asr-1.7b/         # FP32
└── qwen3-asr-1.7b-int4/    # int4 GPTQ+RTN al4 (distributable)
```

Each directory is self-contained and can be used directly by ONNX Runtime consumers (e.g. [transcribe-rs](https://github.com/andrewleech/transcribe-rs)). The `-smooth` directory is an intermediate; distributable artifacts are FP32, INT8, and int4.

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
