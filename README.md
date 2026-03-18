# Qwen3-ASR ONNX Export Tool

Export [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) to ONNX format for use with ONNX Runtime.

Pre-exported models on HuggingFace:

- [andrewleech/qwen3-asr-0.6b-onnx](https://huggingface.co/andrewleech/qwen3-asr-0.6b-onnx) — FP32 + int4
- [andrewleech/qwen3-asr-1.7b-onnx](https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx) — FP32 + int4

## Output Files

Each model directory contains FP32 files plus quantized variants with suffixed names:

| File pattern | Description |
|---|---|
| `encoder.onnx` | FP32 audio encoder (weights inlined) |
| `encoder.int4.onnx` | Encoder for int4 variant (FP32 weights — INT8/FP16 encoders degrade quality) |
| `decoder_init.onnx` + `.data` | FP32 decoder prefill (full sequence, outputs KV cache) |
| `decoder_init.int4.onnx` + `.data` | int4 decoder_init variant |
| `decoder_step.onnx` + `.data` | FP32 decoder autoregressive step (single token + KV cache) |
| `decoder_step.int4.onnx` + `.data` | int4 decoder_step variant |
| `embed_tokens.bin` | Token embedding matrix `[vocab_size, hidden_size]` as raw float32 (see below) |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config + special tokens + mel params |

The decoder `.onnx` files are small graph protos (~2 MB each) with weights in separate `.data` files. Encoder weights are always inlined.

**Why `embed_tokens.bin` exists separately:** The two decoder models use different input strategies. `decoder_init` (prefill) accepts `input_ids` and has the embedding table built into its ONNX graph — this allows it to handle the audio feature scatter internally. `decoder_step` (autoregressive) accepts pre-looked-up `input_embeds` instead, keeping the ~594 MB embedding table out of its graph. The consumer loads `embed_tokens.bin` once at startup and performs the embedding lookup per token before calling `decoder_step`. This avoids duplicating the embedding weights across both decoder files while keeping `decoder_step` small for fast loading.

Multiple quantization variants coexist in a single directory. The consumer selects a variant at load time (e.g. `Quantization::Int4` in [transcribe-rs](https://github.com/andrewleech/transcribe-rs)), which resolves `decoder_init.int4.onnx` with automatic fallback to `decoder_init.onnx` (FP32) for files without a quantized variant.

### Recommended Variants

| Model | Variant | WER | RTF (CPU) | Download |
|---|---|---|---|---|
| **0.6B** | int4 RTN al4 | 5.08% | 0.16x | ~2.6 GB |
| **1.7B** | int4 GPTQ+RTN al4 | 4.25% | 0.37x | ~5.6 GB |
| Parakeet-TDT 0.6B INT8 | reference | 5.45% | 0.16x | — |

The int4 variants use FP32 encoders (`encoder.int4.onnx`). INT8 encoders were tested but degrade 0.6B WER by ~1pp due to quantization of the windowed Conv2D layers.

### All Published Variants

| Model | FP32 | int4 (recommended) |
|---|---|---|
| **0.6B** | ~6.2 GB | ~2.6 GB |
| **1.7B** | ~16 GB | ~5.6 GB |

int4 models use ORT's `MatMulNBitsQuantizer` with per-group scales (block size 64, accuracy_level=4). For 1.7B, GPTQ calibration is used for `decoder_init` (better layer-wise reconstruction); RTN is used for `decoder_step` (no quality difference, avoids impractical KV-cache calibration data).

### WER and Speed (LibriSpeech test-other, 200 samples, CPU)

| Engine | WER | RTF (11s) | RTF (35s) |
|---|---|---|---|
| 1.7B FP32 | 3.79% | ~0.70x | ~0.70x |
| 1.7B int4 GPTQ+RTN al4 | 4.25% | 0.37x | 0.37x |
| 0.6B FP32 | 4.42% | 0.29x | 0.32x |
| **0.6B int4 RTN al4** | **5.08%** | **0.16x** | — |
| 0.6B AWQ INT8 α=0.2 | 5.21% | 0.14x | 0.17x |
| Parakeet-TDT 0.6B INT8 | 5.45% | 0.16x | 0.13x |

RTF < 1.0 = faster than real-time. Lower is better for both WER and RTF. Measured on Ryzen AI 7 PRO 350 (WSL/Linux, ORT 1.22).

**0.6B int4** is the recommended configuration: 5.08% WER at 0.16x RTF, matching Parakeet speed with lower WER and full punctuation output.

**1.7B int4** is the accuracy option: 4.25% WER at 0.37x RTF. AWQ INT8 for 1.7B is not recommended — it causes degraded special token prediction (9% WER).

### GPU Benchmarks (11s JFK audio, 0.6B int4)

| Host | CPU | DirectML | WebGPU |
|---|---|---|---|
| anl (Ryzen AI 7 PRO 350, Radeon 860M iGPU) | **1.77s** (0.16x) | 19.9s (garbage output) | 5.16s (0.47x) |
| step (Ryzen 9 3900X, GTX 1650 SUPER 4 GB) | 2.42s (0.22x) | 2.25s (0.20x) | **1.83s** (0.17x) |

- **NVIDIA (GTX 1650 SUPER):** WebGPU gives 24% speedup over CPU (1.83s vs 2.42s). DirectML works correctly with 7% speedup. All outputs correct.
- **AMD (Radeon 860M iGPU):** DirectML produces garbage output. WebGPU works but 3× slower than CPU. Shared memory architecture limits throughput.
- **CPU int4 on a fast CPU still wins** — anl's Ryzen AI 7 PRO at 1.77s beats step's WebGPU at 1.83s.
- FP16 models are slow on all GPU EPs due to ORT's Cast FP32→FP16 node overhead.
- 1.7B int4 (~5.4 GB) exceeds 4 GB VRAM — GPU acceleration requires 8+ GB.

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

### 3. int4 MatMulNBits Quantization (recommended for both 0.6B and 1.7B)

`quantize_nbits.py` applies ORT's `MatMulNBitsQuantizer` to the decoder files. The encoder uses FP32 (INT8 encoder degrades WER by ~1pp on 0.6B).

```bash
# RTN (no calibration data, fast — adds .int4 files alongside FP32):
python quantize_nbits.py \
    --input output/qwen3-asr-0.6b \
    --output output/qwen3-asr-0.6b \
    --bits 4 --block-size 64 --accuracy-level 4

# Copy FP32 encoder as the int4 encoder:
cp output/qwen3-asr-0.6b/encoder.onnx output/qwen3-asr-0.6b/encoder.int4.onnx
```

`--accuracy-level 4` activates a higher-precision accumulation kernel in ORT that is both faster and more accurate than the default on x86.

For improved load time via GPTQ calibration on decoder_init (1.7B), see the next section.

### 4. GPTQ Calibration + int4 (1.7B, best load time)

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

# Copy FP32 encoder as the int4 encoder
cp output/qwen3-asr-1.7b/encoder.onnx output/qwen3-asr-1.7b/encoder.int4.onnx

# Clean up GPTQ temp files
rm -f output/qwen3-asr-1.7b/*-*-*-*-*.data output/qwen3-asr-1.7b/*_augment.onnx
```

The script handles `config.json` hiding during GPTQ (ORT's neural_compressor calls `AutoConfig.from_pretrained()` which fails on the custom `qwen3_asr` model type) and adds the `com.microsoft` opset import for `MatMulNBits` nodes.

GPTQ writes large UUID-named temporary `.data` files (~6.5 GB each) in the source directory — the cleanup command removes them.

### 5. AWQ Smooth + Quantize (INT8, optional for 0.6B)

INT8 AWQ is not published to HuggingFace but can be built locally for a smaller-footprint variant (1.1 GB, 5.21% WER).

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
```

## Full Reproduction

Complete sequence to produce all output variants from the upstream HuggingFace models. All commands assume `uv run python` (or activate the venv and use `python` directly).

The upstream models are downloaded automatically from HuggingFace on first use and cached in `~/.cache/huggingface/hub/`.

### 0.6B (FP32 + int4)

```bash
# FP32 export (~5 min, ~4 GB RAM)
uv run python export.py --model Qwen/Qwen3-ASR-0.6B

# Validate FP32 ONNX matches PyTorch
uv run python validate.py \
    --onnx-dir output/qwen3-asr-0.6b \
    --audio tests/fixtures/test_audio.wav

# int4 RTN al4 decoders (recommended)
uv run python quantize_nbits.py \
    --input output/qwen3-asr-0.6b \
    --output output/qwen3-asr-0.6b \
    --bits 4 --block-size 64 --accuracy-level 4

# Copy FP32 encoder as the int4 encoder
cp output/qwen3-asr-0.6b/encoder.onnx output/qwen3-asr-0.6b/encoder.int4.onnx
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

# Copy FP32 encoder as the int4 encoder
cp output/qwen3-asr-1.7b/encoder.onnx output/qwen3-asr-1.7b/encoder.int4.onnx

# Clean up GPTQ temp files
rm -f output/qwen3-asr-1.7b/*-*-*-*-*.data output/qwen3-asr-1.7b/*_augment.onnx
```

### WER evaluation

```bash
uv run python evaluate_wer.py \
    --models "0.6B FP32:output/qwen3-asr-0.6b" \
             "0.6B int4:output/qwen3-asr-0.6b" \
             "1.7B FP32:output/qwen3-asr-1.7b" \
             "1.7B int4:output/qwen3-asr-1.7b" \
    --datasets librispeech-other \
    --n-samples 200 --output wer_results.json
```

### Package and upload

```bash
# Package for release (FP32 + int4 + shared metadata)
uv run python package.py --model 0.6b --output release/qwen3-asr-0.6b
uv run python package.py --model 1.7b --output release/qwen3-asr-1.7b

# Upload to HuggingFace (replaces all existing repo contents)
uv run python upload.py \
    --input release/qwen3-asr-0.6b \
    --repo andrewleech/qwen3-asr-0.6b-onnx \
    --model Qwen/Qwen3-ASR-0.6B

uv run python upload.py \
    --input release/qwen3-asr-1.7b \
    --repo andrewleech/qwen3-asr-1.7b-onnx \
    --model Qwen/Qwen3-ASR-1.7B
```

### Output structure

After full reproduction:

```
output/
├── qwen3-asr-0.6b/              # All 0.6B variants in one directory
│   ├── encoder.onnx              # FP32
│   ├── encoder.int4.onnx         # FP32 (copy of encoder.onnx)
│   ├── decoder_init.onnx         # FP32
│   ├── decoder_init.onnx.data
│   ├── decoder_init.int4.onnx    # int4 RTN al4
│   ├── decoder_init.int4.onnx.data
│   ├── decoder_step.onnx         # FP32
│   ├── decoder_step.onnx.data
│   ├── decoder_step.int4.onnx    # int4 RTN al4
│   ├── decoder_step.int4.onnx.data
│   ├── embed_tokens.bin          # Shared across all variants
│   ├── config.json
│   └── tokenizer.json
├── qwen3-asr-1.7b/              # All 1.7B variants in one directory
│   ├── encoder.onnx              # FP32
│   ├── encoder.int4.onnx         # FP32 (copy of encoder.onnx)
│   ├── decoder_init.onnx         # FP32
│   ├── decoder_init.onnx.data
│   ├── decoder_init.int4.onnx    # int4 GPTQ
│   ├── decoder_init.int4.onnx.data
│   ├── decoder_step.onnx         # FP32
│   ├── decoder_step.onnx.data
│   ├── decoder_step.int4.onnx    # int4 RTN al4
│   ├── decoder_step.int4.onnx.data
│   ├── embed_tokens.bin          # Shared across all variants
│   ├── config.json
│   └── tokenizer.json
└── calibration_cache/            # GPTQ calibration data (reusable)
```

Each model directory is self-contained. For distribution, the int4 variant files are: `encoder.int4.onnx` + `decoder_init.int4.onnx*` + `decoder_step.int4.onnx*` + `embed_tokens.bin` + `config.json` + `tokenizer.json`. The FP32 files are optional extras for users who need maximum accuracy.

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
| 0.6B | 4.42% | 6.7% | 5.21% | 5.08% | — |
| 1.7B | 3.79% | — | 9.0% | 4.3% | 4.25% |

AWQ smoothing reduces the 0.6B INT8 WER penalty from +2.3pp (naive) to +0.8pp (α=0.2). 1.7B AWQ INT8 is not recommended — outlier weights in the 1.7B decoder cause special token prediction failures under per-tensor INT8 quantization regardless of smoothing. int4 MatMulNBits with per-group scales handles these outliers locally; the 1.7B int4 WER penalty is only +0.46pp vs FP32.

## DirectML Compatibility

The exported ONNX graphs are patched post-export to remove `allowzero=1` from all `Reshape` nodes. DirectML rejects this attribute, and no shape tensor in the graphs contains a literal zero dimension, so removal is safe. This is done automatically during export.

Note: DirectML produces incorrect output on AMD Radeon iGPUs (tested on 860M). NVIDIA GPUs work correctly with both DirectML and WebGPU.

## Tests

```bash
python -m pytest tests/
```

Audio fixtures in `tests/fixtures/` are tracked with Git LFS. Run `git lfs pull` after cloning to fetch them. See `tests/fixtures/README.md` for provenance.

## License

Apache 2.0, matching the upstream Qwen3-ASR models.
