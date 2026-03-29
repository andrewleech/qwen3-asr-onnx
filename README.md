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
| `encoder.int4.onnx` | Encoder for int4 variant (FP32 for 0.6B, native FP16 for 1.7B — see experiment [110]) |
| `decoder_init.onnx` | FP32 decoder prefill (full sequence, outputs KV cache) |
| `decoder_step.onnx` | FP32 decoder autoregressive step (single token + KV cache) |
| `decoder_weights.data` | Shared external weights for both FP32 decoders |
| `decoder_init.int4.onnx` | int4 decoder_init variant |
| `decoder_step.int4.onnx` | int4 decoder_step variant |
| `decoder_weights.int4.data` | Shared external weights for both int4 decoders |
| `embed_tokens.bin` | Token embedding matrix `[vocab_size, hidden_size]`, FP16 (see below and `embed_tokens_dtype` in config.json) |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config + special tokens + mel params |

The FP32 decoder `.onnx` files are small graph protos (~2 MB each) referencing a single shared `decoder_weights.data` file. ORT memory-maps this file once. int4 decoders use the same shared-weights pattern via `decoder_weights.int4.data`. Encoder weights are always inlined.

**Why `embed_tokens.bin` exists separately:** The two decoder models use different input strategies. `decoder_init` (prefill) accepts `input_ids` and has the embedding table built into its ONNX graph — this allows it to handle the audio feature scatter internally. `decoder_step` (autoregressive) accepts pre-looked-up `input_embeds` instead, keeping the embedding table out of its graph. The consumer loads `embed_tokens.bin` once at startup and performs the embedding lookup per token before calling `decoder_step`. This avoids duplicating the embedding weights across both decoder files while keeping `decoder_step` small for fast loading.

The file is stored as FP16 (half the FP32 size, zero measured WER impact). The `embed_tokens_dtype` field in `config.json` indicates the storage format. Consumers should cast to FP32 after loading for inference.

Multiple quantization variants coexist in a single directory. The consumer selects a variant at load time (e.g. `Quantization::Int4` in [transcribe-rs](https://github.com/andrewleech/transcribe-rs)), which resolves `decoder_init.int4.onnx` with automatic fallback to `decoder_init.onnx` (FP32) for files without a quantized variant.

### Recommended Variants

| Model | Variant | WER | RTF (CPU) | tar.gz download |
|---|---|---|---|---|
| **0.6B** | int4 RTN al4 | 5.16% | 0.17x | 1.3 GB |
| **1.7B** | int4 RTN al4 | 4.20% | 0.32x | 2.5 GB |
| Parakeet-TDT 0.6B INT8 | reference | 5.45% | 0.16x | — |

WER measured on LibriSpeech test-other (200 samples). RTF < 1.0 = faster than real-time.

#### Package contents — int4 (recommended)

Each int4 package contains the files needed for quantized inference:

| Component | 0.6B | 1.7B | Format | Why this format |
|---|---|---|---|---|
| `encoder.int4.onnx` | 717 MB (FP32) | 609 MB (FP16) | **see notes** | 0.6B uses FP32 (FP16 adds 8.7% inference overhead). 1.7B uses native FP16 (size savings outweighs small overhead) |
| `decoder_init.int4.onnx` | 2 MB | 2 MB | **int4** | Graph proto only — weights in shared data file |
| `decoder_step.int4.onnx` | 87 MB | 171 MB | **int4** | Graph proto + inlined lm_head weights (unmatched between init/step) |
| `decoder_weights.int4.data` | 833 MB | 1,953 MB | **int4** | Shared external weights for both decoders (RTN al4 bs64) |
| `embed_tokens.bin` | 296 MB | 593 MB | **FP16** | Zero measured WER impact vs FP32. Consumer casts to FP32 at lookup time |
| `config.json` + `tokenizer.json` | ~11 MB | ~11 MB | — | Architecture config, special tokens, mel params, tokenizer |

Uncompressed total: 0.6B = 2.0 GB, 1.7B = 3.3 GB.

The encoder uses a native FP16 export via `torch.amp.autocast` tracing. Unlike post-hoc FP16 conversion (which inserts Cast nodes and breaks quality), autocast tracing captures native FP16 ops while keeping precision-sensitive operations (LayerNorm, softmax) in FP32. The attention mask uses `torch.finfo(float16).min` = -65504 (valid FP16), avoiding the overflow that breaks post-hoc conversion. Result: half the FP32 encoder size with zero measured WER impact (verified on both 0.6B and 1.7B, 200-sample LibriSpeech test-other).

#### Package contents — FP32

FP32 packages contain unquantized models for maximum accuracy:

| Component | 0.6B | 1.7B | Format |
|---|---|---|---|
| `encoder.onnx` | 716 MB | 1,217 MB | FP32 |
| `decoder_init.onnx` | ~2 MB | ~2 MB | FP32 (graph proto) |
| `decoder_step.onnx` | ~2 MB | ~2 MB | FP32 (graph proto) |
| `decoder_weights.data` | 2,273 MB | 6,563 MB | Shared external weights (loaded once) |
| `embed_tokens.bin` | 297 MB | 593 MB | FP16 |
| `config.json` + `tokenizer.json` | ~11 MB | ~11 MB | — |

The two decoder protos (~2 MB each) reference offsets in the single shared `decoder_weights.data` file. ORT memory-maps it once regardless of how many sessions use it.

### WER and Speed (LibriSpeech test-other, 200 samples, CPU)

| Engine | WER | RTF (11s) | RTF (35s) |
|---|---|---|---|
| 1.7B FP32 | 3.79% | ~0.70x | ~0.70x |
| 1.7B int4 RTN al4 | 4.20% | 0.32x | — |
| 0.6B FP32 | 4.42% | 0.29x | 0.32x |
| **0.6B int4 RTN al4** | **5.16%** | **0.17x** | — |
| 0.6B AWQ INT8 α=0.2 | 5.21% | 0.14x | 0.17x |
| Parakeet-TDT 0.6B INT8 | 5.45% | 0.16x | 0.13x |

RTF measured on Ryzen AI 7 PRO 350 (native Windows, ORT 2.0 rc12). WER from 200-sample LibriSpeech test-other. Lower is better for both.

### GPU Benchmarks (11s JFK audio, 0.6B int4)

| Host | CPU | DirectML | WebGPU |
|---|---|---|---|
| anl (Ryzen AI 7 PRO 350, Radeon 860M iGPU) | **1.85s** (0.17x) | 19.9s (garbage output) | 5.16s (0.47x) |
| step (Ryzen 9 3900X, GTX 1650 SUPER 4 GB) | 2.42s (0.22x) | 2.25s (0.20x) | **1.83s** (0.17x) |

- **NVIDIA (GTX 1650 SUPER):** WebGPU gives 24% speedup over CPU (1.83s vs 2.42s). DirectML works correctly with 7% speedup. All outputs correct.
- **AMD (Radeon 860M iGPU):** DirectML produces garbage output. WebGPU works but 3× slower than CPU. Shared memory architecture limits throughput.
- **CPU int4 on a fast CPU still wins** — anl's Ryzen AI 7 PRO at 1.77s beats step's WebGPU at 1.83s.
- FP16 models are slow on all GPU EPs due to ORT's Cast FP32→FP16 node overhead.
- 1.7B int4 (~3.3 GB) fits in 4 GB VRAM but leaves little headroom for KV cache — GPU acceleration for 1.7B may require 6+ GB.

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

`quantize_nbits.py` applies ORT's `MatMulNBitsQuantizer` to the decoder files. The encoder uses a native FP16 export (half the size of FP32, zero WER impact).

```bash
# RTN (no calibration data, fast — adds .int4 files alongside FP32):
python quantize_nbits.py \
    --input output/qwen3-asr-0.6b \
    --output output/qwen3-asr-0.6b \
    --bits 4 --block-size 64 --accuracy-level 4

# Copy FP32 encoder as the int4 encoder (FP16 adds 8.7% inference overhead on 0.6B, see [110]):
cp output/qwen3-asr-0.6b/encoder.onnx output/qwen3-asr-0.6b/encoder.int4.onnx
```

`--accuracy-level 4` activates a higher-precision accumulation kernel in ORT that is both faster and more accurate than the default on x86.

For 1.7B, use native FP16 encoder instead (size savings outweighs the small overhead at that scale).

GPTQ calibration is also supported but provides no WER benefit over RTN (see section 4).

### 4. GPTQ Calibration (optional, not recommended)

GPTQ calibration via `collect_gptq_calib.py` + `quantize_nbits.py --algo gptq` is supported for experimentation, but testing showed no WER benefit over RTN for either model size (experiment [109]). RTN-only quantization (section 3) is recommended for both 0.6B and 1.7B.

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

# Copy FP32 encoder as the int4 encoder (FP16 adds inference overhead on 0.6B)
cp output/qwen3-asr-0.6b/encoder.onnx output/qwen3-asr-0.6b/encoder.int4.onnx

# Convert embed_tokens to FP16 (halves file size, zero WER impact)
uv run python convert_embed_fp16.py --model-dir output/qwen3-asr-0.6b
```

### 1.7B (FP32 + int4)

```bash
# FP32 export (~15 min, ~10 GB RAM)
uv run python export.py --model Qwen/Qwen3-ASR-1.7B

# Validate
uv run python validate.py \
    --onnx-dir output/qwen3-asr-1.7b \
    --audio tests/fixtures/test_audio.wav

# int4 RTN al4 decoders (same as 0.6B — GPTQ provides no WER benefit, see experiment [109])
uv run python quantize_nbits.py \
    --input output/qwen3-asr-1.7b \
    --output output/qwen3-asr-1.7b \
    --bits 4 --block-size 64 --accuracy-level 4

# Native FP16 encoder (autocast export — half size, zero WER impact)
uv run python export_encoder_native_fp16.py \
    --model Qwen/Qwen3-ASR-1.7B \
    --output output/qwen3-asr-1.7b/encoder.int4.onnx

# Convert embed_tokens to FP16
uv run python convert_embed_fp16.py --model-dir output/qwen3-asr-1.7b
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
│   ├── encoder.int4.onnx         # FP32 (copy of encoder.onnx — FP16 adds overhead, see [110])
│   ├── decoder_init.onnx         # FP32
│   ├── decoder_init.onnx.data
│   ├── decoder_init.int4.onnx    # int4 RTN al4
│   ├── decoder_init.int4.onnx.data
│   ├── decoder_step.onnx         # FP32
│   ├── decoder_step.onnx.data
│   ├── decoder_step.int4.onnx    # int4 RTN al4
│   ├── decoder_step.int4.onnx.data
│   ├── embed_tokens.bin          # FP16 embedding table (see embed_tokens_dtype in config.json)
│   ├── config.json
│   └── tokenizer.json
├── qwen3-asr-1.7b/              # All 1.7B variants in one directory
│   ├── encoder.onnx              # FP32
│   ├── encoder.int4.onnx         # Native FP16 (autocast export, FP32 I/O — size savings worth it at 1.7B)
│   ├── decoder_init.onnx         # FP32
│   ├── decoder_init.onnx.data
│   ├── decoder_init.int4.onnx    # int4 RTN al4
│   ├── decoder_init.int4.onnx.data
│   ├── decoder_step.onnx         # FP32
│   ├── decoder_step.onnx.data
│   ├── decoder_step.int4.onnx    # int4 RTN al4
│   ├── decoder_step.int4.onnx.data
│   ├── embed_tokens.bin          # FP16 embedding table (see embed_tokens_dtype in config.json)
│   ├── config.json
│   └── tokenizer.json
```

Each model directory is self-contained. For distribution, the int4 variant files are: `encoder.int4.onnx` + `decoder_init.int4.onnx` + `decoder_step.int4.onnx` + `decoder_weights.int4.data` + `embed_tokens.bin` (FP16) + `config.json` + `tokenizer.json`. The FP32 files are optional extras for users who need maximum accuracy.

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

| Model | FP32 WER | INT8 naive | INT8 AWQ α=0.2 | int4 RTN al4 |
|---|---|---|---|---|
| 0.6B | 4.42% | 6.7% | 5.21% | **5.16%** |
| 1.7B | 3.79% | — | 9.0% | **4.20%** |

AWQ smoothing reduces the 0.6B INT8 WER penalty from +2.3pp (naive) to +0.8pp (α=0.2). 1.7B AWQ INT8 is not recommended — outlier weights in the 1.7B decoder cause special token prediction failures under per-tensor INT8 quantization regardless of smoothing. int4 MatMulNBits with per-group scales handles these outliers locally; the 1.7B int4 WER penalty is only +0.41pp vs FP32.

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
