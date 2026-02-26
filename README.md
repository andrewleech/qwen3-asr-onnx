# Qwen3-ASR ONNX Export Tool

Export [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) to ONNX format for use with ONNX Runtime.

## Output Files

Each export produces a directory containing:

| File | Description |
|---|---|
| `encoder.onnx` (+`.data`) | Audio encoder (Conv2D stem + transformer + MLP projection) |
| `decoder_init.onnx` (+`.data`) | Decoder prefill (full sequence, outputs KV cache) |
| `decoder_step.onnx` (+`.data`) | Decoder autoregressive step (single token + KV cache) |
| `embed_tokens.bin` | Token embedding matrix [vocab_size, hidden_size] |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config + special tokens + mel params |

The `.onnx.data` files contain the model weights (external data format) and must be kept alongside the `.onnx` files.

### FP32 vs INT8

The export tool produces FP32 models. An optional quantization step produces INT8 models.

Sizes below are for 0.6B; 1.7B models are proportionally larger.

|  | FP32 | INT8 |
|---|---|---|
| **Directory** | `output/qwen3-asr-0.6b/` | `output/qwen3-asr-0.6b-int8/` |
| **Encoder** | 718 MB | 552 MB |
| **Decoder** (init+step) | 2x 2.3 GB | 2x 1.7 GB |
| **Embeddings** | 594 MB (float32) | 297 MB (float16) |
| **Total disk** | ~5.9 GB | ~4.2 GB |
| **Accuracy (≤12s)** | Exact match with native model | Exact match with FP32 |
| **Accuracy (30s+)** | Exact match with native model | Degrades (0-79% token match) |
| **Execution providers** | Any (CPU, CUDA, DirectML, CoreML) | Requires provider with ConvInteger/MatMulInteger support (CUDA, DirectML). **Does not work with CPUExecutionProvider.** |

**Use FP32 unless you have a specific reason to use INT8.** FP32 works on all execution providers and produces accurate results at all audio durations. The INT8 size reduction (~30%) comes with significant accuracy loss on longer audio and restricted provider compatibility.

The decoder weights are the same model exported as two graphs (prefill and autoregressive), so runtime memory is roughly half the total disk size.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Export

```bash
# 0.6B (default)
python export.py --model Qwen/Qwen3-ASR-0.6B

# 1.7B
python export.py --model Qwen/Qwen3-ASR-1.7B
```

Output directories are derived from the model name (`output/qwen3-asr-0.6b`, `output/qwen3-asr-1.7b`) unless overridden with `--output`.

### Validate

```bash
python validate.py --onnx-dir output/qwen3-asr-0.6b --audio tests/fixtures/test_audio.wav
```

### Quantize (INT8)

```bash
python quantize.py --input output/qwen3-asr-0.6b --output output/qwen3-asr-0.6b-int8
```

### Upload to HuggingFace

```bash
python upload.py --input output/qwen3-asr-0.6b --repo andrewleech/qwen3-asr-0.6b-onnx
```

### Compare inference paths

```bash
python compare.py --audio tests/fixtures/librispeech_0.wav tests/fixtures/librispeech_30s.wav \
    --ground-truth "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL" \
    "LAW SEEMED TO HIM WELL ENOUGH AS A SCIENCE..."
```

Runs transcription through 4 paths (native model, PyTorch wrapper, FP32 ONNX, INT8 ONNX) and reports text, timing, token agreement, and encoder feature differences.

## Architecture

- **Encoder**: mel `[1, 128, T]` → windowed Conv2D (100-frame windows, 3x stride-2 convs producing 13 tokens per window) → windowed attention (104-token windows across transformer layers) → MLP projection → `[1, N, output_dim]` where `N = get_feat_extract_output_lengths(T)`. 0.6B: 18 layers, d=896, output_dim=1024. 1.7B: 24 layers, d=1024, output_dim=2048.
- **Decoder**: Qwen3 (28 layers, 16 Q-heads / 8 KV-heads, GQA, SwiGLU, MRoPE). 0.6B: d=1024. 1.7B: d=2048.
- **Integration**: Prefix LM — encoder output replaces `<|audio_pad|>` token embeddings. Decoder uses self-attention only.

The encoder uses windowed processing from the native Qwen3-ASR architecture: mel frames are split into 100-frame convolution windows, each producing 13 tokens via a 3-layer stride-2 Conv2D stack. Tokens are then grouped into 104-token attention windows for the transformer layers. Padding tokens in the final window are masked out.

## Mel Spectrogram Parameters

Identical to Whisper: 16kHz, 128 bins, 25ms window (n_fft=400), 10ms hop (hop_length=160), Hann window, Slaney mel scale, 0-8kHz.

## Comparison Results

Tested against the native Qwen3-ASR model on LibriSpeech samples (5s to 35s).

| Duration | Native vs FP32 | Wrapper vs FP32 | INT8 vs FP32 | Encoder max_diff |
|----------|---------------|-----------------|-------------|-----------------|
| 5-12s | Exact token match | Exact token match | Exact token match | < 6e-6 |
| 30s | Exact token match | Exact token match | ~79% tokens | < 6e-6 |
| 32-35s | Same words, punctuation differs | Exact token match | Degraded | < 6e-6 |

- **Wrapper vs FP32**: Always exact token match — the ONNX export faithfully reproduces the PyTorch wrapper.
- **Native vs FP32**: Identical on short audio. On long audio (>30s), SDPA vs eager attention causes punctuation-level divergence; word content matches.
- **INT8**: Exact on short audio. Degrades on long audio due to quantization error accumulation.

## Tests

```bash
python -m pytest tests/
```

Audio fixtures in `tests/fixtures/` are tracked with Git LFS. Run `git lfs pull` after cloning to fetch them. See `tests/fixtures/README.md` for provenance.
