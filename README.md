# Qwen3-ASR ONNX Export Tool

Export [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) to ONNX format for use with ONNX Runtime.

## Output Files

| File | Description |
|---|---|
| `encoder.onnx` | Audio encoder (Conv2D stem + transformer + MLP projection) |
| `decoder_init.onnx` | Decoder prefill (full sequence, outputs KV cache) |
| `decoder_step.onnx` | Decoder autoregressive step (single token + KV cache) |
| `embed_tokens.bin` | Token embedding matrix [151936, 1024] as raw float32 |
| `tokenizer.json` | HuggingFace tokenizer |
| `config.json` | Architecture config + special tokens + mel params |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Export

```bash
python export.py --model Qwen/Qwen3-ASR-0.6B --output output/qwen3-asr-0.6b
```

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

## Architecture

- **Encoder**: mel `[1, 128, T]` → 3x Conv2D (stride 2, 8x downsample) → 18 transformer layers (d=896) → MLP(896→1024) → `[1, T/8, 1024]`
- **Decoder**: Qwen3-0.6B (28 layers, d=1024, 16 Q-heads / 8 KV-heads, GQA, SwiGLU, MRoPE)
- **Integration**: Prefix LM — encoder output replaces `<|audio_pad|>` token embeddings. Decoder uses self-attention only.

## Mel Spectrogram Parameters

Identical to Whisper: 16kHz, 128 bins, 25ms window (n_fft=400), 10ms hop (hop_length=160), Hann window, Slaney mel scale, 0-8kHz.

## Tests

```bash
python -m pytest tests/
```
