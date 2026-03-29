# Tool Reference

All commands run from `~/qwen3-asr-onnx` with `uv run python`.

## evaluate_wer.py — WER measurement

```bash
uv run python evaluate_wer.py \
  --models "label:path[:quant]" ["label2:path2[:quant2]" ...] \
  --datasets librispeech-other \
  --n-samples 200 \
  --output results.json
```

- `quant`: `int4`, `int8`, or omit for FP32. Resolves `encoder.{quant}.onnx` etc.
- Streams audio from HuggingFace (no full dataset download)
- Output: per-sample WER lines + summary table at end
- Buffered stdout — check `wc -c` on output file for progress

## quantize_nbits.py — INT4 MatMulNBits quantization

```bash
uv run python quantize_nbits.py \
  --input output/qwen3-asr-0.6b \
  --output output/qwen3-asr-0.6b \
  --bits 4 --block-size 64 --accuracy-level 4 \
  [--algo gptq --calib-data calibration_cache/file.npz] \
  [--decoders decoder_init]  # or decoder_step, or both (default)
```

- `--accuracy-level 4`: faster INT8 accumulation kernel, better WER
- `--block-size 64`: only validated size (32 is catastrophic on ORT)
- `--algo gptq`: requires `--calib-data` from `collect_gptq_calib.py`
- `--decoders`: which decoders to quantize (default: both)
- Writes suffixed files (`decoder_init.int4.onnx` + `.data`) alongside FP32
- GPTQ creates ~6.5 GB temp `.data` files in source dir — clean up after

## collect_gptq_calib.py — GPTQ calibration data

```bash
uv run python collect_gptq_calib.py \
  --model output/qwen3-asr-0.6b \
  --n-samples 32 --decoder-steps 8 \
  --output calibration_cache/0.6b_gptq_init.npz \
  --target decoder_init
```

## share_weights.py — deduplicate decoder weights

```bash
uv run python share_weights.py output/qwen3-asr-0.6b [--suffix int4]
```

- Hash-matches tensors between decoder_init and decoder_step
- Creates shared `decoder_weights[.int4].data` file
- Only works when both decoders use same quantization method (RTN+RTN, not GPTQ+RTN)

## quantize.py — dynamic INT8 quantization

```bash
uv run python quantize.py \
  --input output/qwen3-asr-0.6b-smooth \
  --output output/qwen3-asr-0.6b
```

- Adds `.int8` suffixed files alongside FP32
- Best after AWQ smoothing (awq_smooth.py)

## awq_smooth.py — AWQ activation smoothing

```bash
uv run python awq_smooth.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --output output/qwen3-asr-0.6b-smooth \
  --alpha 0.2 --n-samples 128 --verify \
  [--activations-cache output/.../calibration_activations.npz]
```

- alpha=0.2 is optimal for 0.6B INT8
- Cached activations reusable across alpha sweeps

## export.py — FP32 ONNX export from PyTorch

```bash
uv run python export.py --model Qwen/Qwen3-ASR-0.6B
uv run python export.py --model Qwen/Qwen3-ASR-1.7B
```

## export_encoder_native_fp16.py — native FP16 encoder

```bash
uv run python export_encoder_native_fp16.py \
  --model Qwen/Qwen3-ASR-0.6B \
  --output output/qwen3-asr-0.6b/encoder.fp16.native.onnx
```

- Uses `torch.amp.autocast` for native mixed-precision tracing
- Zero WER impact but 9-13% CPU inference regression from I/O Cast nodes ([110])
- Useful for GPU targets only

## convert_embed_fp16.py — FP16 embed_tokens

```bash
uv run python convert_embed_fp16.py --model-dir output/qwen3-asr-0.6b
```

- Creates `embed_tokens.fp16.bin` from `embed_tokens.bin`
- Zero WER impact ([104])

## validate.py — FP32 ONNX vs PyTorch comparison

```bash
uv run python validate.py \
  --onnx-dir output/qwen3-asr-0.6b \
  --audio tests/fixtures/test_audio.wav
```

## package.py — release packaging

```bash
uv run python package.py --model 0.6b --output release/qwen3-asr-0.6b [--hardlink]
```

## Key experimental findings to avoid re-testing

- block_size=32: catastrophic (99.98% WER) — ORT kernel bug [106]
- GPTQ on 0.6B: degrades WER by 0.85pp vs RTN [105]
- FP16 weight-only encoder: 100% WER — attention mask overflow [107]
- Native FP16 encoder: zero WER impact but 9-13% CPU slowdown [108][110]
- INT8 encoder: +1pp WER degradation [100]
- INT4 encoder: +0.17pp WER degradation [104]
- FP16 embed_tokens: zero WER impact [104]
- GPTQ vs RTN on 1.7B: equivalent WER (+0.04pp noise) [109]
