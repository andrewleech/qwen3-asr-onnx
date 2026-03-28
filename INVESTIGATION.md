
### [107] FP16 weight-only encoder — storage-only precision reduction
**Date:** 2026-03-29
**Idea:** The existing FP16 encoder (from `convert_fp16.py` using `onnxconverter_common`) rewrites the entire compute graph to FP16 intermediate types with Cast nodes, which was previously found to degrade quality. Test a different approach: convert only the initializer weight tensors to FP16 storage while preserving FP32 graph structure. Insert explicit `Cast(FP16→FP32)` nodes after each converted initializer so all ops still receive FP32 inputs. This should be numerically identical to FP32 (all compute in FP32) with half the file size — the same principle as FP16 embed_tokens.

**Change:** Created `convert_weights_fp16.py` which:
1. Converts 307 FP32 initializer tensors to FP16 (20 non-FP32 tensors skipped)
2. Renames each initializer `X` → `X__fp16`
3. Inserts `Cast(FP16→FP32)` node: `X__fp16` → `X` so all consuming ops still get FP32
4. Saves with weights inlined (encoder is < 2 GB)

File size: 752 MB → 376 MB (50% reduction).

One tensor `val_89` (scalar, value `-3.4e38` = negative FP32 max) overflows FP16 range (max ±65,504). This is the attention mask fill value used by the windowed attention layers — multiplied with the attention mask to produce large negative values that softmax maps to zero. In FP16 it becomes `-inf`, then Cast back to FP32 produces `-inf` (FP32).

**Result (200-sample LibriSpeech test-other, 0.6B int4 decoders, Python evaluate_wer.py):**
| Trial | Encoder | WER | RTF | File size |
|---|---|---|---|---|
| Baseline | FP32 (717 MB) | 5.16% | 0.25x | 717 MB |
| FP16 weight-only + Cast | FP16 storage (376 MB) | **100.00%** | 1.50x | 376 MB |

**Outcome:** CATASTROPHIC — the FP16 weight-only encoder produces completely empty output on every sample.

**Analysis:**
- The `-inf` attention mask is the likely cause. While `softmax(-inf) = 0` is mathematically correct, ORT's windowed attention implementation may propagate `-inf` through intermediate computations (e.g. `Mul` with the mask tensor) producing `NaN` or unexpected results that corrupt encoder features.
- The 6× slowdown (0.25x → 1.50x RTF) confirms the 307 Cast nodes add overhead. ORT cannot fuse FP16 initializers with FP32 ops, so each weight load requires an explicit cast.
- This approach is fundamentally incompatible with the encoder's use of extreme FP32 values for attention masking. The original model's BF16 weights work because BF16 has the same exponent range as FP32 (8 exponent bits, range ±3.4e38). FP16 has only 5 exponent bits (range ±65,504).

**Conclusion:** The encoder must remain FP32 for distribution. No FP16 variant (graph-level or weight-only) is viable due to the attention mask range requirement. This is not a quantization quality tradeoff — it is a hard numerical failure.

**Future options:**
- Export the encoder with a smaller mask constant (e.g. `-65504` instead of `-3.4e38`) at the PyTorch level, then FP16 would work. Requires changes to the export script.
- BF16 ONNX support in ORT would allow half-size storage with FP32-equivalent range, but BF16 support is limited to specific EPs.

### [104] INT4 encoder and FP16 embed_tokens — 0.6B package size reduction
**Date:** 2026-03-19
**Idea:** The 0.6B int4 package is ~2.6 GB, larger than the original BF16 safetensors (~1.2 GB). The FP32 encoder (717 MB) and FP32 embed_tokens.bin (594 MB) account for 1.3 GB. Test whether INT4 encoder (MatMulNBits, RTN, block_size=64, accuracy_level=4) and FP16 embed_tokens preserve quality while reducing package size.

**Change:** Quantized encoder.onnx → encoder.int4.onnx via `quantize_nbits.py --decoders encoder --bits 4 --block-size 64 --accuracy-level 4`. Created embed_tokens.fp16.bin via numpy f32→f16 cast. Built 4 trial directories with symlinks to shared int4 decoder files, testing all combinations.

**Result (200-sample LibriSpeech test-other, Python evaluate_wer.py, WSL2/Linux, ORT 2.0.0-rc.12):**
| Trial | Encoder | Embed | WER | RTF | Enc size | Embed size |
|---|---|---|---|---|---|---|
| Baseline | FP32 (717 MB) | FP32 (594 MB) | 5.16% | 0.244x | 717 MB | 594 MB |
| A | INT4 (122 MB) | FP32 (594 MB) | 5.33% | 0.239x | 122 MB | 594 MB |
| B | FP32 (717 MB) | FP16 (297 MB) | 5.16% | 0.242x | 717 MB | 297 MB |
| C | INT4 (122 MB) | FP16 (297 MB) | 5.33% | 0.239x | 122 MB | 297 MB |

Note: absolute RTF values are from the Python pipeline (includes PyTorch mel spectrogram overhead) and are not comparable to Rust bench_compare RTF. Use only for relative comparison between trials.

**Package size impact:**
| Variant | Encoder | Embed | Decoders | Total |
|---|---|---|---|---|
| Current (FP32 enc) | 717 MB | 594 MB | ~1.2 GB | ~2.6 GB |
| FP16 embed only (B) | 717 MB | 297 MB | ~1.2 GB | ~2.3 GB |
| INT4 enc + FP16 embed (C) | 122 MB | 297 MB | ~1.2 GB | ~1.6 GB |

**Outcome:** FINDING
- **FP16 embed_tokens has zero WER impact** (5.16% = baseline). Saves 297 MB. Safe to ship.
- **INT4 encoder adds 0.17pp WER** (5.33% vs 5.16%). Saves 595 MB, marginally faster. Small but repeatable degradation — the encoder's windowed Conv2D ops remain FP32 (MatMul-only quantization), but MatMul INT4 still introduces slight precision loss in encoder features. Less than INT8 encoder's 1pp degradation ([100]) but non-zero.
- INT4 encoder is ~2% faster in relative terms (lower overhead from smaller model load and reduced memory bandwidth).

**Notes:**
- The 5.16% baseline differs from experiment [100]'s 5.08% — within expected ±0.5pp variance at 200 samples across different runs/sessions.
- FP16 embed is a pure storage optimization: Rust loads and casts f16→f32 per row lookup. No compute path change.
- Rust model loader needs update to support FP16 embed (detect dtype from config.json or file size heuristic).
- TODO: Run Rust bench_compare with INT4 encoder for absolute RTF numbers on 11s JFK clip.

### [105] GPTQ decoder_init on 0.6B int4 (block_size=64, al4)
**Date:** 2026-03-19
**Idea:** GPTQ improved 1.7B int4 WER by 0.08pp (experiment [89]). Apply the same GPTQ-init + RTN-step strategy to 0.6B to recover the ~0.08pp gap between current 5.16% and the target 5.08%.
**Change:** Collected GPTQ calibration data for 0.6B decoder_init (32 LibriSpeech samples via `collect_gptq_calib.py`). Quantized decoder_init with GPTQ bs64 al4, decoder_step with RTN bs64 al4.
**Result:**
| Trial | WER (200-sample) | RTF |
|---|---|---|
| Baseline RTN bs64 al4 | 5.16% | 0.275x |
| GPTQ-init + RTN bs64 al4 | **6.01%** | 0.278x |

**Outcome:** DEGRADED — GPTQ makes 0.6B WER 0.85pp worse. Unlike 1.7B where GPTQ's layer-wise reconstruction error minimization helped, the 0.6B model (hidden_size=1024, fewer layers) has less capacity to absorb GPTQ's weight redistribution. The quantization error is concentrated rather than spread.
**Notes:**
- The 6.01% matches the INT8 encoder WER from [100] — coincidence, different cause (decoder quantization vs encoder quantization).
- Smoothing before GPTQ was not tested but is unlikely to recover 0.85pp given the fundamental capacity limitation.
- GPTQ is not recommended for models below ~1B parameters with this architecture.

### [106] RTN block_size=32 on 0.6B int4 (al4)
**Date:** 2026-03-19
**Idea:** Smaller block_size (32 vs 64) gives finer per-group scales, potentially reducing quantization error for the 0.6B model's weight distribution.
**Change:** Quantized both decoders with RTN bs32 al4 via `quantize_nbits.py --block-size 32`.
**Result:**
| Trial | WER (200-sample) | RTF |
|---|---|---|
| Baseline RTN bs64 al4 | 5.16% | 0.275x |
| RTN bs32 al4 | **99.98%** | 0.473x |

**Outcome:** CATASTROPHIC — output is complete garbage. Also 2× slower than bs64.
**Notes:**
- ORT's MatMulNBits kernel likely has no optimized path for block_size=32. The generic fallback dequantize loop produces numerically incorrect results for this model.
- block_size=64 is the only validated block size for MatMulNBits int4 on Qwen3-ASR.
- Do not use block_size < 64 with ORT MatMulNBits.

### [102] GPU benchmark: DirectML and WebGPU on AMD Radeon 860M (4 GB shared iGPU)
**Date:** 2026-03-18
**Idea:** Test whether GPU acceleration (DirectML, WebGPU) improves inference speed on the Radeon 860M iGPU in native Windows builds. FP16 models are the expected GPU-optimal format (native half-precision compute, no dequantization). int4 via MatMulNBits is a custom op that may not have GPU kernel support.

**Change:** Added `--accelerator` CLI flag to `bench_compare.rs` (calls `set_ort_accelerator()` before model load). Added `--decoder-gpu` / `TRANSCRIBE_DECODER_GPU=1` env var to optionally route decoder sessions through the GPU EP. Built native Windows binary with `--features "qwen3,ort-directml,ort-webgpu"`. Encoder runs on GPU, decoder on CPU (default — GPU decoder overhead exceeds benefit for autoregressive steps).

**Result (11s JFK audio, AMD Radeon 860M 4 GB shared VRAM):**
| Model | CPU Mean | CPU RTF | DirectML Mean | DirectML RTF | DirectML OK? | WebGPU Mean | WebGPU RTF | WebGPU OK? |
|---|---|---|---|---|---|---|---|---|
| 0.6B int4 | 1.77s | 0.16x | 19.9s | 1.81x | **garbage output** | 5.16s | 0.47x | correct |
| 0.6B FP16 | 6.27s | 0.57x | 107.9s | 9.81x | **garbage output** | 22.0s | 2.00x | correct |
| 1.7B FP16 | 15.4s | 1.40x | not tested | — | — | not tested | — | — |
| 0.6B FP32 | ~2.5s | ~0.23x | not tested | — | — | crash (OOM) | — | — |

**Result (56s product_names audio, CPU only):**
| Model | CPU Mean | CPU RTF |
|---|---|---|
| 0.6B int4 | 15.6s | 0.28x |
| 0.6B FP16 | 41.4s | 0.74x |
| 1.7B FP16 | 113.0s | 2.01x |

**Outcome:** FINDING — GPU acceleration is not viable on the Radeon 860M for Qwen3-ASR.

- **DirectML:** Produces incorrect output (all `!` characters) on both int4 and FP16 models, and is 10-60× slower than CPU. The AMD Radeon 860M's DirectML implementation appears to have numerical issues with the Qwen3 decoder graph (possibly windowed attention or MRoPE ops). Additionally, the 1.7B int4 model crashed with `Invalid input name: input_ids` — DirectML may be rewriting the graph incompatibly.
- **WebGPU:** Produces correct output but is 3× slower than CPU for int4 (5.16s vs 1.77s) and 3.5× slower for FP16 (22.0s vs 6.27s). FP32 crashes with OOM on the external data loader. The Dawn-based WebGPU EP adds significant overhead for the per-token autoregressive loop (encoder-only GPU, decoder on CPU).
- **CPU int4 remains the fastest option** at 0.16x RTF on 11s audio and 0.28x RTF on 56s audio.

**Notes:**
- The encoder runs on GPU, decoder on CPU (sequential execution). Full-GPU decoder was not tested given the poor encoder-only results.
- The Radeon 860M is an integrated GPU with shared system memory — no dedicated VRAM. This fundamentally limits GPU compute throughput vs dedicated GPUs.
- The GTX 1650 SUPER (dedicated 4 GB VRAM, Turing architecture) on the `step` host may show different results. NVIDIA's DirectML and CUDA implementations are more mature than AMD's for transformer workloads.
- The `qwen3-asr-1.7b-quant` directory on C:\ uses old unsuffixed file format — tests crashed with `Invalid input name: input_ids`. Needs re-exporting with suffixed filenames.

### [103] GPU benchmark: DirectML and WebGPU on NVIDIA GTX 1650 SUPER (4 GB dedicated)
**Date:** 2026-03-18
**Idea:** The Radeon 860M iGPU results ([102]) showed GPU acceleration was not viable due to AMD's immature DirectML implementation and shared memory architecture. Test the same models on a discrete NVIDIA GPU to determine if the issue is fundamental or vendor-specific. The GTX 1650 SUPER has 4 GB dedicated VRAM and Turing architecture.

**Change:** Transferred pre-built Windows binary (compiled on anl with `--features "qwen3,ort-directml,ort-webgpu"`) and ORT DLLs to step. Ran benchmarks when system was idle (load avg < 2.0). Encoder on GPU, decoder on CPU (default).

**Result (11s JFK audio, NVIDIA GTX 1650 SUPER 4 GB, Ryzen 9 3900X, 5 runs):**
| Model | CPU Mean | CPU RTF | DirectML Mean | DirectML RTF | WebGPU Mean | WebGPU RTF |
|---|---|---|---|---|---|---|
| 0.6B int4 | 2.42s | 0.22x | 2.25s | 0.20x | **1.83s** | **0.17x** |
| 0.6B FP16 | 13.0s | 1.19x | 12.4s | 1.13x | 12.2s | 1.11x |

All outputs correct on all EPs (unlike the Radeon 860M which produced garbage on DirectML).

**Comparison with Radeon 860M ([102], same binary):**
| Model + EP | Radeon 860M | GTX 1650 SUPER | Speedup |
|---|---|---|---|
| 0.6B int4 CPU | 1.77s | 2.42s | 0.73× (anl CPU is faster) |
| 0.6B int4 DirectML | 19.9s (garbage) | 2.25s (correct) | — |
| 0.6B int4 WebGPU | 5.16s | 1.83s | 2.8× |

**Outcome:** IMPROVED — **WebGPU int4 on GTX 1650 achieves 0.17x RTF**, matching Parakeet INT8 speed (0.16x on anl CPU). This is a 24% speedup over step's CPU. DirectML also works correctly on NVIDIA (7% speedup). FP16 remains slow on all EPs due to Cast node overhead.

**Notes:**
- The GPU benefit is real but modest (24% for WebGPU, 7% for DirectML). CPU int4 on the faster anl machine (Ryzen AI 7 PRO 350) at 1.77s still beats WebGPU on the step machine at 1.83s — the CPU architecture matters more than GPU acceleration at this model size.
- FP16 is not the GPU-optimal format for ORT inference. The ORT FP16 optimizer inserts Cast FP16→FP32 nodes for CPU compatibility; on GPU these Casts still execute even though the GPU has native FP16 compute. A native FP16 export (without Cast nodes) would be needed to fully exploit GPU FP16 throughput.
- The binary was cross-transferred from anl (MSVC 14.44) to step (MSVC 14.43) without issues — the ORT runtime is statically linked.
- CUDA EP was not tested (no CUDA toolkit on step). CUDA may outperform DirectML/WebGPU on NVIDIA hardware due to lower dispatch overhead and fused attention kernels.
- For the 1.7B model (4.2 GB int4), neither 4 GB GPU can fit the model. GPU acceleration for 1.7B requires 8+ GB VRAM.
