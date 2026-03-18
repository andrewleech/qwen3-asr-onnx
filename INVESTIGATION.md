
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
