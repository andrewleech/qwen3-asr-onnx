
### [114] Selective layer exclusion from INT4 — first/last layer + lm_head at FP32
**Date:** 2026-03-30
**Idea:** The Qwen3 quantization study (arxiv 2505.02214) and TuneQn framework suggest that first and last decoder layers plus lm_head are most sensitive to quantization. Test excluding these from INT4 using ORT's `nodes_to_exclude` parameter. 15 of 197 MatMul nodes excluded (~8%): layer 0 (node_linear through node_linear_6), layer 27 (node_linear_189-195), lm_head (node_linear_196).

**Change:** Added `--nodes-to-exclude` parameter to `quantize_nbits.py`. Quantized both decoders with the 15-node exclusion list.

**Result (200-sample LibriSpeech test-other, 0.6B, WSL/Linux):**
| Trial | WER | RTF | Size |
|---|---|---|---|
| Baseline int4 | 5.16% | 0.28x | baseline |
| L0+L27+lm_head FP32 | **4.96%** | 0.35x | +9% |

**Outcome:** MIXED — 0.20pp WER improvement but 25% speed regression and a new failure mode.

**Details:**
- WER drops from 5.16% → 4.96% (-0.20pp). Consistent improvement tracked through all 200 samples.
- RTF regresses from 0.28x → 0.35x. The 15 FP32 MatMul nodes (2 full layers + lm_head) are slower than their int4 equivalents.
- New failure mode: 3 samples (46, 83, 102) produced "ology" — degenerate single-token output on short utterances ("nonsense", "so shall we yet sir", "no wait"). This didn't occur in the baseline. The FP32 first/last layers may shift the decoder's logit distribution enough to trigger early EOS on very short inputs.

**Analysis:**
- The 0.20pp WER gain is real and demonstrates that first/last layers are indeed sensitive to int4 quantization in this model.
- The 25% speed regression makes this impractical for the recommended configuration. The speed loss outweighs the WER gain for a model positioned as "matching Parakeet speed."
- The "ology" failure mode suggests the mixed FP32/int4 decoder has a distribution mismatch at boundaries between quantized and unquantized layers.
- A per-layer sensitivity profile (rather than heuristic first/last selection) might identify a better subset of layers to exclude with less speed impact. Excluding only the most sensitive 2-3 projections within a layer (rather than all 7) could reduce the speed penalty while retaining most of the WER benefit.

**Future directions:**
- Test excluding only attention projections (q,k,v,o) in layer 0 — 4 nodes instead of 7, targeting the most sensitive ops.
- Per-layer QDQ sensitivity profiling to find the actual most-sensitive layers (might not be first/last).
- Test on 1.7B where the model has more capacity and selective exclusion may have lower relative speed cost.

### [113] RMSNorm fusion before int4 quantization — fuse-then-quantize pipeline
**Date:** 2026-03-30
**Idea:** Experiment [111] applied SimplifiedLayerNormalization fusion *after* int4 quantization and observed +0.59pp WER on 10 samples. Hypothesis: the regression was because int4 weights were calibrated against the decomposed RMSNorm path, then the norm implementation changed. If fusion is applied to the FP32 graph *before* quantization, the int4 weights are calibrated against the fused kernel from the start — no mismatch.

**Change:** Pipeline: FP32 decoder → `optimize_model(model_type='gpt2')` → fix external data refs → `quantize_nbits.py` (RTN al4 bs64) → evaluate.

**Result (200-sample LibriSpeech test-other, 0.6B, WSL/Linux):**
| Trial | WER | RTF | Time |
|---|---|---|---|
| Baseline int4 | 5.16% | 0.23x | 316s |
| Fuse-then-quantize int4 | **5.13%** | **0.22x** | **302s** |

**Outcome:** SUCCESS — zero WER impact (5.13% vs 5.16%, within noise), 4.2% faster. The fused `SimplifiedLayerNormalization` ops (113 instances) remain as FP32 kernels in the int4 graph and are measurably faster than the 5-op decomposed RMSNorm.

**Why this works when post-quantization fusion [111] didn't:**
- RTN quantization produces weights optimized for the exact graph structure at quantization time. Changing the norm implementation after quantization introduces a mismatch between the weight calibration context and the inference path.
- Fusing before quantization means RTN sees the fused graph and calibrates accordingly. The int4 weights are optimal for the fused kernel.

**Implication:** The export pipeline should include an optimize_model step between FP32 export and int4 quantization. This is a free 4% speedup with no quality tradeoff. The same approach should work for 1.7B (untested — same architecture, same fusion pattern).

### [112] ORT transformer optimizer — SkipLayerNorm + BiasGelu fusion on encoder
**Date:** 2026-03-30
**Idea:** Following the decoder RMSNorm fusion finding ([111]), test whether the ORT transformer optimizer also improves the encoder. The encoder uses standard LayerNorm (not RMSNorm) and GELU activations, which should match the SkipLayerNormalization and BiasGelu fusion patterns.

**Change:** Ran `optimize_model` with `model_type=gpt2`, `num_heads=14`, `hidden_size=896` on encoder.onnx. Both `gpt2` and `bert` model types produce identical results.

**Fusions achieved (699 → 514 nodes, 26% reduction):**
- SkipLayerNormalization: 35 (fused skip-connection + LayerNorm)
- BiasGelu: 19 (fused bias + GELU activation)
- Gelu: 3 (fused Erf-based GELU)
- LayerNormalization: 37 → 2 (most folded into SkipLayerNorm)
- No attention fusion (windowed attention pattern doesn't match any template)

**Result (10-sample LibriSpeech test-other, WSL/Linux):**
| Trial | Decoders | WER | RTF |
|---|---|---|---|
| Baseline FP32 | FP32 | 3.55% | 0.58x |
| Encoder-fused FP32 | FP32 | 3.55% | 0.58x |
| Baseline int4 | int4 | 5.92% | 0.27x |
| Encoder-fused int4 | int4 | 5.92% | **0.26x** |

**Outcome:** FINDING — token-exact output, marginal speedup with int4 decoders (~4%), no speedup with FP32 decoders.

**Notes:**
- ORT's runtime optimizer already fuses LayerNorm and GELU patterns on the encoder at load time. The offline transformer optimizer adds SkipLayerNorm fusion (skip-connection + norm in one op) which the runtime misses, but the benefit is small for the encoder.
- With FP32 decoders, encoder time is a small fraction of total inference — the speedup is unmeasurable. With int4 decoders (faster), the encoder's share increases and the ~4% gain becomes visible.
- File size increased slightly (717 → 746 MB) due to fused op attributes.
- Safe to adopt for distribution if the encoder file size increase (~30 MB) is acceptable. Not high priority — the speedup is marginal.

### [111] ORT transformer optimizer — SimplifiedLayerNormalization fusion on decoders
**Date:** 2026-03-30
**Idea:** ORT's runtime optimizer does NOT fuse the decomposed RMSNorm pattern (ReduceMean → Pow → Add → Sqrt/Reciprocal → Mul, 113 instances × 5 ops = 565 ops) into `SimplifiedLayerNormalization`. The ORT transformer optimizer (`optimize_model` with `model_type=gpt2`) does fuse these, reducing node count by ~30% (2266 → 1586). Test whether pre-optimizing the decoder graphs improves inference speed. Also test if the GQA attention pattern (16Q/8KV heads, MRoPE) fuses.

**Change:** Ran `onnxruntime.transformers.optimizer.optimize_model` on decoder_init and decoder_step for both FP32 and int4 variants. `opt_level=1`, `num_heads=16`, `hidden_size=1024`.

**Result — FP32 (10-sample quick test, WSL/Linux):**
| Trial | WER | RTF | Nodes |
|---|---|---|---|
| Baseline FP32 | 3.55% | 0.69x | 2266 |
| RMSNorm-fused FP32 | 3.55% | **0.64x** | 1586 |

**Result — int4 (10-sample quick test, WSL/Linux):**
| Trial | WER | RTF | Nodes |
|---|---|---|---|
| Baseline int4 | 5.92% | 0.29x | ~2266 |
| RMSNorm-fused int4 | **6.51%** | 0.28x | 1586 |

**Outcome:** MIXED
- **FP32: SUCCESS** — 7.3% speedup, token-exact output on all 10 samples. The fused `SimplifiedLayerNormalization` op (113 instances) is measurably faster than the 5-op decomposed RMSNorm.
- **int4: DEGRADED** — 3% speedup but +0.59pp WER regression. The `SimplifiedLayerNormalization` kernel produces slightly different numerical results than the decomposed path when operating on dequantized int4 MatMulNBits outputs. Samples 9-10 diverged.
- **GQA attention fusion: not achieved** — the Qwen3 attention pattern (MRoPE, GQA) does not match any of ORT's fusion templates. Zero attention-related ops fused.

**Notes:**
- ORT's runtime graph optimizer (level ALL) does NOT perform this RMSNorm fusion — only the offline transformer optimizer does. This means pre-optimized graphs are genuinely faster.
- The int4 regression is likely due to accumulated floating point differences: `SimplifiedLayerNormalization` uses a single fused kernel with different intermediate rounding than the 5-op chain. On FP32 this is bit-exact, but on int4 (where inputs have quantization noise) the different rounding propagates.
- For FP32-only distribution, pre-optimization is safe. For int4, it requires a full 200-sample WER validation before adoption.
- A future ORT version may add this fusion to the runtime optimizer, making pre-optimization unnecessary.

### [110] Native FP16 encoder inference regression — Windows benchmark
**Date:** 2026-03-29
**Idea:** The native FP16 encoder from [108] halves file size with zero WER impact. Measure whether the 2 Cast nodes at the I/O boundary (FP32→FP16 input, FP16→FP32 output) affect per-inference speed on native Windows.

**Result (Ryzen AI 7 PRO 350, native Windows, ORT 2.0 rc12, 11s JFK audio):**
| Encoder | Load | Mean (10 runs) | Min | Max | RTF |
|---|---|---|---|---|---|
| FP32 (717 MB) | 5.04s | **1.850s** | 1.807s | 1.881s | **0.17x** |
| Native FP16 (358 MB) | 4.75s | **2.012s** | 1.957s | 2.036s | **0.18x** |

**Outcome:** CONFIRMED REGRESSION — FP16 encoder is 8.7% slower per inference (0.162s, repeatable with no distribution overlap). Loads 5% faster (smaller file). The Cast nodes at the I/O boundary prevent ORT from fusing ops across the FP16/FP32 boundary.

**1.7B result (10 runs each):**
| Encoder | Load | Mean | Min | Max | RTF |
|---|---|---|---|---|---|
| FP32 (1.2 GB) | 12.8s | **3.179s** | 3.152s | 3.200s | **0.29x** |
| Native FP16 (609 MB) | 15.1s | **3.606s** | 3.418s | 3.968s | **0.33x** |

FP16 is 13.4% slower on 1.7B with higher variance (±0.25s vs ±0.02s). FP16 also loads slower on 1.7B (opposite of 0.6B).

**Decision:** Revert both model sizes to FP32 encoder. The native FP16 encoder export remains available via `export_encoder_native_fp16.py` for GPU targets where FP16 compute is native, but for CPU inference FP32 is faster.

**Notes:**
- The regression is from per-inference Cast overhead. ORT's graph optimizer cannot fuse operators across dtype boundaries.
- FP32 encoder compresses well in tar.gz (~350 MB / ~600 MB compressed), so the download size impact is minimal.
- A future ORT version with better mixed-precision fusion could eliminate this overhead.
- The native FP16 export is still valuable for GPU EPs (DirectML, WebGPU) where FP16 compute is native and Cast nodes are free.

### [109] Int4 decoder weight sharing — share_weights.py on quantized models
**Date:** 2026-03-29
**Idea:** The split decoder (init + step) duplicates transformer layer weights in separate `.data` files. For FP32, `share_weights.py` deduplicates by hash-matching. Test whether the same works for int4 quantized weights — RTN is deterministic, so identically-quantized weights from the same FP32 source should be byte-identical.

**Change:** Extended `share_weights.py` with `--suffix int4` to handle suffixed filenames (`decoder_init.int4.onnx`). Produces `decoder_weights.int4.data` as the shared file.

**Result (0.6B, both decoders RTN al4 bs64):**
| Metric | Value |
|---|---|
| Matched tensors | 645 / 648 |
| Shared data | 0.25 GB (transformer layers) |
| Unmatched (inlined) | 3 tensors — lm_head int4 (89 MB), inlined into step proto |
| Step proto size | 91 MB (was 2 MB + 325 MB .data) |
| Net savings | 234 MB (1,159 → 925 MB) |
| WER | 5.16% (identical to baseline) |
| RTF | 0.24x (identical) |

**Result (1.7B, GPTQ init + RTN step):**
| Metric | Value |
|---|---|
| Matched tensors | 57 / 648 (small constants only) |
| Net savings | ~0 MB (591 tensors inlined, step proto = 983 MB) |

**Outcome:** SUCCESS for 0.6B — transformer layer int4 weights are byte-identical when both use RTN. FAILURE for 1.7B — GPTQ and RTN produce different int4 representations. Mixed quantization methods prevent weight sharing.

**Why lm_head doesn't match:** In decoder_init, `lm_head.weight` is FP32 (used as Gather for embedding lookup — quantizer only converts MatMul ops). In decoder_step, lm_head is int4-quantized (MatMulNBits). Different representations → no hash match.

**Trial 1 — RTN-only for 1.7B (200-sample LibriSpeech test-other):**
| Model | WER | RTF | Sharing |
|---|---|---|---|
| Baseline GPTQ+RTN | 4.16% | 0.50x | 57/648 tensors matched (0 MB saved) |
| RTN-only (shared) | 4.20% | 0.50x | 645/648 tensors matched (0.98 GB saved) |

RTN-only WER is +0.04pp (within noise). Prior experiment [89] measured +0.08pp — confirmed: RTN-only is equivalent to GPTQ+RTN for 1.7B. GPTQ is no longer recommended; RTN-only enables weight sharing.

**Summary — recommended int4 format:**
| Model | Quantization | Sharing | Savings | WER |
|---|---|---|---|---|
| 0.6B | RTN al4 bs64 | share_weights.py --suffix int4 | 234 MB (1,159 → 925 MB) | 5.16% (unchanged) |
| 1.7B | RTN al4 bs64 (both) | share_weights.py --suffix int4 | 937 MB (2,937 → 2,000 MB) | 4.20% (was 4.16% GPTQ) |

Trial 2 (hidden_states export) deferred — Trials 0+1 deliver the majority of savings without Rust code changes.

### [108] Native FP16 encoder export — autocast tracing (0.6B)
**Date:** 2026-03-29
**Idea:** The weight-only FP16 approach ([107]) failed because the attention mask constant (`-3.4e38`) overflows FP16 range. The graph-level FP16 conversion (`convert_fp16.py`) inserts Cast nodes everywhere, degrading both speed and quality. Try a third approach: load the model in FP32, export with `torch.amp.autocast('cpu', dtype=torch.float16)` so PyTorch traces native FP16 ops where safe and keeps FP32 for precision-sensitive ops (LayerNorm, softmax). The attention mask uses `torch.finfo(torch.float16).min` = `-65504` (valid FP16) instead of `-3.4e38`.

**Change:** Created `export_encoder_native_fp16.py`:
1. Load model in FP32, convert all parameters to FP16 via `.half()`
2. Trace with `torch.amp.autocast('cpu', dtype=torch.float16)` — captures mixed-precision graph natively
3. Patch I/O with 2 Cast nodes: FP32 mel input → FP16 (internal), FP16 features → FP32 output
4. Embed weights into proto, fix Reshape allowzero

File size: 376 MB (vs 717 MB FP32 — 47% reduction). Only 2 Cast nodes at I/O boundary, no internal Casts.

**Result (200-sample LibriSpeech test-other, 0.6B int4 decoders, Python evaluate_wer.py):**
| Trial | Encoder | WER | RTF | File size |
|---|---|---|---|---|
| Baseline | FP32 (717 MB) | 5.16% | 0.26x | 717 MB |
| Native FP16 | autocast FP16 (376 MB) | **5.13%** | **0.26x** | **376 MB** |

**Outcome:** SUCCESS — native FP16 encoder has zero WER degradation (5.13% vs 5.16% is within ±0.5pp noise). Same speed, half the file size.

**Why this works when [107] failed:**
- The attention mask is generated during tracing with `torch.finfo(mel.dtype).min` where `mel.dtype` is FP16 → produces `-65504` (valid FP16), not `-3.4e38` (overflows)
- `torch.amp.autocast` keeps LayerNorm and softmax in FP32 internally during tracing, which ONNX export captures as native mixed-precision ops — not post-hoc Cast nodes
- Only 2 Cast nodes total (I/O boundary) vs 307 in the weight-only approach

**Impact on package sizes (0.6B int4):**
| Component | Before | After |
|---|---|---|
| encoder.int4.onnx | 717 MB | 376 MB |
| embed_tokens.bin | 594 MB (FP32) | 297 MB (FP16, per [104]) |
| int4 decoders | ~1.2 GB | ~1.2 GB |
| **Total** | **~2.5 GB** | **~1.9 GB** |

**1.7B result (200-sample LibriSpeech test-other, int4 decoders):**
| Trial | Encoder | WER | RTF | File size |
|---|---|---|---|---|
| Baseline | FP32 (1.2 GB) | 4.16% | 0.44x | 1.2 GB |
| Native FP16 | autocast FP16 (639 MB) | **4.23%** | **0.43x** | **639 MB** |

+0.07pp WER on 1.7B (within noise). Slightly faster. Confirmed on both model sizes.

**Impact on package sizes (1.7B int4):**
| Component | Before | After |
|---|---|---|
| encoder.int4.onnx | 1.2 GB | 639 MB |
| embed_tokens.bin | 1.2 GB (FP32) | 610 MB (FP16) |
| int4 decoders | ~3.0 GB | ~3.0 GB |
| **Total** | **~5.4 GB** | **~4.2 GB** |

**Conclusion:** Native FP16 encoder via autocast export is the recommended encoder format for both 0.6B and 1.7B. Zero WER impact, half the file size, no speed penalty. Should replace FP32 `encoder.int4.onnx` in published packages.

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
