# ONNX ASR Model Packaging Conventions

Research conducted 2026-03-11 covering sherpa-onnx (k2-fsa/csukuangfj), onnx-asr (istupakov),
HuggingFace Optimum, and the ONNX spec for external data.

---

## 1. External Data Files vs Embedded Weights

### ONNX Spec

Protobuf has a hard 2 GB limit. Models exceeding this must use external data files.
The `.onnx` file stores a reference per tensor with three fields:
- `location` — relative path to the data file
- `offset` — byte offset within that file
- `length` — byte length

All tensors can be consolidated into a single external file via
`onnx.save_model(..., all_tensors_to_one_file=True, location="model.onnx.data")`.

**Never rename the external data file** — the location string is baked into the proto.

Sources:
- https://onnx.ai/onnx/repo-docs/ExternalData.html
- https://onnxruntime.ai/docs/tutorials/web/large-models.html

### In Practice

| Project | Embed or External? | Naming |
|---|---|---|
| istupakov | External `.onnx.data` when model > 2 GB | `encoder-model.onnx` + `encoder-model.onnx.data` |
| sherpa-onnx (Whisper large) | External `.weights` file | `large-v3-encoder.onnx` + `large-v3-encoder.weights` |
| sherpa-onnx (small models) | Embedded (all < 2 GB) | Single `.onnx` per component |
| HuggingFace Optimum | External `.onnx.data` when > 2 GB | `encoder_model.onnx` + `encoder_model.onnx.data` |

**Takeaway:** Models under ~1.5 GB always embed weights. Above 2 GB, external data is mandatory.
The two common data-file extensions are `.onnx.data` (HF/istupakov convention) and `.weights`
(sherpa-onnx convention). The `.onnx.data` convention is more widespread.

---

## 2. File Naming Conventions

### istupakov/onnx-asr

Flat directory, all files at repo root. File names use the pattern `{component}-model.{quant}.onnx`.

**Encoder/decoder models (NeMo Conformer/Parakeet/Canary):**
```
encoder-model.onnx              # FP32 encoder
encoder-model.int8.onnx         # INT8 encoder
encoder-model.onnx.data         # external weights (if > 2 GB)
decoder-model.onnx              # FP32 decoder (AED models)
decoder-model.int8.onnx         # INT8 decoder
decoder_joint-model.onnx        # FP32 decoder+joint (RNN-T/TDT models)
decoder_joint-model.int8.onnx   # INT8 decoder+joint
vocab.txt                       # token vocabulary
config.json                     # model metadata
```

Source code confirming these names (`src/onnx_asr/models/nemo.py`):
```python
# CTC (single model):
{"model": f"model{suffix}.onnx"}

# RNN-T / TDT (encoder + decoder_joint):
{"encoder": f"encoder-model{suffix}.onnx",
 "decoder_joint": f"decoder_joint-model{suffix}.onnx"}

# AED (encoder + decoder):
{"encoder": f"encoder-model{suffix}.onnx",
 "decoder": f"decoder-model{suffix}.onnx"}
```

Where `suffix` is `""` for FP32 or `".int8"` (etc) for quantized.
The `?` in the filename pattern (e.g. `model?.int8.onnx`) is used as a glob wildcard
in the resolver — it matches `model.int8.onnx`.

**Whisper models:**
```
whisper-base_beamsearch.onnx
whisper-base_beamsearch.int8.onnx
vocab.json, merges.txt, tokenizer_config.json, ...
```

**GigaAM models (multiple architectures in one repo):**
```
v3_ctc.onnx / v3_ctc.int8.onnx
v3_e2e_ctc.onnx / v3_e2e_ctc.int8.onnx
v3_rnnt_encoder.onnx / v3_rnnt_encoder.int8.onnx
v3_rnnt_decoder.onnx / v3_rnnt_decoder.int8.onnx
v3_rnnt_joint.onnx / v3_rnnt_joint.int8.onnx
v3_ctc.yaml / v3_rnnt.yaml   # per-architecture config
v3_vocab.txt / v3_e2e_ctc_vocab.txt / v3_e2e_rnnt_vocab.txt
config.json
```

### sherpa-onnx (csukuangfj)

File names encode the model name, epoch/checkpoint info, and quantization.

**Zipformer (encoder/decoder/joiner, RNN-T):**
```
encoder-epoch-99-avg-1.onnx
encoder-epoch-99-avg-1.int8.onnx
decoder-epoch-99-avg-1.onnx
decoder-epoch-99-avg-1.int8.onnx
joiner-epoch-99-avg-1.onnx
joiner-epoch-99-avg-1.int8.onnx
tokens.txt
bpe.model
test_wavs/
```

**Whisper:**
```
base-encoder.onnx / base-encoder.int8.onnx
base-decoder.onnx / base-decoder.int8.onnx
base-tokens.txt
test_wavs/
```

**CTC-only (single model file):**
```
model.int8.onnx
tokens.txt
test_wavs/
```

### HuggingFace Optimum

Uses underscores, not hyphens. Splits encoder-decoder models by default.

```
encoder_model.onnx
encoder_model.onnx.data          # if > 2 GB
decoder_model.onnx
decoder_model.onnx.data
decoder_with_past_model.onnx     # KV-cache variant (merged by default)
decoder_with_past_model.onnx.data
config.json                      # full HF model config
tokenizer.json / vocab.json / ...
```

The `--monolith` flag forces a single ONNX file. The `--no-post-process` flag
prevents merging decoder and decoder_with_past.

---

## 3. Quantization Handling

All three projects use the same convention: quantization type is inserted as a
**secondary extension** before `.onnx`.

| FP32 | INT8 | Pattern |
|---|---|---|
| `encoder-model.onnx` | `encoder-model.int8.onnx` | `{name}.{quant}.onnx` |
| `model.onnx` | `model.int8.onnx` | same |
| `base-encoder.onnx` | `base-encoder.int8.onnx` | same |

istupakov's resolver constructs the quantized filename by inserting the quantization
string before `.onnx`:
```python
suffix = "?" + quantization if quantization else ""
# e.g. "encoder-model.int8.onnx" when quantization="int8"
```

The resolver also auto-downloads external data: for every `.onnx` file in the allow
list, it appends a `.onnx?data` pattern (HuggingFace glob syntax matching `.onnx.data`).

No project uses separate directories for FP32 vs INT8 — both live side-by-side in the
same flat directory. The consumer picks which to load.

---

## 4. Metadata / Config Files

### istupakov `config.json`

Minimal JSON with model-type routing and preprocessing parameters:

```json
// Parakeet TDT
{"model_type": "nemo-conformer-tdt", "features_size": 128, "subsampling_factor": 8}

// Canary AED
{"model_type": "nemo-conformer-aed", "features_size": 128}

// GigaAM v3
{"model_type": "gigaam", "version": "v3", "features_size": 64,
 "subsampling_factor": 4, "max_tokens_per_step": 3}
```

The `model_type` field is used by the resolver to select the correct Python class
when loading from an arbitrary HuggingFace repo (i.e. not a known model name).

### sherpa-onnx

No `config.json` — model type is inferred from the repo name and file presence.
Metadata files:
- `tokens.txt` — integer-to-token mapping (one per line: `token id`)
- `bpe.model` — SentencePiece BPE model (when applicable)
- `test_wavs/` — sample audio for smoke testing
- `README.md` — usage instructions
- `LICENSE`

### HuggingFace Optimum

Full HuggingFace `config.json` (the original model config, not a custom minimal one),
plus all tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`,
`merges.txt`, `special_tokens_map.json`, etc.).

---

## 5. HuggingFace Repo Structure

### istupakov Pattern

Repo name: `istupakov/{original-model-name}-onnx`
Examples:
- `istupakov/parakeet-tdt-0.6b-v3-onnx`
- `istupakov/canary-1b-v2-onnx`
- `istupakov/gigaam-v3-onnx`
- `istupakov/whisper-base-onnx`

All files at repo root (flat). Both FP32 and INT8 variants in the same repo.
Repo includes `config.json` for model-type routing.

Actual file listing for `parakeet-tdt-0.6b-v3-onnx`:
```
config.json                      (97 B)
encoder-model.onnx               (41.8 MB proto + 2.43 GB .data)
encoder-model.onnx.data          (2.43 GB)
encoder-model.int8.onnx          (652 MB, self-contained)
decoder_joint-model.onnx         (72.5 MB)
decoder_joint-model.int8.onnx    (18.2 MB)
nemo128.onnx                     (140 KB, mel preprocessor)
vocab.txt                        (94 KB)
README.md
```

Note: the FP32 encoder needs external data (2.43 GB total) while the INT8 encoder
fits in a single file (652 MB). The FP32 decoder_joint is small enough to embed.

Actual file listing for `canary-1b-v2-onnx`:
```
config.json                      (68 B)
encoder-model.onnx               (4.8 MB proto + 3.3 GB .data)
encoder-model.onnx.data          (3.3 GB)
encoder-model.int8.onnx          (859 MB)
decoder-model.onnx               (676 MB)
decoder-model.int8.onnx          (170 MB)
vocab.txt                        (208 KB)
README.md
```

### csukuangfj/sherpa-onnx Pattern

Repo name: `csukuangfj/sherpa-onnx-{model-type}-{lang}-{variant}-{date}`
Examples:
- `csukuangfj/sherpa-onnx-whisper-base`
- `csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`
- `csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-int8-2025-11-12`

The date suffix and architecture name are baked into the repo name. INT8-only repos
sometimes have `int8` in the repo name itself rather than publishing both variants.

Actual file listing for `sherpa-onnx-whisper-base`:
```
base-encoder.onnx                (90.7 MB)
base-encoder.int8.onnx           (27.8 MB)
base-decoder.onnx                (187.4 MB)
base-decoder.int8.onnx           (124.5 MB)
base-tokens.txt                  (798 KB)
test_wavs/
```

Actual file listing for `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`:
```
encoder-epoch-99-avg-1.onnx      (330 MB)
encoder-epoch-99-avg-1.int8.onnx (182 MB)
decoder-epoch-99-avg-1.onnx      (13.9 MB)
decoder-epoch-99-avg-1.int8.onnx (13.1 MB)
joiner-epoch-99-avg-1.onnx       (12.8 MB)
joiner-epoch-99-avg-1.int8.onnx  (3.2 MB)
tokens.txt                       (56 KB)
bpe.model                        (245 KB)
export-onnx-stateless7-streaming-zh.sh
test_wavs/
```

### HuggingFace Optimum Pattern

Repo name matches original model or uses `onnx-community/` prefix.
Contains full tokenizer files alongside ONNX models. Uses `encoder_model.onnx` /
`decoder_model.onnx` naming (underscores).

---

## 6. Summary: Conventions Relevant to Qwen3-ASR Packaging

Based on the research, the istupakov convention is the closest match for a new ONNX
ASR model repo:

1. **Repo naming**: `{author}/{model-name}-onnx` (e.g. `andrewleech/qwen3-asr-0.6b-onnx`)

2. **File naming**: `{component}-model.{quant}.onnx` for multi-component models,
   or `model.{quant}.onnx` for single-file models. Quantization as secondary extension.

3. **External data**: Use `.onnx.data` suffix when model exceeds ~1.5 GB. INT8 models
   often fit in a single file even when FP32 requires external data.

4. **Config file**: Minimal `config.json` with `model_type`, `features_size`, and any
   architecture-specific parameters needed by the consumer.

5. **Vocabulary**: `vocab.txt` (one token per line) or tokenizer JSON files.

6. **Both precisions in one repo**: FP32 and INT8 side-by-side, same directory.

7. **No subdirectories**: Flat structure, everything at repo root.

The sherpa-onnx convention (epoch numbers in filenames, `tokens.txt`, `test_wavs/`)
is specific to k2/icefall-exported models and less applicable to transformer
encoder-decoder architectures like Qwen3-ASR.
