# Reproducible validation

The supported numerical environment is Python 3.12.13 with uv 0.11.20,
torch 2.14.1, transformers 4.57.6, qwen-asr 0.0.6, ONNX 1.23.2 and
ONNX Runtime 1.31.0. `uv.lock` records the complete resolution; the runtime-only
`requirements.txt` is generated from it. Numerical checks target Apple Silicon
with the CPU execution provider. The universal lock also resolves Linux CUDA
dependencies; it does not establish numerical validation on Linux or CUDA.
Dataset audio evaluation additionally needs the platform's FFmpeg libraries
for TorchCodec.

```sh
uv sync --locked --python 3.12.13
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy .
uv run --locked pytest -q --strict-markers
```

Default tests execute small official Qwen architectures with seeded weights,
small ONNX graphs and an attributed official chat-template fixture without
downloading checkpoints. Tests requiring complete exports are marked
`integration` and excluded by default. Unexpected warnings fail the suite;
the pinned legacy exporter deprecations and the dependency's invalid escape
SyntaxWarning are filtered. The legacy ONNX exporter preserves the tested
dynamic axes and requested opset. Each legacy export traces in a private directory,
then streams validated external tensor ranges into `<graph>.onnx.data`. Graph
fixups finish before the graph and its data are published with rollback. Retained
graph dependencies cannot be replaced by another export. This boundary does not
provide atomic visibility across files or support concurrent writers.

Run the explicit checkpoint suite separately:

```sh
uv run --locked pytest -m integration tests/full/test_checkpoint_encoder.py -s
```

It uses `Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0`
and `Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5`.
It compares official native, wrapper and dynamic ONNX encoder outputs at very
short inputs and convolution/attention boundaries, recording versions and
per-length errors. Long-input native references preserve the official attention
windows. Missing fixtures or skipped cases do not count as successful validation.
Feature parity is not evidence of improved transcription accuracy.

`build_prompt_ids` requires a tokenizer loaded with `load_prompt_tokenizer` so
role IDs, context and language forcing follow the official template. Exports
normalize the selected ASR template to Jinja for offline use and record embedding
shape and stored dtype. The tokenizer-copy stage shared by raw export and AWQ
refuses an output containing legacy `chat_template.json` before its own tokenizer
load or save; use a clean output directory. This check does not roll back earlier
model loading or artifact writes in either command.
Tokenizer-family checks validate
the embedded, JSON and root Jinja representations transported by the tools.
Converters reject invalid or internally conflicting existing template representations before conversion and
validate their resulting prompt metadata before reporting success.
The validator loads checkpoint and artifact tokenizers independently, requires
matching prompt IDs and decodes each output with its own reader before comparing
text. It checks complete sessions, finite outputs, shapes, numerical
tolerances against the independent native encoder and exact decoded transcripts:

```sh
uv run --locked python validate.py --model /path/to/pinned/checkpoint \
  --onnx-dir /path/to/complete/export --audio /path/to/audio.wav
```

Regenerate dependency locks intentionally and repeat affected numerical checks:

```sh
uv lock
uv export --locked --no-dev --format requirements-txt --output-file requirements.txt
```

The optimizer serializes default-domain RMS normalization fusions as local ONNX
functions made of standard operators at the requested opset. Their function
boundary preserves FP32 statistics through activation conversion to FP16 and
casts normalized values back before scaling, matching native Qwen semantics.
Existing Microsoft-domain skip normalization fusions retain their implementation.
Regression checks cover opsets 17/18, repeat optimization, inline/external data,
FP16 converters and quantization followed by sharing with CPU execution.

Prompt loading validates the selected ASR contract, and each render checks the actual roles and requested context before encoding. Calibration refuses existing caches whose model and prompt provenance cannot be verified; it preserves their bytes and requires fresh collection at a clean cache path.

Optional inlining retains original external storage because other graphs may still reference it. Graph readers preserve the lexical ONNX address when resolving relative external data.
