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

Prompt loading validates the selected ASR contract, and each render checks the actual roles and requested context before encoding.

Optional inlining retains original external storage because other graphs may still reference it. Graph readers preserve the lexical ONNX address when resolving relative external data.

## Precision and artifacts

| Algorithm | Bits | Exclusions | Supported behavior in ORT 1.31.0 |
| --- | --- | --- | --- |
| RTN | 4 | yes | Quantize, check every MatMulNBits operator, then label output |
| GPTQ | 4 | no | Quantize with calibration, check every operator, then label output |
| RTN or GPTQ | 8 | any | Reject before output creation |
| GPTQ | 4 | requested | Reject before output creation |

Quantization checks that every external tensor occurrence is covered by the pinned
ONNX loader and writer. Unsupported placements, including sparse values/indices,
training graphs and function-default attributes, are rejected before mutation.
Path-based quantizers stage the input closure so ORT inference files cannot replace
source weights. Scratch cleanup belongs to the producer rather than filename patterns.

An all-excluded/no-quantized-node model is rejected instead of receiving an int4
label. GPTQ receives an isolated model and its external-data closure, never the
exporter's adjacent configuration. External data is copied, because ONNX rejects
hardlinks. Smaller models are materialized inline for ORT's byte-based calibration
sessions; larger models keep the external-data path. Staging can temporarily
require the source weights' size in additional disk space. Input configuration
and weights stay untouched on success or failure.

Split n-bit invocations preserve existing output variants, including arbitrary
supported model stems, together with their recorded quantization metadata. Before
quantizing, the command inventories those graphs and their data dependencies and
checks every generated, copied, rewritten or removed destination. Identical copies
are omitted; incompatible destinations fail before output mutation, even if the
source dependency closure reintroduces a previously excluded graph. Use a clean
output directory when source and retained models require different bytes at one path.
Graph inventory discovers decoded models by content, while explicitly selected
models remain strict. Opaque unselected files are preserved, including tensor
storage with an `.onnx` suffix. The same mutation plan protects retained graphs
across precision variants and controls the actual sharing of selected decoders.
Quantization metadata is merged and written only after supporting assets succeed.

Embedding metadata declares `embed_tokens_shape` and `embed_tokens_dtype`.
Readers validate dimensions, dtype, byte count and finite values. Legacy exports
may derive shape from `decoder.vocab_size` and `decoder.hidden_size`; absent dtype
means the historical FP32 format. An explicit unknown dtype or conflicting shape
is an error. FP16 input is preserved when repackaged; FP32 conversion must remain
finite. The raw binary is separate from ONNX session creation and needs its own
validation.

Converters preserve available configuration and preprocessing metadata from the
shared model-metadata inventory; graph-only conversions remain supported.
Converters, quantizers and packages transport each ONNX graph with its external-data
closure. They reject dependencies that would be replaced by a generated graph or
rewritten metadata before writing output. Decoder canonical data may be reused by
its owning decoders; an encoder or another variant cannot claim that destination.
Copied assets and rewritten configuration replace directory entries so an existing
link cannot change retained bytes.

Legacy Torch exports run in private staging because their external filenames can
come from tensor names shared by different graphs. Before publication, validated
external ranges are streamed into `<graph>.onnx.data`, then the graph and weights
are published together with rollback. The destination graph is checked before
publication. Staging temporarily needs additional space for consolidated weights.

Packages require complete graph sets plus their actual external-data references,
configuration, tokenizer and embeddings. Inline, per-decoder and shared weights
are alternatives. The release and archives are staged and validated before
replacing a prior usable release. Every publication destination must be separate from the source. Existing emitted
archives replaced by `package.py` must be regular, recognizable model archives with the expected root and
required artifacts; inspection never extracts members or follows archive links.
An archive for an inactive variant is preserved and causes a preflight error, so
a successful release cannot leave an ambiguous stale variant beside it. Choose a
clean output name or move that file aside explicitly before retrying. The shared tokenizer-asset contract
validates every transported embedded/JSON/root Jinja template representation across
conversion and archiving. Export keeps the selected ASR template in a self-contained
family and removes stale legacy representations. Converters prepare an independent
family before graph work. Families whose model configuration affects their reader
are serialized with the resolved class, backend, settings and named templates;
the original processor template is carried separately with its own legacy JSON,
root Jinja, then processor-config precedence. Embedded-only input gains a standalone
processor companion. The same preparation owns package and direct-archive metadata; final release
validation requires a usable standalone processor template. In-place completion
only adds the missing companion. Deferred regex changes are disabled after the actual backend is materialized.
Conflict checks compare those prepared bytes with incoming and retained graph
storage, allowing identical bytes to be reused. Late publication preserves the
initial distinction between newly written graphs and independent retained graphs.
Family replacements use destination-local staging and atomic publication,
allowing stale root templates to be repaired. Unsupported additional template directories are refused before writes. A locally renderable ASR prompt is required before packaging. Tokenizer assets are validated together in staging,
and publication exceptions trigger rollback of their previous set. A failed rollback
reports the preserved backup path. Operating-system cleanup errors after a
completed publication report the leftover path and completed state; a primary
operation error keeps its identity when staging cleanup raises an `OSError`.
This does not make an entire
converter invocation transactional. An archive or validation failure is a failed
command. Standalone archives use the same release validator as packages: complete
graph triples, finite metadata-consistent embeddings, preprocessing metadata, ASR
tokenizer, strict ONNX checking and CPU runtime loading. They include the actual
ONNX external-data closure and replace a prior tar only after a complete staged write. Tokenizer validation uses built-in
classes; local asset metadata cannot authorize Python modules. The shared tokenizer
loader validates selected ASR roles, delimiters, context and language forcing before
returning, so calibration and evaluation inherit the same prompt contract. Family
validation additionally checks every shipped template representation.

Source-preservation checks track the actual paths used to read metadata, embeddings
and graph storage, including intermediate symlinks. They distinguish replacing a
directory entry from modifying the inode behind another hardlink. Protected source
access paths containing Windows junctions are refused before publication; ordinary
safe symlinks remain supported. A separate Windows CI job exercises this filesystem
guard with real junctions using only the Python standard library. It does not
establish support for the full exporter or numerical pipeline on Windows.

## Calibration cache provenance

Static calibration and AWQ use numeric NPZ caches with one versioned provenance
record and no object deserialization. Static identity includes the actual ONNX
and external-data bytes, embeddings, tokenizer files and collection parameters.
AWQ hashes the loaded checkpoint tensors in bounded chunks, model configuration,
selected tokenizer/template and collection parameters before smoothing. Both
record the producing code and relevant dependency versions.

Default cache filenames include that identity. Legacy caches remain untouched;
an upgrade collects a new entry, and only a matching entry is reused. An explicit
AWQ cache path with absent or incompatible provenance fails without replacing it;
choose a clean path to collect a current cache. This policy does not change the
explicit GPTQ calibration-data input interface.

Inlining changes only the selected graph and retains raw storage. A conventional
`.onnx.data` filename does not establish ownership; another graph can still use
those bytes. Stage owners retire their recorded inputs, while packages include
only the selected graphs' actual dependency closure.

## Shared-weight recovery

Sharing publishes the canonical `decoder_weights[.<suffix>].data` only after
preparing and validating its data and graph references. Original data remains
available while each graph is replaced. Required shape/control tensors and
operator attributes remain inline, and staged CPU ORT sessions must load before
publication. The supported q4 and q4f16 CPU paths are covered by real quantized
fixtures; missing kernels on another target need an explicit reviewed exception. Those replacements are individually
atomic; the filesystem does not provide a multi-file transaction. A retry must
validate the real graphs and bytes before declaring an already-shared result.
Any existing canonical file with incompatible bytes requires an explicit rebuild
error rather than being overwritten, even when no live graph refers to it.
Canonical symlink entries are rejected before graph publication.

Successful sharing writes the two graph files and canonical shared data file while
retaining original data dependencies, including conventional per-decoder files.
Neither a dependency reference nor a conventional filename proves deletion ownership.
An exclusive producer stage can remove the inputs it created after publishing the
canonical outputs.
Sharing stages in a sibling directory so abrupt process
exit cannot contaminate the model inventory or leave directories in the canonical
output directory. The model directory must have a writable parent on the
same filesystem for staging and atomic replacement. A terminated process can leave that sibling behind;
only remove it after establishing ownership. Historical unmarked staging inside
a model directory is preserved and inventoried like other foreign content;
invalid references produce an actionable error before publication. Keep retained data
until its ownership is known. Packaging includes only the verified file closure.
Sharing rejects graph replacement when an independent graph uses that graph file as
tensor storage. Transport preflight tracks dependency roles even when a file is also
a selected graph, and reserves optional test-audio destinations before writing.
Callers must serialize writes to a model directory.
