#!/usr/bin/env bash
# run_encoder_embed_trials.sh — Overnight experiment: INT4 encoder × FP16 embed_tokens
#
# Trials:
#   baseline  — FP32 encoder  + FP32 embed  (existing output/qwen3-asr-0.6b with int4 decoders)
#   trial-A   — INT4 encoder  + FP32 embed
#   trial-B   — FP32 encoder  + FP16 embed
#   trial-C   — INT4 encoder  + FP16 embed
#
# All trials use the same int4 decoder_init / decoder_step from the base model dir.
# Trial dirs are constructed with symlinks to shared files; only the encoder and
# embed-related config differ between dirs.
#
# Prerequisites:
#   - output/qwen3-asr-0.6b contains:
#       encoder.onnx            (FP32, 717 MB)
#       encoder.fp16.onnx       (FP16, 359 MB)  — FP16 copy for FP32-encoder trials
#       encoder.int4.onnx       (currently a FP32 copy — will be replaced by step 1)
#       embed_tokens.bin        (FP32, 594 MB)
#       decoder_init.int4.onnx, decoder_step.int4.onnx
#       config.json, tokenizer.json, vocab.json, tokenizer_config.json, special_tokens_map.json
#
# NOTE: encoder.int4.onnx is produced in-place alongside the source files because
# quantize_nbits.py needs config.json in the same directory to update it. The
# original FP32 copy is backed up as encoder.int4.onnx.bak.

set -euo pipefail

MODEL_DIR="output/qwen3-asr-0.6b"
RESULTS_DIR="output/trial-results"
mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Step 1: Create real INT4 encoder (replaces the existing FP32 copy)
# ---------------------------------------------------------------------------
echo "=== Step 1: Quantizing encoder to INT4 ==="

if [ -f "$MODEL_DIR/encoder.int4.onnx" ]; then
    echo "  Backing up existing encoder.int4.onnx → encoder.int4.onnx.bak"
    mv "$MODEL_DIR/encoder.int4.onnx" "$MODEL_DIR/encoder.int4.onnx.bak"
fi

uv run python quantize_nbits.py \
    --input "$MODEL_DIR" \
    --output "$MODEL_DIR" \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --decoders encoder

echo ""

# ---------------------------------------------------------------------------
# Step 2: Create FP16 embed_tokens
# ---------------------------------------------------------------------------
echo "=== Step 2: Creating FP16 embed_tokens ==="

uv run python convert_embed_fp16.py --model-dir "$MODEL_DIR"

echo ""

# ---------------------------------------------------------------------------
# Step 3: Build trial directories with symlinks
# ---------------------------------------------------------------------------
echo "=== Step 3: Building trial directories ==="

# Shared files: decoders, tokenizer, vocab — same across all trials.
# Each trial dir gets symlinks to these, plus the trial-specific encoder
# and (if needed) a modified config.json.

SHARED_FILES=(
    decoder_init.onnx
    decoder_init.onnx.data
    decoder_step.onnx
    decoder_step.onnx.data
    decoder_init.int4.onnx
    decoder_init.int4.onnx.data
    decoder_step.int4.onnx
    decoder_step.int4.onnx.data
    tokenizer.json
    vocab.json
)
# Include optional files if present
for f in tokenizer_config.json special_tokens_map.json generation_config.json; do
    [ -f "$MODEL_DIR/$f" ] && SHARED_FILES+=("$f")
done

# embed_tokens.bin is also shared by trials using FP32 embed
EMBED_FP32_FILES=(embed_tokens.bin)
# encoder.onnx is used by trials using FP32 encoder
ENCODER_FP32_FILE="encoder.onnx"
# encoder.int4.onnx is used by trials using INT4 encoder
ENCODER_INT4_FILE="encoder.int4.onnx"

# Helper: create symlinks from trial dir to source files
# Usage: link_files <trial_dir> <source_dir> <file1> [file2 ...]
link_files() {
    local trial_dir="$1"
    local source_dir="$2"
    shift 2
    local abs_source
    abs_source="$(realpath "$source_dir")"
    for f in "$@"; do
        if [ -f "$abs_source/$f" ]; then
            ln -sf "$abs_source/$f" "$trial_dir/$f"
        else
            echo "  WARNING: $source_dir/$f not found, skipping symlink"
        fi
    done
}

# Also handle external .data files for ONNX models (ORT external data)
# Convention: <basename>.onnx.data or <basename>.onnx_data
link_onnx_with_data() {
    local trial_dir="$1"
    local source_dir="$2"
    local onnx_name="$3"
    local abs_source
    abs_source="$(realpath "$source_dir")"
    ln -sf "$abs_source/$onnx_name" "$trial_dir/$onnx_name"
    # Link companion .data files if present
    for ext in ".data" "_data"; do
        if [ -f "$abs_source/${onnx_name}${ext}" ]; then
            ln -sf "$abs_source/${onnx_name}${ext}" "$trial_dir/${onnx_name}${ext}"
        fi
    done
}

ABS_MODEL="$(realpath "$MODEL_DIR")"

# --- Trial A: INT4 encoder + FP32 embed ---
TRIAL_A="output/trial-int4enc-fp32embed"
echo "  Creating $TRIAL_A"
mkdir -p "$TRIAL_A"
link_files "$TRIAL_A" "$MODEL_DIR" "${SHARED_FILES[@]}" "${EMBED_FP32_FILES[@]}"
link_onnx_with_data "$TRIAL_A" "$MODEL_DIR" "$ENCODER_INT4_FILE"
# config.json: copy from base (no embed_tokens_dtype override needed for fp32)
cp "$MODEL_DIR/config.json" "$TRIAL_A/config.json"

# --- Trial B: FP32 encoder + FP16 embed ---
TRIAL_B="output/trial-fp32enc-fp16embed"
echo "  Creating $TRIAL_B"
mkdir -p "$TRIAL_B"
link_files "$TRIAL_B" "$MODEL_DIR" "${SHARED_FILES[@]}"
link_onnx_with_data "$TRIAL_B" "$MODEL_DIR" "$ENCODER_FP32_FILE"
# FP16 embed: symlink fp16 bin as embed_tokens.bin name... but evaluate_wer.py
# always reads "embed_tokens.bin" and checks embed_tokens_dtype in config.json.
# So: symlink embed_tokens.fp16.bin → embed_tokens.bin in the trial dir.
ln -sf "$ABS_MODEL/embed_tokens.fp16.bin" "$TRIAL_B/embed_tokens.bin"
# config.json: copy and add embed_tokens_dtype: float16
ABS_MODEL="$ABS_MODEL" python3 - <<'PYEOF'
import json, os
src = os.environ["ABS_MODEL"]
with open(f"{src}/config.json") as f:
    cfg = json.load(f)
cfg["embed_tokens_dtype"] = "float16"
with open("output/trial-fp32enc-fp16embed/config.json", "w") as f:
    json.dump(cfg, f, indent=2)
PYEOF

# --- Trial C: INT4 encoder + FP16 embed ---
TRIAL_C="output/trial-int4enc-fp16embed"
echo "  Creating $TRIAL_C"
mkdir -p "$TRIAL_C"
link_files "$TRIAL_C" "$MODEL_DIR" "${SHARED_FILES[@]}"
link_onnx_with_data "$TRIAL_C" "$MODEL_DIR" "$ENCODER_INT4_FILE"
ln -sf "$ABS_MODEL/embed_tokens.fp16.bin" "$TRIAL_C/embed_tokens.bin"
ABS_MODEL="$ABS_MODEL" python3 - <<'PYEOF'
import json, os
src = os.environ["ABS_MODEL"]
with open(f"{src}/config.json") as f:
    cfg = json.load(f)
cfg["embed_tokens_dtype"] = "float16"
with open("output/trial-int4enc-fp16embed/config.json", "w") as f:
    json.dump(cfg, f, indent=2)
PYEOF

echo ""

# ---------------------------------------------------------------------------
# Step 4: WER evaluation
# ---------------------------------------------------------------------------
echo "=== Step 4: WER evaluation (200 samples, librispeech-other) ==="

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_JSON="$RESULTS_DIR/encoder_embed_trials_${TIMESTAMP}.json"

uv run python evaluate_wer.py \
    --models \
        "baseline (FP32 enc + FP32 embed):$MODEL_DIR:int4" \
        "trial-A (INT4 enc + FP32 embed):$TRIAL_A:int4" \
        "trial-B (FP32 enc + FP16 embed):$TRIAL_B:int4" \
        "trial-C (INT4 enc + FP16 embed):$TRIAL_C:int4" \
    --datasets librispeech-other \
    --n-samples 200 \
    --output "$OUTPUT_JSON"

echo ""
echo "Results written to $OUTPUT_JSON"
