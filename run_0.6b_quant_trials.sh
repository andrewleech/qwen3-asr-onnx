#!/usr/bin/env bash
# run_0.6b_quant_trials.sh — Trial GPTQ and block_size=32 on 0.6B int4
#
# Experiments [105] and [106]:
#   [105] GPTQ decoder_init + RTN al4 decoder_step (block_size=64)
#   [106] RTN al4 both decoders (block_size=32)
#
# Baseline: RTN al4 block_size=64 (current) = 5.16% WER
# Target: recover to ~5.08%

set -euo pipefail

MODEL_DIR="output/qwen3-asr-0.6b"
RESULTS_DIR="output/trial-results"
mkdir -p "$RESULTS_DIR" calibration_cache

echo "================================================================"
echo "=== [105] GPTQ decoder_init + RTN al4 decoder_step (bs=64)  ==="
echo "================================================================"
echo ""

# Step 1: Collect GPTQ calibration data for 0.6B decoder_init
echo "--- Collecting GPTQ calibration data ---"
uv run python collect_gptq_calib.py \
    --model "$MODEL_DIR" \
    --n-samples 32 --decoder-steps 8 --threads 16 \
    --target decoder_init \
    --output calibration_cache/0.6b_gptq_init.npz

echo ""

# Step 2: GPTQ decoder_init
echo "--- Quantizing decoder_init with GPTQ ---"
TRIAL_105="output/trial-0.6b-gptq-bs64"
mkdir -p "$TRIAL_105"

uv run python quantize_nbits.py \
    --input "$MODEL_DIR" \
    --output "$TRIAL_105" \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --algo gptq --calib-data calibration_cache/0.6b_gptq_init.npz \
    --decoders decoder_init

# Step 3: RTN decoder_step (same as current)
echo "--- Quantizing decoder_step with RTN ---"
uv run python quantize_nbits.py \
    --input "$MODEL_DIR" \
    --output "$TRIAL_105" \
    --bits 4 --block-size 64 --accuracy-level 4 \
    --algo rtn --decoders decoder_step

# Step 4: Clean GPTQ temp files from source dir
echo "--- Cleaning GPTQ temp files ---"
rm -f "$MODEL_DIR"/*-*-*-*-*.data "$MODEL_DIR"/*_augment.onnx 2>/dev/null || true

echo ""
echo "================================================================"
echo "=== [106] RTN al4 both decoders (block_size=32)             ==="
echo "================================================================"
echo ""

TRIAL_106="output/trial-0.6b-rtn-bs32"
mkdir -p "$TRIAL_106"

echo "--- Quantizing both decoders with RTN block_size=32 ---"
uv run python quantize_nbits.py \
    --input "$MODEL_DIR" \
    --output "$TRIAL_106" \
    --bits 4 --block-size 32 --accuracy-level 4 \
    --algo rtn

echo ""
echo "================================================================"
echo "=== WER Evaluation                                           ==="
echo "================================================================"
echo ""

# Baseline: FP32 encoder trial dir (encoder.int4.onnx → encoder.onnx symlink)
# to avoid picking up the real INT4 encoder from experiment [104] in MODEL_DIR.
BASELINE_DIR="output/trial-fp32enc-fp32embed"
uv run python evaluate_wer.py \
    --models \
        "baseline RTN bs64 (FP32 enc):$BASELINE_DIR:int4" \
        "[105] GPTQ-init+RTN bs64:$TRIAL_105:int4" \
        "[106] RTN bs32:$TRIAL_106:int4" \
    --datasets librispeech-other \
    --n-samples 200 \
    --output "$RESULTS_DIR/quant_trials_105_106.json"

echo ""
echo "Results written to $RESULTS_DIR/quant_trials_105_106.json"
echo ""
cat "$RESULTS_DIR/quant_trials_105_106.json"
