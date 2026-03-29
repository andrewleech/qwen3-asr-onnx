---
name: model-experiment
description: Run structured model optimization experiments for Qwen3-ASR ONNX. Use when testing quantization variants, encoder formats, weight sharing, or other model changes that need WER/RTF measurement. Triggered by requests to "run an experiment", "test a variant", "try quantizing", "measure WER", or any model optimization trial in the qwen3-asr-onnx repo.
---

# Model Experiment Workflow

Structured trial-based workflow for Qwen3-ASR ONNX model optimization. Each experiment modifies one variable, measures WER/RTF, and records results in INVESTIGATION.md regardless of outcome.

## Workflow

### 1. Read prior work

Read the top of `INVESTIGATION.md` to find the latest experiment number. The next experiment is `[N+1]`. Scan recent experiments for context on what's been tried and current baselines.

Current baselines (update from INVESTIGATION.md if stale):
- 0.6B int4 RTN al4: ~5.16% WER, ~0.17x RTF
- 1.7B int4 RTN al4: ~4.20% WER, ~0.29x RTF

### 2. Set up trial directory

Create a trial directory with symlinks to avoid duplicating multi-GB files. Only the file under test should be a real copy or new file.

```bash
# Example: testing a new encoder variant
cp -rs $(pwd)/output/qwen3-asr-0.6b output/trial-<name>
rm output/trial-<name>/encoder.int4.onnx  # remove symlink for the file under test
cp <new-encoder> output/trial-<name>/encoder.int4.onnx
```

The trial dir must contain ALL files needed by `evaluate_wer.py` including `tokenizer_config.json`, `added_tokens.json`, `vocab.json` — missing any causes `AutoTokenizer.from_pretrained()` to fall through to `AutoConfig` which fails on `model_type: qwen3_asr`.

### 3. Run WER evaluation

```bash
cd ~/qwen3-asr-onnx
uv run python evaluate_wer.py \
  --models \
    "baseline:output/qwen3-asr-0.6b:int4" \
    "trial-name:output/trial-<name>:int4" \
  --datasets librispeech-other \
  --n-samples 200 \
  --output /tmp/wer_<experiment>.json
```

The `:int4` suffix resolves `encoder.int4.onnx`, `decoder_init.int4.onnx`, etc. Omit for FP32. Run in background for long evals (~30-60 min for 200 samples with 2 models).

Check process is alive: `ps aux | grep evaluate_wer`
Output is buffered — check file size growth: `wc -c <output-file>`
Final results appear in the last lines with a summary table.

### 4. Record results in INVESTIGATION.md

Prepend the new experiment at the TOP of the file (newest first). Use this format:

```markdown
### [N] Title — brief description
**Date:** YYYY-MM-DD
**Idea:** What hypothesis is being tested and why. Reference prior experiments.
**Change:** What was modified (files, scripts, parameters).
**Result (200-sample LibriSpeech test-other):**
| Trial | Description | WER | RTF | Size |
|---|---|---|---|---|
| Baseline | ... | X.XX% | X.XXx | X MB |
| Trial | ... | X.XX% | X.XXx | X MB |

**Outcome:** SUCCESS/FAILURE/DEGRADED/FINDING — one-line summary.
**Notes:**
- Why it worked or failed (root cause analysis)
- Implications for future experiments
- References to related experiments [N]
```

Outcome categories:
- **SUCCESS**: meets or exceeds target, ready for adoption
- **FAILURE**: broken output, crashes, or catastrophic degradation
- **DEGRADED**: works but measurably worse than baseline (document the tradeoff)
- **FINDING**: neutral result that informs future work (e.g. measurement, characterization)

### 5. Commit or revert

**On SUCCESS:**
1. Commit model file changes (new scripts, quantized files, config updates)
2. Commit INVESTIGATION.md entry
3. Update README.md if the finding changes recommended variants or package contents
4. Update memory with key findings

**On FAILURE/DEGRADED:**
1. Revert any model file changes: `rm -rf output/trial-<name>`
2. Commit ONLY the INVESTIGATION.md entry — negative results are valuable
3. Update memory if the finding is surprising or corrects assumptions

### 6. Clean up

```bash
rm -rf output/trial-<name>
rm /tmp/wer_<experiment>.json
```

## Disk space constraints

WSL VHD grows but never shrinks. Check C:\ before large operations: `df -h /mnt/c`. Use symlinks/hardlinks for trial dirs. Delete trial dirs promptly after recording results.

## Tool reference

See `references/tools.md` for parameters and usage of all Python tools in the repo.
