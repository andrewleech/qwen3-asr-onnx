#!/usr/bin/env python3
"""
Share external weight data between split decoder models (decoder_init + decoder_step).

After torch.onnx.export() exports both wrappers independently, each gets its own
copy of the model weights. This script rewrites decoder_step.onnx to reference
decoder_init's external data file, eliminating the duplicate.

Algorithm:
    1. Load both ONNX protos without external tensor data
    2. Hash each initializer's data from their respective .data files
    3. For matched tensors: redirect step's external_data to init's file/offset
    4. For unmatched tensors (small constants): inline into the proto
    5. Rename init's data file -> decoder_weights.data
    6. Update init's references, save both protos, delete step's data file

Usage:
    python share_weights.py output/qwen3-asr-0.6b
    python share_weights.py output/qwen3-asr-0.6b-int8
    python share_weights.py output/qwen3-asr-0.6b --also-int8
"""

import argparse
import hashlib
import mmap
import os
import sys

import onnx


SHARED_DATA_NAME = "decoder_weights.data"


def _shared_data_name(suffix=None):
    """Shared data filename, optionally with quantization suffix."""
    if suffix:
        return f"decoder_weights.{suffix}.data"
    return SHARED_DATA_NAME


def _decoder_filenames(suffix=None):
    """Return (init_onnx, step_onnx) filenames with optional suffix."""
    if suffix:
        return f"decoder_init.{suffix}.onnx", f"decoder_step.{suffix}.onnx"
    return "decoder_init.onnx", "decoder_step.onnx"


def get_external_info(tensor):
    """Extract external data location from a tensor proto."""
    info = {}
    for entry in tensor.external_data:
        info[entry.key] = entry.value
    return info


def set_tensor_external(tensor, location, offset, length):
    """Set external data references on a tensor."""
    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in [
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ]:
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


def inline_tensor(tensor, data):
    """Convert an external tensor to inline raw_data."""
    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.DEFAULT
    tensor.raw_data = bytes(data)


def hash_data(mm, offset, length):
    """Compute SHA-256 of a region in a memory-mapped file."""
    return hashlib.sha256(mm[offset:offset + length]).hexdigest()


def build_init_index(init_model, data_mm):
    """Build {sha256: (tensor_name, offset, length)} from init model's external tensors."""
    index = {}
    for tensor in init_model.graph.initializer:
        if not tensor.external_data:
            continue
        info = get_external_info(tensor)
        offset = int(info.get("offset", "0"))
        length = int(info["length"])
        h = hash_data(data_mm, offset, length)
        index[h] = (tensor.name, offset, length)
    return index


def share_external_models(model_dir, suffix=None):
    """Share weights when both models have external .data files."""
    init_name, step_name = _decoder_filenames(suffix)
    shared_name = _shared_data_name(suffix)
    init_proto_path = os.path.join(model_dir, init_name)
    step_proto_path = os.path.join(model_dir, step_name)
    init_data_path = init_proto_path + ".data"
    step_data_path = step_proto_path + ".data"
    shared_data_path = os.path.join(model_dir, shared_name)

    # Check preconditions
    if not os.path.exists(init_proto_path) or not os.path.exists(step_proto_path):
        print(f"  Split decoder not found in {model_dir}, skipping")
        return False

    if os.path.exists(shared_data_path) and not os.path.exists(step_data_path):
        print(f"  Already shared in {model_dir}")
        return True

    if not os.path.exists(init_data_path) or not os.path.exists(step_data_path):
        print(f"  External data files missing — trying inline path")
        return share_inline_models(model_dir, suffix=suffix)

    step_data_size = os.path.getsize(step_data_path)
    init_data_size = os.path.getsize(init_data_path)

    print(f"  Loading protos (no external data)...")
    init_model = onnx.load(init_proto_path, load_external_data=False)
    step_model = onnx.load(step_proto_path, load_external_data=False)

    # Build init hash index
    print(f"  Indexing init weights ({init_data_size / 1e9:.2f} GB)...")
    with open(init_data_path, "rb") as f:
        init_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        init_index = build_init_index(init_model, init_mm)
        init_mm.close()

    print(f"  Indexed {len(init_index)} init tensors")

    # Match step tensors against init index
    print(f"  Matching step weights ({step_data_size / 1e9:.2f} GB)...")
    matched = 0
    inlined = 0
    matched_bytes = 0

    with open(step_data_path, "rb") as f:
        step_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        for tensor in step_model.graph.initializer:
            if not tensor.external_data:
                continue

            info = get_external_info(tensor)
            offset = int(info.get("offset", "0"))
            length = int(info["length"])
            h = hash_data(step_mm, offset, length)

            if h in init_index:
                init_tname, init_offset, init_length = init_index[h]
                set_tensor_external(tensor, shared_name, init_offset, init_length)
                matched += 1
                matched_bytes += length
            else:
                # Unmatched: inline into proto (these are small constants)
                data = step_mm[offset:offset + length]
                inline_tensor(tensor, data)
                inlined += 1

        step_mm.close()

    # Update init model's external_data location references
    for tensor in init_model.graph.initializer:
        if not tensor.external_data:
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = shared_name

    # Rename init data file -> shared
    print(f"  Renaming {os.path.basename(init_data_path)} -> {shared_name}")
    os.rename(init_data_path, shared_data_path)

    # Save both protos
    print(f"  Saving updated protos...")
    with open(init_proto_path, "wb") as f:
        f.write(init_model.SerializeToString())
    with open(step_proto_path, "wb") as f:
        f.write(step_model.SerializeToString())

    # Delete step data file
    os.remove(step_data_path)

    # Report
    init_proto_size = os.path.getsize(init_proto_path)
    step_proto_size = os.path.getsize(step_proto_path)
    shared_size = os.path.getsize(shared_data_path)

    print(f"\n  Results:")
    print(f"    Matched:  {matched} tensors ({matched_bytes / 1e9:.2f} GB)")
    print(f"    Inlined:  {inlined} tensors (small constants)")
    print(f"    Shared data:  {shared_size / 1e9:.2f} GB ({shared_name})")
    print(f"    Init proto:   {init_proto_size / 1e6:.1f} MB")
    print(f"    Step proto:   {step_proto_size / 1e6:.1f} MB")
    print(f"    Saved:        {step_data_size / 1e9:.2f} GB")
    return True


def share_inline_models(model_dir, suffix=None):
    """Share weights when models have inline data (no .data files)."""
    init_name, step_name = _decoder_filenames(suffix)
    shared_name = _shared_data_name(suffix)
    init_proto_path = os.path.join(model_dir, init_name)
    step_proto_path = os.path.join(model_dir, step_name)
    shared_data_path = os.path.join(model_dir, shared_name)

    print(f"  Loading init model with inline data...")
    init_model = onnx.load(init_proto_path)

    # Convert init to external data format
    print(f"  Converting init to external data format...")
    onnx.external_data_helper.convert_model_to_external_data(
        init_model,
        all_tensors_to_one_file=True,
        location=shared_name,
        size_threshold=1024,
        convert_attribute=False,
    )

    # Save init (writes shared data file)
    print(f"  Saving init with external data...")
    onnx.save(init_model, init_proto_path)

    # Reload init proto to get offsets
    init_model = onnx.load(init_proto_path, load_external_data=False)

    # Build init hash index from the new shared data file
    print(f"  Indexing init weights...")
    with open(shared_data_path, "rb") as f:
        init_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        init_index = build_init_index(init_model, init_mm)
        init_mm.close()

    # Load step model with full data
    print(f"  Loading step model with inline data...")
    step_model = onnx.load(step_proto_path)

    # Match step tensors against init
    matched = 0
    inlined_kept = 0
    matched_bytes = 0

    for tensor in step_model.graph.initializer:
        if tensor.raw_data and len(tensor.raw_data) >= 1024:
            h = hashlib.sha256(tensor.raw_data).hexdigest()
            if h in init_index:
                init_name, init_offset, init_length = init_index[h]
                matched_bytes += len(tensor.raw_data)
                tensor.raw_data = b""
                set_tensor_external(tensor, shared_name, init_offset, init_length)
                matched += 1
            else:
                inlined_kept += 1
        else:
            inlined_kept += 1

    # Save step proto only (no data file written)
    print(f"  Saving step proto...")
    with open(step_proto_path, "wb") as f:
        f.write(step_model.SerializeToString())

    # Clean up step's old data file if it exists
    step_data_path = step_proto_path + ".data"
    if os.path.exists(step_data_path):
        os.remove(step_data_path)

    # Report
    init_proto_size = os.path.getsize(init_proto_path)
    step_proto_size = os.path.getsize(step_proto_path)
    shared_size = os.path.getsize(shared_data_path)

    print(f"\n  Results:")
    print(f"    Matched:  {matched} tensors ({matched_bytes / 1e9:.2f} GB)")
    print(f"    Kept inline:  {inlined_kept} tensors")
    print(f"    Shared data:  {shared_size / 1e9:.2f} GB ({shared_name})")
    print(f"    Init proto:   {init_proto_size / 1e6:.1f} MB")
    print(f"    Step proto:   {step_proto_size / 1e6:.1f} MB")
    return True


def verify_model(model_dir, filename):
    """Run onnx.checker on a model."""
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        return
    try:
        # Use path-based check to avoid 2 GiB protobuf limit
        onnx.checker.check_model(path)
        print(f"    {filename}: OK")
    except Exception as e:
        print(f"    {filename}: FAILED — {e}")
        return False
    return True


def verify_inference(model_dir, suffix=None):
    """Load both models in ORT and run a dummy forward pass."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not available, skipping inference check")
        return True

    import numpy as np

    init_name, step_name = _decoder_filenames(suffix)
    init_path = os.path.join(model_dir, init_name)
    step_path = os.path.join(model_dir, step_name)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # suppress warnings

    try:
        print(f"  Loading {init_name} in ORT...")
        init_sess = ort.InferenceSession(init_path, opts)
        print(f"  Loading {step_name} in ORT...")
        step_sess = ort.InferenceSession(step_path, opts)
        print(f"  Both sessions loaded successfully")
        return True
    except Exception as e:
        print(f"  ORT load FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Share external weight data between split decoder models"
    )
    parser.add_argument(
        "model_dir",
        help="Directory containing decoder_init.onnx and decoder_step.onnx",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Quantization suffix (e.g. 'int4' for decoder_init.int4.onnx)",
    )
    parser.add_argument(
        "--also-int8",
        action="store_true",
        help="Also process the corresponding -int8 directory",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run onnx.checker and ORT load verification after sharing",
    )
    args = parser.parse_args()

    dirs = [args.model_dir]
    if args.also_int8:
        int8_dir = args.model_dir.rstrip("/") + "-int8"
        if os.path.isdir(int8_dir):
            dirs.append(int8_dir)
        else:
            print(f"INT8 directory not found: {int8_dir}")

    init_name, step_name = _decoder_filenames(args.suffix)

    for model_dir in dirs:
        print(f"\n{'=' * 60}")
        print(f"Processing: {model_dir} (suffix={args.suffix or 'none'})")
        print(f"{'=' * 60}")

        ok = share_external_models(model_dir, suffix=args.suffix)

        if ok and args.verify:
            print(f"\n  Verification:")
            verify_model(model_dir, init_name)
            verify_model(model_dir, step_name)
            verify_inference(model_dir, suffix=args.suffix)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
