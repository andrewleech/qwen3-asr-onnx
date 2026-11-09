#!/usr/bin/env python3
"""Share decoder data without invalidating the source models during publication.

Data and both graphs are validated in staging. The canonical data file is
published before either graph, and source data is retained until both graphs
have been replaced. Individual replacements are atomic; the group is not.
Source data files are retained, including after a retry. Only a producer that
owns the working directory can authorize their cleanup.
"""

import argparse
import hashlib
import os
from pathlib import Path

import onnx

from src.artifact_publication import ArtifactStaging
from src.artifact_sources import same_entry
from src.onnx_artifacts import (
    copy_artifact_entry as copy_artifact_entry,
)
from src.onnx_artifacts import (
    external_data_files as external_data_files,
)
from src.onnx_artifacts import (
    external_tensor_range,
    file_digest,
    inventory_graphs,
    storage_tensors,
    validate_generated_destinations,
)
from src.onnx_artifacts import (
    get_external_info as get_external_info,
)
from src.onnx_artifacts import (
    graph_file_closure as graph_file_closure,
)
from src.onnx_artifacts import (
    model_tensors as model_tensors,
)
from src.onnx_artifacts import (
    transport_graph_files as transport_graph_files,
)
from src.onnx_artifacts import (
    write_artifact_json as write_artifact_json,
)

SHARED_DATA_NAME = "decoder_weights.data"
_CHUNK_BYTES = 8 * 1024 * 1024


def _shared_data_name(suffix=None):
    return f"decoder_weights.{suffix}.data" if suffix else SHARED_DATA_NAME


def _decoder_filenames(suffix=None):
    tag = f".{suffix}" if suffix else ""
    return f"decoder_init{tag}.onnx", f"decoder_step{tag}.onnx"


def _can_share(tensor, root, initializer):
    # ORT materializes attributes and shape/control inputs before loading external weights.
    if not initializer:
        return False
    if tensor.data_type in (
        onnx.TensorProto.INT16,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
        onnx.TensorProto.UINT16,
        onnx.TensorProto.UINT32,
        onnx.TensorProto.UINT64,
        onnx.TensorProto.BOOL,
        onnx.TensorProto.STRING,
    ):
        return False
    length = (
        external_tensor_range(root, tensor)[2]
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        else len(tensor.raw_data)
    )
    return length >= 1024


def _tensor_chunks(tensor, root):
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        yield tensor.raw_data
        return
    path, offset, remaining = external_tensor_range(root, tensor)
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(remaining, _CHUNK_BYTES))
            if not chunk:
                raise ValueError(f"Unexpected end of data for {tensor.name}")
            remaining -= len(chunk)
            yield chunk


def set_tensor_external(tensor, location, offset, length):
    tensor.ClearField("raw_data")
    tensor.ClearField("external_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", str(offset)), ("length", str(length))):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


def _write_shared_data(storage, root, path):
    index = {}
    with path.open("wb") as stream:
        for tensor, shareable in storage:
            if tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.raw_data:
                continue
            if not shareable:
                if tensor.data_location == onnx.TensorProto.EXTERNAL:
                    data = b"".join(_tensor_chunks(tensor, root))
                    tensor.ClearField("external_data")
                    tensor.data_location = onnx.TensorProto.DEFAULT
                    tensor.raw_data = data
                continue
            digest = hashlib.sha256()
            length = 0
            for chunk in _tensor_chunks(tensor, root):
                digest.update(chunk)
                length += len(chunk)
            key = (tensor.data_type, tuple(tensor.dims), length, digest.digest())
            if key not in index:
                index[key] = stream.tell()
                for chunk in _tensor_chunks(tensor, root):
                    stream.write(chunk)
            set_tensor_external(tensor, path.name, index[key], length)
        stream.flush()
        os.fsync(stream.fileno())


def _live_data_files(root, excluded):
    inventory = inventory_graphs(root)
    return {
        root / dependency
        for name, dependencies in inventory.items()
        if not any(same_entry(root / name, selected) for selected in excluded)
        for dependency in dependencies
    }


def shared_data_requires_publication(staged, target):
    if target.is_symlink():
        raise ValueError("Canonical shared data cannot be a symlink; rebuild in a clean output directory")
    validate_generated_destinations(target.parent, [target.name])
    if target.exists():
        identical = (
            target.is_file()
            and staged.stat().st_size == target.stat().st_size
            and file_digest(staged) == file_digest(target)
        )
        if identical:
            return False
        raise ValueError("Canonical shared data has incompatible bytes; rebuild in a clean output directory")
    return True


def _publish_data(staged, target):
    if shared_data_requires_publication(staged, target):
        os.replace(staged, target)


def _validate_models(root, names):
    for name in names:
        external_data_files(root / name)
        onnx.checker.check_model(str(root / name))


def _validate_runtime_models(root, names):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for name in names:
        ort.InferenceSession(str(root / name), options, providers=["CPUExecutionProvider"])


def share_external_models(model_dir, suffix=None):
    root = Path(model_dir).resolve()
    names = _decoder_filenames(suffix)
    if not all((root / name).is_file() for name in names):
        raise FileNotFoundError(f"Both split decoder models are required in {root}")
    models = [onnx.load(root / name, load_external_data=False) for name in names]
    old_files = set().union(*(external_data_files(root / name) for name in names))
    shared_name = _shared_data_name(suffix)
    validate_generated_destinations(root, [shared_name])
    _validate_models(root, names)
    storage = [
        (tensor, _can_share(tensor, root, initializer))
        for model in models
        for tensor, initializer in storage_tensors(model)
    ]
    live = _live_data_files(root, {root / name for name in names})
    canonical = root / shared_name
    already_shared = (
        canonical.is_file()
        and old_files <= {canonical}
        and all(
            tensor.data_location == onnx.TensorProto.EXTERNAL and external_tensor_range(root, tensor)[0] == canonical
            for tensor, shareable in storage
            if shareable
        )
        and (bool(old_files) or (canonical.stat().st_size == 0 and not any(shareable for _, shareable in storage)))
    )
    if already_shared:
        _validate_runtime_models(root, names)
        return True
    conflicts = {path for path in live | old_files if any(same_entry(path, root / name) for name in names)}
    if conflicts:
        raise ValueError(f"ONNX transport destination conflict: {', '.join(sorted(path.name for path in conflicts))}")
    # A process exit must not leave transient graphs inside the recursive consumer inventory.
    with ArtifactStaging(prefix=".share-", dir=root.parent) as staging:
        stage = staging.path
        _write_shared_data(storage, root, stage / shared_name)
        for model, name in zip(models, names):
            with open(stage / name, "wb") as stream:
                stream.write(model.SerializeToString())
                stream.flush()
                os.fsync(stream.fileno())
        _validate_models(stage, names)
        _validate_runtime_models(stage, names)
        _publish_data(stage / shared_name, root / shared_name)
        for name in names:
            os.replace(stage / name, root / name)
        _validate_models(root, names)
        staging.mark_complete()
    return True


def share_inline_models(model_dir, suffix=None):
    return share_external_models(model_dir, suffix=suffix)


def verify_model(model_dir, filename):
    try:
        _validate_models(Path(model_dir), [filename])
        return True
    except Exception as error:
        print(f"{filename}: FAILED: {error}")
        return False


def verify_inference(model_dir, suffix=None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        for name in _decoder_filenames(suffix):
            ort.InferenceSession(str(Path(model_dir) / name), opts, providers=["CPUExecutionProvider"])
        return True
    except Exception as error:
        print(f"ORT load FAILED: {error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Share weights between split decoder models")
    parser.add_argument("model_dir")
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--also-int8", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    dirs = [args.model_dir]
    if args.also_int8:
        dirs.append(args.model_dir.rstrip("/") + "-int8")
    try:
        for model_dir in dirs:
            share_external_models(model_dir, suffix=args.suffix)
            if args.verify and not verify_inference(model_dir, suffix=args.suffix):
                return 1
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
