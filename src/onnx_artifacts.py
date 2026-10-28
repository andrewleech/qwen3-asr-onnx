import hashlib
import math
from pathlib import Path

import onnx
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import DecodeError


def storage_tensors(message, initializer=False):
    if isinstance(message, onnx.TensorProto):
        yield message, initializer
        return
    for field, value in message.ListFields():
        if field.type != FieldDescriptor.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            for item in value:
                yield from storage_tensors(item, field.name == "initializer")
        else:
            yield from storage_tensors(value, field.name == "initializer")


def model_tensors(message):
    for tensor, _ in storage_tensors(message):
        yield tensor


def get_external_info(tensor):
    return {entry.key: entry.value for entry in tensor.external_data}


def external_tensor_range(root, tensor):
    info = get_external_info(tensor)
    location = info.get("location", "")
    if not location or "\\" in location or Path(location).is_absolute():
        raise ValueError(f"Invalid external data location: {location!r}")
    root = Path(root).resolve()
    if ".." in Path(location).parts:
        raise ValueError(f"Invalid external data location: {location!r}")
    path = root / location
    if path.resolve() != path or not path.is_file():
        raise ValueError(f"External data missing or outside model directory: {location}")
    size = path.stat().st_size
    offset = int(info.get("offset", "0"))
    length = int(info.get("length", str(size - offset)))
    if offset < 0 or length < 0 or offset + length > size:
        raise ValueError(f"Invalid external data range for {tensor.name}")
    dtype_name = onnx.TensorProto.DataType.Name(tensor.data_type)
    if dtype_name == "STRING":
        raise ValueError("External string tensors are unsupported")
    if dtype_name in ("INT4", "UINT4", "FLOAT4E2M1"):
        bits = 4
    elif dtype_name in ("INT2", "UINT2"):
        bits = 2
    else:
        bits = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize * 8
    expected = (math.prod(tensor.dims) * bits + 7) // 8
    if length != expected:
        raise ValueError(f"External tensor {tensor.name} has {length} bytes; expected {expected}")
    return path, offset, length


def external_data_files(model_path):
    path = Path(model_path).absolute()
    model = onnx.load(path, load_external_data=False)
    if not model.HasField("graph") or not model.ir_version:
        raise ValueError(f"Not an ONNX graph: {path}")
    return {
        external_tensor_range(path.parent, tensor)[0]
        for tensor in model_tensors(model)
        if tensor.data_location == onnx.TensorProto.EXTERNAL
    }


def inventory_graphs(root, exclude=()):
    root = Path(root).resolve()
    graphs = {}
    exclude = set(exclude)
    for path in sorted(root.rglob("*.onnx")):
        name = path.relative_to(root).as_posix()
        if name in exclude:
            continue
        if not path.is_file():
            continue
        if path.stat().st_size >= onnx.checker.MAXIMUM_PROTOBUF:
            continue
        # Tensor storage can use this suffix; only decoded models establish graph roles.
        try:
            model = onnx.load(path, load_external_data=False)
        except DecodeError:
            continue
        if not model.HasField("graph") or not model.ir_version:
            continue
        try:
            graphs[name] = {
                external_tensor_range(path.parent, tensor)[0].relative_to(root).as_posix()
                for tensor in model_tensors(model)
                if tensor.data_location == onnx.TensorProto.EXTERNAL
            }
        except Exception as error:
            raise ValueError(
                f"Cannot inventory independent graph {path}; preserve it and repair its references "
                "or move abandoned staging outside the model directory before retrying."
            ) from error
    return graphs


def preflight_artifact_mutations(
    source, output, copies, *, generated=(), rewritten=(), removed=(), reads=(), existing=None, copied_assets=()
):
    destination_root = Path(output).absolute()
    source, output = Path(source).resolve(), destination_root.resolve()
    generated, rewritten, removed = set(generated), set(rewritten), set(removed)
    validate_generated_destinations(destination_root, generated)
    assets = set(copied_assets)
    files = {name for name in set(copies) | assets if (source / name).is_file()} if source != output else set()
    incoming = inventory_graphs(source, set(generated) - set(reads) if source == output else ())
    copied_graphs = files & incoming.keys()
    dependencies = set().union(*(incoming[name] for name in copied_graphs))
    files.update(dependencies)
    validate_destination_parents(destination_root, files | generated | rewritten | removed | assets)
    selected_reads = copied_graphs | {name for name in reads if (source / name).is_file()}
    source_graphs = {source / name for name in selected_reads}
    source_dependencies = set().union(*(external_data_files(path) for path in source_graphs))
    if source == output:
        protected_source = source_dependencies
        source_mutations = generated | rewritten | removed
    else:
        source_entries = source_graphs | {source / name for name in files}
        protected_source = source_dependencies | source_entries | {path.resolve() for path in source_entries}
        source_mutations = files | generated | rewritten | removed | assets
    source_conflicts = {name for name in source_mutations if output / name in protected_source}
    if source_conflicts:
        raise ValueError(f"ONNX source graph destination conflict: {', '.join(sorted(source_conflicts))}")
    conflicts = (files & generated) | (dependencies & (rewritten | removed))
    if source == output:
        read_dependencies = set().union(*(incoming[name] for name in reads if name in incoming))
        conflicts.update(read_dependencies & (generated | rewritten | removed))
    if conflicts:
        raise ValueError(f"ONNX transport destination conflict: {', '.join(sorted(conflicts))}")
    if existing is None:
        existing = (
            incoming
            if source == output
            else (inventory_graphs(output, generated | copied_graphs) if output.is_dir() else {})
        )
    surviving = existing.keys() - copied_graphs - generated
    reserved = set(surviving) | set().union(*(existing[name] for name in surviving))
    conflicts = reserved & (generated | rewritten | removed)
    for name in files & reserved:
        original, retained = source / name, output / name
        if original.stat().st_size == retained.stat().st_size and file_digest(original) == file_digest(retained):
            files.remove(name)
        else:
            conflicts.add(name)
    if conflicts:
        raise ValueError(f"ONNX retained graph destination conflict: {', '.join(sorted(conflicts))}")
    validate_generated_destinations(destination_root, generated)
    for name in rewritten | removed | assets:
        if (output / name).is_dir():
            raise ValueError(f"Artifact destination is a directory: {output / name}")
    return files - assets


def validate_destination_parents(output, names):
    root = Path(output).resolve()
    for name in names:
        relative = Path(name)
        parent = (root / relative).parent
        if relative.is_absolute() or ".." in relative.parts or not parent.is_relative_to(root):
            raise ValueError(
                f"ONNX transport destination conflict: artifact destination escapes output directory: {name}"
            )
        if parent.resolve() != parent:
            raise ValueError(f"ONNX transport destination conflict: linked parent directory: {name}")


def validate_generated_destinations(output, names):
    root = Path(output).absolute()
    if root.is_symlink():
        raise ValueError(f"ONNX transport destination conflict: linked output directory {root}")
    validate_destination_parents(root, names)
    for name in names:
        destination = root / name
        if destination.is_symlink() or (destination.exists() and destination.stat().st_nlink > 1):
            raise ValueError(f"ONNX transport destination conflict: linked {name}")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"ONNX transport destination conflict: not a regular file: {name}")


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()
