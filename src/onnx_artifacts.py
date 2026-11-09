import hashlib
import json
import math
import os
import shutil
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import onnx
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import DecodeError

from src.artifact_publication import ArtifactStaging
from src.artifact_sources import same_entry, validate_publication_sources


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
        if any(same_entry(path, root / entry) for entry in exclude):
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


def retained_quantization(output, suffix, generated, *, existing=None):
    generated = set(generated)
    graphs = inventory_graphs(output, generated) if existing is None else existing
    retained = {
        name
        for name in graphs
        if name.endswith(f"{suffix}.onnx")
        and not any(same_entry(Path(output) / name, Path(output) / selected) for selected in generated)
    }
    config = Path(output) / "config.json"
    if not config.is_file():
        return retained, {}
    previous = json.loads(config.read_text()).get("quantization", {})
    tags = (
        {name: tag for name, tag in previous.items() if f"{name}{suffix}.onnx" in retained}
        if isinstance(previous, dict)
        else {}
    )
    return retained, tags


def prepare_onnx_publication(stage, output, graphs, *, shared_data=()):
    stage, output = Path(stage).absolute(), Path(output)
    graphs = set(graphs)
    data = set()
    replacements = []
    for name in sorted(graphs):
        path = stage / name
        dependencies = external_data_files(path)
        allowed = {stage / f"{name}.data", *(stage / name for name in shared_data)}
        if dependencies - allowed:
            raise ValueError(f"Unexpected converter output dependencies: {name}")
        onnx.checker.check_model(str(path))
        data.update(dependencies)
        private = path.with_suffix(".proto")
        path.rename(private)
        replacements.append((private, output / name))
    return (
        [(path, output / path.name) for path in sorted(data)] + replacements,
        graphs | {path.name for path in data},
    )


def preflight_artifact_mutations(
    source,
    output,
    copies,
    *,
    generated=(),
    rewritten=(),
    removed=(),
    reads=(),
    read_files=(),
    existing=None,
    copied_assets=(),
    asset_source=None,
):
    source_root = Path(source).absolute()
    destination_root = Path(output).absolute()
    source, output = Path(source).resolve(), destination_root.resolve()
    generated, rewritten, removed = set(generated), set(rewritten), set(removed)
    read_files = set(read_files)
    validate_generated_destinations(destination_root, generated)
    assets = set(copied_assets)
    asset_root = source if asset_source is None else Path(asset_source)
    copy_reads = {name for name in copies if (source / name).is_file()}
    files = copy_reads | {name for name in assets if (asset_root / name).is_file()} if source != output else set()
    incoming = inventory_graphs(source, set(generated) - set(reads) if source == output else ())
    copied_graphs = files & incoming.keys()
    dependencies = set().union(*(incoming[name] for name in copied_graphs))
    files.update(dependencies)
    validate_destination_parents(destination_root, files | generated | rewritten | removed | assets)
    selected_reads = copied_graphs | {name for name in reads if (source / name).is_file()}
    protected_graphs = selected_reads | (read_files & incoming.keys())
    source_graphs = {source / name for name in protected_graphs}
    source_dependencies = set().union(*(external_data_files(path) for path in source_graphs))
    source_entries = source_graphs | {
        source / name for name in files | copy_reads | read_files if (source / name).is_file()
    }
    if source == output:
        protected_source = source_dependencies
        source_mutations = generated | rewritten | removed
    else:
        protected_source = source_dependencies | source_entries | {path.resolve() for path in source_entries}
        source_mutations = files | generated | rewritten | removed | assets
    source_conflicts = {name for name in source_mutations if output / name in protected_source}
    if source_conflicts:
        raise ValueError(f"ONNX source graph destination conflict: {', '.join(sorted(source_conflicts))}")
    source_reads = source_entries | source_dependencies
    validate_publication_sources(
        [*source_reads, *(source_root / path.relative_to(source) for path in source_reads)],
        files=(destination_root / name for name in source_mutations),
        direct_rewrites=(
            [destination_root / name for name in (generated & selected_reads) | rewritten] if source == output else ()
        ),
    )
    conflicts = (files & generated) | (dependencies & (rewritten | removed))
    for name in dependencies & assets:
        if file_digest(source / name) != file_digest(asset_root / name):
            conflicts.add(name)
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
    surviving = {
        name
        for name in existing
        if not any(same_entry(output / name, output / replacement) for replacement in copied_graphs | generated)
    }
    reserved = set(surviving) | set().union(*(existing[name] for name in surviving))
    conflicts = {
        name
        for name in generated | rewritten | removed
        if any(same_entry(output / name, output / retained) for retained in reserved)
    }
    for name in files.copy():
        if not any(same_entry(output / name, output / retained) for retained in reserved):
            continue
        original, retained = (asset_root if name in assets else source) / name, output / name
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


def graph_file_closure(root, names):
    root = Path(root).resolve()
    graphs = set(names)
    return graphs | _graph_dependencies(root, graphs)


def _graph_dependencies(root, names):
    return {path.relative_to(root).as_posix() for name in names for path in external_data_files(root / name)}


def transport_graph_files(root, graphs, generated=(), rewritten=()):
    root = Path(root).resolve()
    graphs = set(graphs)
    dependencies = _graph_dependencies(root, graphs)
    files = graphs | dependencies
    conflicts = (files & set(generated)) | (dependencies & set(rewritten))
    if conflicts:
        raise ValueError(f"ONNX transport destination conflict: {', '.join(sorted(conflicts))}")
    return files


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


def copy_artifact_entry(source, output, name):
    root = Path(output).resolve()
    validate_destination_parents(root, [name])
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_destination_parents(root, [name])
    with ArtifactStaging(prefix=".artifact-", dir=destination.parent) as staging:
        staged = staging.path / "entry"
        shutil.copy2(Path(source) / name, staged)
        os.replace(staged, destination)
        staging.mark_complete()


def write_artifact_json(output, name, value):
    root = Path(output).resolve()
    validate_destination_parents(root, [name])
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_destination_parents(root, [name])
    with ArtifactStaging(prefix=".artifact-", dir=destination.parent) as staging:
        staged = staging.path / "value.json"
        with staged.open("w") as stream:
            json.dump(value, stream, indent=2)
        staged.chmod(0o644)
        os.replace(staged, destination)
        staging.mark_complete()


def validate_quantization_storage(model):
    def external_occurrences(tensors):
        return Counter(
            tensor.SerializeToString(deterministic=True)
            for tensor in tensors
            if tensor.data_location == onnx.TensorProto.EXTERNAL
        )

    requested = external_occurrences(model_tensors(model))
    # ONNX loading and saving share this narrower traversal, so metadata storage can be lost.
    supported = external_occurrences(onnx.external_data_helper._get_all_tensors(model))
    if requested != supported:
        raise ValueError("Quantization of this external tensor storage is unsupported by ONNX's loader and writer")


@contextmanager
def isolated_onnx_input(input_path, *, prefix="qwen-onnx-"):
    source = Path(input_path).absolute()
    model = onnx.load(source, load_external_data=False)
    validate_quantization_storage(model)
    dependencies = sorted(external_data_files(source))
    with ArtifactStaging(prefix=prefix, dir=None) as staging:
        stage = staging.path
        # ORT writes inferred graphs beside its input; tensor storage needs a separate namespace.
        locations = {path: f"weights/{index}.data" for index, path in enumerate(dependencies)}
        for path, location in locations.items():
            target = stage / location
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
        for tensor in model_tensors(model):
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            info = get_external_info(tensor)
            dependency = (source.parent / info["location"]).resolve()
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = locations[dependency]
        staged_model = stage / "model.onnx"
        staged_model.write_bytes(model.SerializeToString())
        yield staged_model, stage
        staging.mark_complete()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()
