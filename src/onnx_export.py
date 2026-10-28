from pathlib import Path

import onnx
import torch

from src.artifact_publication import ArtifactStaging, publish_artifacts
from src.onnx_artifacts import external_data_files, external_tensor_range, model_tensors, preflight_artifact_mutations
from src.onnx_fixup import fix_reshape_allowzero


def _consolidate_external_data(model, source, target):
    ranges = {}
    with target.open("wb") as destination:
        for tensor in model_tensors(model):
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            key = external_tensor_range(source, tensor)
            if key not in ranges:
                path, offset, length = key
                ranges[key] = destination.tell()
                with path.open("rb") as stream:
                    stream.seek(offset)
                    remaining = length
                    while remaining:
                        chunk = stream.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError(f"External tensor data ended early: {path}")
                        destination.write(chunk)
                        remaining -= len(chunk)
            del tensor.external_data[:]
            for name, value in (("location", target.name), ("offset", ranges[key]), ("length", key[2])):
                entry = tensor.external_data.add()
                entry.key, entry.value = name, str(value)


def export_onnx(model, inputs, output_path, *, postprocess=None, **kwargs):
    output = Path(output_path).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Legacy Torch writes sidecars by tensor name, independently of its graph filename.
    with ArtifactStaging(prefix="qwen-export-", dir=None) as trace:
        traced = trace.path / "model.onnx"
        torch.onnx.export(model, inputs, str(traced), dynamo=False, **kwargs)
        fixes = fix_reshape_allowzero(str(traced))
        if postprocess is not None:
            postprocess(str(traced))
        model_proto = onnx.load(traced, load_external_data=False)
        # Publication stays on the destination filesystem; .proto avoids live-graph inventory.
        with ArtifactStaging(prefix=".export-", dir=output.parent) as ready:
            staged = ready.path / "model.proto"
            data = ready.path / (output.name + ".data")
            external = any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model_tensors(model_proto))
            if external:
                _consolidate_external_data(model_proto, trace.path, data)
            staged.write_bytes(model_proto.SerializeToString())
            external_data_files(staged)
            onnx.checker.check_model(str(staged))
            names = {output.name, *([data.name] if external else [])}
            preflight_artifact_mutations(output.parent, output.parent, (), generated=names)
            replacements = [(data, output.parent / data.name)] if external else []
            replacements.append((staged, output))
            publish_artifacts(replacements)
            ready.mark_complete()
        trace.mark_complete()
    return fixes
