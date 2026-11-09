import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from src.artifact_publication import ArtifactStaging, publish_artifacts
from src.prompt import tokenizer_fingerprint as tokenizer_fingerprint


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_fingerprints(root, names):
    root = Path(root)
    result = {}
    for name in sorted(names):
        with (root / name).open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def model_fingerprint(model):
    import torch

    digest = hashlib.sha256()
    config = model.config.to_dict() if hasattr(getattr(model, "config", None), "to_dict") else {}
    attention = getattr(getattr(model, "config", None), "_attn_implementation", None)
    identity = [type(model).__module__, type(model).__qualname__, model.training, attention, config]
    digest.update(_json(identity).encode())
    for name, tensor in sorted(model.state_dict().items()):
        if not torch.is_tensor(tensor):
            raise ValueError(f"Unsupported checkpoint state for calibration cache: {name}")
        metadata = _json([name, str(tensor.dtype), list(tensor.shape)]).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        flat = tensor.detach().reshape(-1)
        chunk_elements = max(1, 4 * 1024 * 1024 // tensor.element_size())
        for start in range(0, flat.numel(), chunk_elements):
            chunk = flat[start : start + chunk_elements].contiguous().cpu().view(torch.uint8).numpy()
            digest.update(memoryview(chunk))
    return digest.hexdigest()


def recipe(kind, inputs, parameters, producer):
    source = Path(__file__).resolve().parent
    return {
        "schema": 1,
        "kind": kind,
        "inputs": inputs,
        "parameters": parameters,
        "implementation": {
            **file_fingerprints(
                source,
                [
                    "calibration_cache.py",
                    "prompt.py",
                    "mel.py",
                    "native_encoder.py",
                    "encoder_wrapper.py",
                    "decoder_wrapper.py",
                    "embeddings.py",
                    "onnx_artifacts.py",
                ],
            ),
            **file_fingerprints(source.parent, ["export.py", "validate.py"]),
            producer.name: hashlib.sha256(producer.read_bytes()).hexdigest(),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "onnx",
                "onnxruntime",
                "transformers",
                "qwen-asr",
                "librosa",
                "datasets",
                "soundfile",
            )
        },
    }


def cache_key(identity):
    return hashlib.sha256(_json(identity).encode()).hexdigest()[:16]


def load_cache(path, identity):
    path = Path(path)
    try:
        if not path.exists() and not path.is_symlink():
            return None
        if not path.is_file():
            raise ValueError("Calibration cache is not a regular file")
        with np.load(path, allow_pickle=False) as archive:
            stored = archive["__recipe__"]
            if stored.shape != () or stored.dtype.kind != "U" or json.loads(stored.item()) != identity:
                raise ValueError("Calibration cache provenance does not match this run")
            arrays = {name: archive[name] for name in archive.files if name != "__recipe__"}
        _validate_arrays(arrays)
        return arrays
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unverifiable calibration cache {path}; preserve it and choose a clean cache path") from error


def _validate_arrays(arrays):
    if "__recipe__" in arrays:
        raise ValueError("Reserved calibration cache key")
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"Calibration cache requires finite numeric arrays: {name}")


def save_cache(path, identity, arrays):
    path = Path(path)
    _validate_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactStaging(prefix=".calibration-", dir=path.parent) as staging:
        staged = staging.path / "cache.npz"
        np.savez(staged, __recipe__=np.array(_json(identity)), **arrays)
        load_cache(staged, identity)
        publish_artifacts([(staged, path)])
        staging.mark_complete()


def pack_samples(groups):
    arrays = {}
    for group, samples in groups.items():
        arrays[f"{group}_count"] = np.array(len(samples), dtype=np.int64)
        for index, sample in enumerate(samples):
            arrays.update({f"{group}_{index}_{name}": value for name, value in sample.items()})
    return arrays


def unpack_samples(arrays, groups):
    result = {}
    consumed = set()
    for group in groups:
        key = f"{group}_count"
        count = arrays[key]
        if count.shape != () or count.dtype.kind not in "iu" or not 0 <= int(count) <= len(arrays):
            raise ValueError("Invalid calibration cache sample count")
        consumed.add(key)
        samples = []
        for index in range(int(count)):
            prefix = f"{group}_{index}_"
            sample = {name[len(prefix) :]: value for name, value in arrays.items() if name.startswith(prefix)}
            if not sample:
                raise ValueError("Incomplete calibration cache sample")
            consumed.update(name for name in arrays if name.startswith(prefix))
            samples.append(sample)
        result[group] = samples
    if consumed != arrays.keys():
        raise ValueError("Unexpected calibration cache arrays")
    return result
