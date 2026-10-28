import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def embedding_spec(config: Mapping) -> tuple[tuple[int, int], np.dtype]:
    decoder = config.get("decoder", {})
    if not isinstance(decoder, Mapping):
        raise ValueError("decoder must be an object")
    decoder_shape = (decoder.get("vocab_size"), decoder.get("hidden_size"))
    shape = config.get("embed_tokens_shape", decoder_shape)
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("embed_tokens_shape must contain two positive integers")
    shape = tuple(shape)
    for actual, expected in zip(shape, decoder_shape):
        if expected is not None and (type(expected) is not int or expected <= 0 or actual != expected):
            raise ValueError("Embedding shape disagrees with decoder dimensions")
    dtype = config.get("embed_tokens_dtype", "float32")
    if dtype not in ("float32", "float16"):
        raise ValueError(f"Unsupported embedding dtype: {dtype!r}")
    return shape, np.dtype(dtype)


def validate_embedding_array(array: np.ndarray, config: Mapping) -> None:
    shape, stored_dtype = embedding_spec(config)
    if array.shape != shape:
        raise ValueError(f"Embedding shape {array.shape} does not match {shape}")
    if array.dtype not in (np.dtype("float32"), np.dtype("float16")):
        raise ValueError(f"Unsupported embedding dtype: {array.dtype}")
    if array.dtype != stored_dtype:
        raise ValueError("Embedding dtype disagrees with metadata")
    if not np.isfinite(array).all():
        raise ValueError("Embedding data contains non-finite values")


def load_embed_tokens(model_dir, *, filename="embed_tokens.bin", dtype=None, output_dtype="float32") -> np.ndarray:
    model_dir = Path(model_dir)
    with (model_dir / "config.json").open() as stream:
        config = json.load(stream)
    shape, stored_dtype = embedding_spec(config)
    if dtype is not None:
        if dtype not in ("float32", "float16"):
            raise ValueError(f"Unsupported embedding dtype: {dtype!r}")
        stored_dtype = np.dtype(dtype)
    if output_dtype not in ("float32", "float16"):
        raise ValueError(f"Unsupported embedding output dtype: {output_dtype!r}")
    path = model_dir / filename
    expected = shape[0] * shape[1] * stored_dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{filename} byte count does not match {shape} {stored_dtype}: expected {expected}")
    array = np.fromfile(path, dtype=stored_dtype).reshape(shape)
    validate_embedding_array(array, {**config, "embed_tokens_dtype": str(stored_dtype)})
    with np.errstate(over="ignore", invalid="ignore"):
        converted = array.astype(output_dtype, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(f"Embedding conversion to {output_dtype} overflowed")
    return converted
