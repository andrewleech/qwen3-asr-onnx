import json
from pathlib import Path

import onnx
import onnxruntime as ort

from src.artifact_metadata import MODEL_METADATA_FILES
from src.embeddings import load_embed_tokens
from src.onnx_artifacts import transport_graph_files
from src.tokenizer_assets import (
    tokenizer_asset_changes,
    tokenizer_asset_names,
    validate_processor_template,
    validate_tokenizer_assets,
)

REQUIRED_METADATA = [*MODEL_METADATA_FILES, "tokenizer.json", "tokenizer_config.json"]


def graph_names(suffix=""):
    tag = f".{suffix}" if suffix else ""
    return [f"encoder{tag}.onnx", f"decoder_init{tag}.onnx", f"decoder_step{tag}.onnx"]


def validate_release_metadata(root):
    root = Path(root)
    for name in [*REQUIRED_METADATA, "embed_tokens.bin"]:
        if not (root / name).is_file():
            raise ValueError(f"Required artifact missing: {name}")
    for name in REQUIRED_METADATA:
        with (root / name).open() as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError(f"Required metadata must be a JSON object: {name}")
    validate_tokenizer_assets(root)


def validate_release(root, selected_graphs, *, tokenizer_root=None):
    root = Path(root).resolve()
    names = set(selected_graphs)
    if any(Path(name).name != name for name in names):
        raise ValueError("Release graph roots must be model filenames")
    variants = set()
    for name in names:
        if name == "encoder.onnx":
            variants.add("")
        elif name.startswith("encoder.") and name.endswith(".onnx"):
            variants.add(name[len("encoder.") : -len(".onnx")])
    expected = {name for variant in variants for name in graph_names(variant)}
    if not expected or names != expected:
        raise ValueError("A release requires complete encoder, decoder_init and decoder_step graph sets")
    validate_release_metadata(root)
    tokenizer_root = root if tokenizer_root is None else Path(tokenizer_root)
    validate_processor_template(tokenizer_root)
    load_embed_tokens(root)
    members = {*MODEL_METADATA_FILES, "embed_tokens.bin", *tokenizer_asset_names(tokenizer_root)}
    members.update(transport_graph_files(root, names, rewritten=tokenizer_asset_changes(root, tokenizer_root)))
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    for name in sorted(names):
        path = root / name
        onnx.checker.check_model(str(path))
        ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return members
