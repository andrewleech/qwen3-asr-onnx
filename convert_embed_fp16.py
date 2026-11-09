"""Convert embeddings using the shape and storage dtype declared by config.json."""

import argparse
import os
import tempfile
from pathlib import Path

from src.artifact_sources import validate_publication_sources
from src.embeddings import load_embed_tokens


def convert_embed_fp16(model_dir):
    root = Path(model_dir)
    embed = load_embed_tokens(root, output_dtype="float16")
    destination = root / "embed_tokens.fp16.bin"
    validate_publication_sources([root / "embed_tokens.bin", root / "config.json"], files=[destination])
    descriptor, tmp = tempfile.mkstemp(prefix=".embed-fp16-", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            embed.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, destination)
    finally:
        Path(tmp).unlink(missing_ok=True)
    print(f"Converted embeddings: {embed.shape}, float16, {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Convert embed_tokens.bin to float16")
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    try:
        convert_embed_fp16(args.model_dir)
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
