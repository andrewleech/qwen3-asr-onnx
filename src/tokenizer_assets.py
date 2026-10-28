import json
from pathlib import Path

from src import prompt
from src.prompt import validate_prompt_tokenizer as _validate_prompt_tokenizer

TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "chat_template.json",
    }
)


def tokenizer_asset_names(root) -> set[str]:
    root = Path(root)
    return {name for name in TOKENIZER_FILES if (root / name).is_file()}


def validate_tokenizer_assets(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Tokenizer directory does not exist: {root}")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"Required tokenizer asset missing: {name}")
    try:
        tokenizer = prompt.load_prompt_tokenizer(str(root), local_files_only=True)
        _validate_prompt_tokenizer(tokenizer)
        _validate_shipped_templates(root, tokenizer)
    except Exception as error:
        raise ValueError(f"Invalid ASR tokenizer assets in {root}: {error}") from error
    return tokenizer


def _template_values(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict) and value and all(isinstance(item, str) for item in value.values()):
        return list(value.values())
    if (
        isinstance(value, list)
        and value
        and all(
            isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("template"), str)
            for item in value
        )
    ):
        return [item["template"] for item in value]
    raise ValueError("Chat template must be a string or a nonempty named template collection")


def _validate_shipped_templates(root, tokenizer):
    config = json.loads((root / "tokenizer_config.json").read_text())
    templates = []
    if "chat_template" in config:
        templates.extend(_template_values(config["chat_template"]))
    legacy = root / "chat_template.json"
    if legacy.is_file():
        document = json.loads(legacy.read_text())
        if not isinstance(document, dict) or "chat_template" not in document:
            raise ValueError("chat_template.json must contain chat_template")
        templates.extend(_template_values(document["chat_template"]))
    jinja = root / "chat_template.jinja"
    if jinja.is_file():
        templates.append(jinja.read_text())
    selected = tokenizer.chat_template
    try:
        for template in templates:
            tokenizer.chat_template = template
            _validate_prompt_tokenizer(tokenizer)
    finally:
        tokenizer.chat_template = selected
