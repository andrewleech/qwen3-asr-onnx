import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src import prompt
from src.artifact_publication import ArtifactStaging, publish_artifacts
from src.artifact_sources import validate_publication_sources
from src.onnx_artifacts import file_digest, inventory_graphs, preflight_artifact_mutations
from src.prompt import tokenizer_fingerprint, validate_prompt_tokenizer

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
CANONICAL_TOKENIZER_FILES = frozenset({"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"})


def tokenizer_asset_names(root) -> set[str]:
    root = Path(root)
    return {name for name in TOKENIZER_FILES if (root / name).is_file()}


def tokenizer_asset_changes(source, prepared):
    source, prepared = Path(source), Path(prepared)
    names = tokenizer_asset_names(prepared)
    return (TOKENIZER_FILES - names) | {
        name
        for name in names
        if not (source / name).is_file() or file_digest(source / name) != file_digest(prepared / name)
    }


def validate_tokenizer_destination(root):
    directory = Path(root) / "additional_chat_templates"
    if directory.is_dir():
        raise ValueError(f"Unsupported tokenizer template directory: {directory}; use a clean output directory")


def _validate_tokenizer_bindings(root, tokenizer=None):
    from transformers import PreTrainedTokenizerFast
    from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
    from transformers.tokenization_utils_base import get_fast_tokenizer_file

    path = Path(root) / "tokenizer_config.json"
    config = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(config, dict):
        raise ValueError("Invalid tokenizer binding configuration")
    tokenizer_class = type(tokenizer) if tokenizer is not None else None
    if tokenizer_class is None and isinstance(config.get("tokenizer_class"), str):
        tokenizer_class = tokenizer_class_from_name(config["tokenizer_class"])
    file_slots = set(PreTrainedTokenizerFast.vocab_files_names)
    if tokenizer_class is not None:
        file_slots.update(tokenizer_class.vocab_files_names)
    for slot in file_slots:
        if slot in config:
            raise ValueError(f"Unsupported tokenizer file binding: {slot}")
    if config.get("from_slow") or any(
        config.get(slot) is not None for slot in ("tokenizer_object", "__slow_tokenizer", "gguf_file")
    ):
        raise ValueError("Unsupported tokenizer construction binding")
    if get_fast_tokenizer_file(config.get("fast_tokenizer_files", [])) != "tokenizer.json":
        raise ValueError("Unsupported tokenizer backend binding; expected tokenizer.json")


def _tokenizer_source_reads(root, tokenizer):
    root = Path(root)
    names = TOKENIZER_FILES | {"config.json", "processor_config.json"} | set(tokenizer.vocab_files_names.values())
    reads = {root / name for name in names if (root / name).is_file()}
    templates = root / "additional_chat_templates"
    if templates.is_dir():
        reads.update(path for path in templates.glob("*.jinja") if path.is_file())
    return reads


def tokenizer_checkpoint_reads(model_id, tokenizer):
    from transformers.utils import cached_file

    root = Path(model_id)
    if not root.is_dir():
        for name in ("tokenizer_config.json", "config.json", "tokenizer.json"):
            resolved = cached_file(model_id, name, local_files_only=True, _raise_exceptions_for_missing_entries=False)
            if resolved is not None:
                root = Path(resolved).parent
                break
        else:
            raise ValueError(f"Cannot establish tokenizer checkpoint source: {model_id}")
    _validate_tokenizer_bindings(root, tokenizer)
    return _tokenizer_source_reads(root, tokenizer)


def validate_tokenizer_assets(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Tokenizer directory does not exist: {root}")
    validate_tokenizer_destination(root)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"Required tokenizer asset missing: {name}")
    try:
        _validate_tokenizer_bindings(root)
        tokenizer = prompt.load_prompt_tokenizer(str(root), local_files_only=True)
        _validate_tokenizer_bindings(root, tokenizer)
        if not tokenizer.is_fast:
            raise ValueError("Unsupported tokenizer binding; expected a fast tokenizer")
        validate_prompt_tokenizer(tokenizer)
        _validate_shipped_templates(root, tokenizer)
        _processor_template(root, tokenizer)
    except Exception as error:
        raise ValueError(f"Invalid ASR tokenizer assets in {root}: {error}") from error
    return tokenizer


def _processor_template(root, tokenizer, *, required=False):
    root = Path(root)
    legacy, jinja = root / "chat_template.json", root / "chat_template.jinja"
    if legacy.is_file():
        value = json.loads(legacy.read_text())["chat_template"]
    elif jinja.is_file():
        value = jinja.read_text()
    else:
        if required:
            raise ValueError("Missing standalone processor chat template; prepare the family before release")
        config = root / "processor_config.json"
        document = json.loads(config.read_text()) if config.is_file() else {}
        value = document.get("chat_template")
        if value is None:
            return tokenizer.get_chat_template()
    if not isinstance(value, str) and not (
        isinstance(value, dict)
        and isinstance(value.get("default"), str)
        and all(isinstance(item, str) for item in value.values())
    ):
        raise ValueError("Processor chat template must have a usable string default")
    selected = tokenizer.chat_template
    try:
        for template in _template_values(value):
            tokenizer.chat_template = template
            validate_prompt_tokenizer(tokenizer)
    finally:
        tokenizer.chat_template = selected
    return value


def validate_processor_template(root):
    tokenizer = validate_tokenizer_assets(root)
    _processor_template(root, tokenizer, required=True)


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
            validate_prompt_tokenizer(tokenizer)
    finally:
        tokenizer.chat_template = selected


@dataclass(frozen=True)
class PreparedTokenizerAssets:
    source: Path
    target: Path
    root: Path
    names: frozenset[str]
    reads: frozenset[Path]
    noop: bool = False
    add_only: bool = False

    @property
    def read_files(self):
        return {str(path.relative_to(self.source)) for path in self.reads}

    @property
    def removed(self):
        return frozenset() if self.noop or self.add_only else TOKENIZER_FILES - self.names

    def publish(self, *, written_files=()):
        if self.noop:
            return
        replacements = {"chat_template.json"} if self.add_only else self.names
        preflight_artifact_mutations(
            self.root,
            self.target,
            (),
            copied_assets=self.names,
            removed=self.removed,
            existing=inventory_graphs(self.target, written_files) if self.target.is_dir() else {},
        )
        validate_publication_sources(self.reads, files=(self.target / name for name in replacements | self.removed))
        self.target.mkdir(parents=True, exist_ok=True)
        with ArtifactStaging(prefix=".tokenizer-", dir=self.target) as publication:
            for name in replacements:
                shutil.copyfile(self.root / name, publication.path / name)
            publish_artifacts(
                [(publication.path / name, self.target / name) for name in sorted(replacements)],
                [self.target / name for name in sorted(self.removed)],
            )
            publication.mark_complete()


def _same_tokenizer(first, second):
    return tokenizer_fingerprint(first) == tokenizer_fingerprint(second) and first.chat_template == second.chat_template


@contextmanager
def prepare_tokenizer_assets(source, target):
    source, target = Path(source), Path(target)
    validate_tokenizer_destination(target)
    if not tokenizer_asset_names(source):
        yield None
        return
    tokenizer = validate_tokenizer_assets(source)
    reads = frozenset(_tokenizer_source_reads(source, tokenizer))
    processor_template = _processor_template(source, tokenizer)
    same_root = source.resolve() == target.resolve()
    complete = any((source / name).is_file() for name in ("chat_template.json", "chat_template.jinja"))
    if same_root and complete:
        yield PreparedTokenizerAssets(source, target, source, frozenset(tokenizer_asset_names(source)), reads, True)
        return
    mutations = {"chat_template.json"} if same_root else TOKENIZER_FILES
    validate_publication_sources(reads, files=(target / name for name in mutations))
    with ArtifactStaging(prefix=".tokenizer-", dir=None) as staging:
        copied = staging.path / "copied"
        copied.mkdir()
        names = frozenset(tokenizer_asset_names(source))
        for name in names:
            shutil.copyfile(source / name, copied / name)
        try:
            config = json.loads((copied / "tokenizer_config.json").read_text())
            independent = same_root or (
                bool(config.get("tokenizer_class"))
                and not config.get("fix_mistral_regex")
                and _same_tokenizer(tokenizer, validate_tokenizer_assets(copied))
            )
        except ValueError:
            independent = False
        prepared = copied
        if not independent:
            prepared = staging.path / "canonical"
            tokenizer.save_pretrained(prepared, legacy_format=False, save_jinja_files=False)
            if {path.name for path in prepared.iterdir()} != CANONICAL_TOKENIZER_FILES:
                raise ValueError("Unsupported serialized tokenizer family")
            config_path = prepared / "tokenizer_config.json"
            config = json.loads(config_path.read_text())
            # The serialized backend already contains the source-selected regex behavior.
            config["fix_mistral_regex"] = False
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            if not _same_tokenizer(tokenizer, validate_tokenizer_assets(prepared)):
                raise ValueError("Serialized tokenizer family changes tokenizer identity")
            names = CANONICAL_TOKENIZER_FILES
        if not independent or not complete:
            (prepared / "chat_template.json").write_text(
                json.dumps({"chat_template": processor_template}, indent=2) + "\n"
            )
            names = names | {"chat_template.json"}
        if not same_root and not _same_tokenizer(tokenizer, validate_tokenizer_assets(prepared)):
            raise ValueError("Prepared tokenizer family changes tokenizer identity")
        if same_root:
            _processor_template(prepared, tokenizer, required=True)
        else:
            validate_processor_template(prepared)
        yield PreparedTokenizerAssets(source, target, prepared, names, reads, add_only=same_root)
        staging.mark_complete()


def copy_tokenizer_assets(source, target) -> None:
    with prepare_tokenizer_assets(source, target) as prepared:
        if prepared is None:
            validate_tokenizer_assets(source)
        else:
            prepared.publish()
