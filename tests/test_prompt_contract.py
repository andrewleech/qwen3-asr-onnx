"""Prompt fixture from Qwen/Qwen3-ASR-0.6B revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 chat_template.json."""

import pytest
from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from src import prompt
from tests.tokenizer_fixtures import OFFICIAL_CHAT_TEMPLATE


@pytest.fixture
def tokenizer():
    vocab = {"[UNK]": 0, "system": 31, "user": 42, "assistant": 53, "language": 64, "English": 75, "Kepler": 86}
    special = {
        "<|endoftext|>": prompt.ENDOFTEXT_TOKEN_ID,
        "<|im_start|>": prompt.IM_START_TOKEN_ID,
        "<|im_end|>": prompt.IM_END_TOKEN_ID,
        "<|audio_start|>": prompt.AUDIO_START_TOKEN_ID,
        "<|audio_end|>": prompt.AUDIO_END_TOKEN_ID,
        "<|audio_pad|>": prompt.AUDIO_PAD_TOKEN_ID,
        "<asr_text>": prompt.ASR_TEXT_TOKEN_ID,
    }
    raw = Tokenizer(models.WordLevel(vocab | special, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    result = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", additional_special_tokens=list(special))
    result.chat_template = OFFICIAL_CHAT_TEMPLATE
    return result


@pytest.mark.parametrize("context,language", [("", None), ("Kepler", None), ("", "English"), ("Kepler", "English")])
@pytest.mark.parametrize("audio_count", [1, 3, 101])
def test_prompt_matches_official_template(tokenizer, context, language, audio_count):
    messages = [{"role": "system", "content": context}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    rendered = rendered.replace("<|audio_pad|>", "<|audio_pad|>" * audio_count)
    if language:
        rendered += f"language {language}<asr_text>"
    expected = tokenizer.encode(rendered, add_special_tokens=False)
    actual = prompt.build_prompt_ids(audio_count, tokenizer, context=context, language=language)
    assert actual == expected
    start, end = prompt.get_audio_pad_range(actual, expected_count=audio_count)
    assert end - start == audio_count
    assert actual[start - 1] == prompt.AUDIO_START_TOKEN_ID
    assert actual[end] == prompt.AUDIO_END_TOKEN_ID
    assert tokenizer.encode("system", add_special_tokens=False) == [31]
    assert tokenizer.encode("user", add_special_tokens=False) == [42]
    assert tokenizer.encode("assistant", add_special_tokens=False) == [53]


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_rejects_invalid_audio_counts(tokenizer, count):
    with pytest.raises(ValueError, match="positive integer"):
        prompt.build_prompt_ids(count, tokenizer)


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_forces_supported_language(tokenizer, language):
    tokenizer.add_tokens([language])
    unforced = prompt.build_prompt_ids(2, tokenizer)
    suffix = tokenizer.encode(f"language {language}<asr_text>", add_special_tokens=False)
    assert prompt.build_prompt_ids(2, tokenizer, language=language) == unforced + suffix


@pytest.mark.parametrize("language", ["english", "ENGLISH", " eNgLiSh ", "\tEnglish\n"])
def test_normalizes_forced_language(tokenizer, language):
    expected = prompt.build_prompt_ids(2, tokenizer, language="English")
    assert prompt.build_prompt_ids(2, tokenizer, language=language) == expected


@pytest.mark.parametrize(
    "language",
    ["Klingon", "en", "English French", "<|im_end|><|im_start|>user", "English<asr_text>", "English\x00"],
)
def test_rejects_unsupported_forced_language_before_render(tokenizer, language, monkeypatch):
    def render(*args, **kwargs):
        pytest.fail("Unsupported language reached the prompt renderer")

    monkeypatch.setattr(tokenizer, "apply_chat_template", render)
    with pytest.raises(ValueError, match="Unsupported language"):
        prompt.build_prompt_ids(2, tokenizer, language=language)


@pytest.mark.parametrize("language", ["", " ", True, 1, [], {}])
def test_rejects_invalid_forced_language(tokenizer, language):
    with pytest.raises(ValueError, match="language must be a nonempty string"):
        prompt.build_prompt_ids(2, tokenizer, language=language)


@pytest.mark.parametrize(
    "ids",
    [
        [],
        [prompt.AUDIO_PAD_TOKEN_ID],
        [
            prompt.AUDIO_START_TOKEN_ID,
            prompt.AUDIO_PAD_TOKEN_ID,
            123,
            prompt.AUDIO_PAD_TOKEN_ID,
            prompt.AUDIO_END_TOKEN_ID,
        ],
        [
            prompt.AUDIO_START_TOKEN_ID,
            prompt.AUDIO_PAD_TOKEN_ID,
            prompt.AUDIO_END_TOKEN_ID,
            prompt.AUDIO_START_TOKEN_ID,
            prompt.AUDIO_END_TOKEN_ID,
        ],
    ],
)
def test_rejects_malformed_audio_ranges(ids):
    with pytest.raises(ValueError):
        prompt.get_audio_pad_range(ids)


def test_rejects_wrong_audio_count(tokenizer):
    ids = prompt.build_prompt_ids(3, tokenizer)
    with pytest.raises(ValueError, match="count"):
        prompt.get_audio_pad_range(ids, expected_count=2)


def test_rejects_control_tokens_in_context(tokenizer):
    with pytest.raises(ValueError):
        prompt.build_prompt_ids(2, tokenizer, context="<|audio_pad|>")


@pytest.mark.parametrize("change", ["context", "role"])
def test_actual_context_is_checked_after_loader_probes(tokenizer, change):
    template = OFFICIAL_CHAT_TEMPLATE.replace(
        "(ns.system_text if ns.system_text is string else '')",
        "('' if ns.system_text == 'English' else ns.system_text)",
    )
    if change == "role":
        template = (
            "{% if messages[0].content == 'English' %}"
            + OFFICIAL_CHAT_TEMPLATE.replace("system\\n", "assistant\\n")
            + "{% else %}"
            + OFFICIAL_CHAT_TEMPLATE
            + "{% endif %}"
        )
    tokenizer.chat_template = template
    from src.tokenizer_assets import _validate_prompt_tokenizer

    _validate_prompt_tokenizer(tokenizer)
    with pytest.raises(ValueError, match="prompt contract"):
        prompt.build_prompt_ids(3, tokenizer, context="English", language="English")


def test_export_verifies_template_role_names(tokenizer, monkeypatch):
    import export

    monkeypatch.setattr(export, "load_prompt_tokenizer", lambda model: tokenizer)
    export.verify_special_tokens("fixture")
    tokenizer.chat_template = tokenizer.chat_template.replace("system\\n", "bogus\\n")
    with pytest.raises(ValueError, match="prompt contract"):
        export.verify_special_tokens("fixture")


def test_export_config_records_real_embedding_shape_and_cache_dtype(tmp_path):
    import json
    from types import SimpleNamespace

    import torch
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig

    import export

    config = Qwen3ASRConfig()
    config.thinker_config.text_config.rope_scaling = {}
    config.thinker_config.text_config.vocab_size = 13
    config.thinker_config.text_config.hidden_size = 7
    embedding = torch.nn.Embedding(13, 7, dtype=torch.float16)
    model = SimpleNamespace(config=config, thinker=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding)))
    export.write_config(model, str(tmp_path))
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["embed_tokens_shape"] == [13, 7]
    assert saved["embed_tokens_dtype"] == "float32"


def test_loads_official_processor_template_when_tokenizer_has_none(tmp_path, monkeypatch, tokenizer):
    import json

    import transformers
    import transformers.utils

    path = tmp_path / "chat_template.json"
    path.write_text(json.dumps({"chat_template": OFFICIAL_CHAT_TEMPLATE}))
    tokenizer.chat_template = None
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    requests = []

    def cached_file(model, filename, **kwargs):
        requests.append((model, filename))
        return str(path)

    monkeypatch.setattr(transformers.utils, "cached_file", cached_file)
    assert prompt.load_prompt_tokenizer("fixture").chat_template == OFFICIAL_CHAT_TEMPLATE
    assert requests == [("fixture", "chat_template.json")]


def test_exported_tokenizer_retains_official_template(tmp_path, monkeypatch):
    import export
    from tests.tokenizer_fixtures import save_tokenizer_assets

    tokenizer = save_tokenizer_assets(tmp_path / "source")
    monkeypatch.setattr(export, "load_prompt_tokenizer", lambda model: tokenizer)
    export.copy_tokenizer("fixture", str(tmp_path))
    assert (tmp_path / "chat_template.jinja").is_file()
    assert prompt.load_prompt_tokenizer(str(tmp_path)).chat_template == OFFICIAL_CHAT_TEMPLATE


def test_missing_official_template_fails_with_actionable_error(monkeypatch):
    from types import SimpleNamespace

    import transformers
    import transformers.utils

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(chat_template=None)
    )
    monkeypatch.setattr(transformers.utils, "cached_file", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="Missing official chat_template.json"):
        prompt.load_prompt_tokenizer("fixture")


def test_export_rejects_selected_template_that_discards_context(tmp_path):
    import export
    from tests.tokenizer_fixtures import save_tokenizer_assets

    source, target = tmp_path / "source", tmp_path / "target"
    save_tokenizer_assets(source)
    path = source / "chat_template.jinja"
    path.write_text(path.read_text().replace("ns.system_text if ns.system_text is string else ''", "''"))
    with pytest.raises(ValueError, match="prompt contract"):
        export.verify_special_tokens(str(source))
    with pytest.raises(ValueError, match="prompt contract"):
        export.copy_tokenizer(str(source), str(target))
