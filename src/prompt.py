import json
from numbers import Integral

ENDOFTEXT_TOKEN_ID = 151643
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_PAD_TOKEN_ID = 151676
ASR_TEXT_TOKEN_ID = 151704
EOS_TOKEN_IDS = [ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID]
NEWLINE_TOKEN_ID = 198


def load_prompt_tokenizer(model_id: str, *, local_files_only: bool = False):
    from transformers import AutoTokenizer
    from transformers.utils import cached_file

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, local_files_only=local_files_only)
    if not tokenizer.chat_template:
        template_path = cached_file(model_id, "chat_template.json", local_files_only=local_files_only)
        if template_path is None:
            raise ValueError(f"Missing official chat_template.json for {model_id}")
        with open(template_path) as stream:
            tokenizer.chat_template = json.load(stream)["chat_template"]
    validate_prompt_tokenizer(tokenizer)
    return tokenizer


def validate_prompt_tokenizer(tokenizer) -> None:
    special = {
        "<|endoftext|>": ENDOFTEXT_TOKEN_ID,
        "<|im_start|>": IM_START_TOKEN_ID,
        "<|im_end|>": IM_END_TOKEN_ID,
        "<|audio_start|>": AUDIO_START_TOKEN_ID,
        "<|audio_end|>": AUDIO_END_TOKEN_ID,
        "<|audio_pad|>": AUDIO_PAD_TOKEN_ID,
        "<asr_text>": ASR_TEXT_TOKEN_ID,
    }
    for token, expected in special.items():
        if tokenizer.convert_tokens_to_ids(token) != expected:
            raise ValueError(f"Tokenizer has an invalid ASR special token: {token}")
    for role in ("system", "user", "assistant"):
        ids = tokenizer.encode(role, add_special_tokens=False)
        if not ids or tokenizer.unk_token_id in ids:
            raise ValueError(f"Tokenizer cannot encode the {role} role")
    for context, language in (("", None), ("Kepler", None), ("", "English"), ("Kepler", "English")):
        build_prompt_ids(2, tokenizer, context=context, language=language)


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    from src.encoder_wrapper import _get_feat_extract_output_lengths

    return int(_get_feat_extract_output_lengths(input_lengths))


def build_prompt_ids(audio_token_count: int, tokenizer, language: str | None = None, context: str = "") -> list[int]:
    if isinstance(audio_token_count, bool) or not isinstance(audio_token_count, Integral) or audio_token_count <= 0:
        raise ValueError("audio_token_count must be a positive integer")
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise ValueError("language must be a nonempty string")
    if language is not None:
        from qwen_asr.inference.utils import normalize_language_name, validate_language

        language = normalize_language_name(language)
        validate_language(language)
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    expected_text = (
        f"<|im_start|>system\n{context}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if text != expected_text:
        raise ValueError("Tokenizer chat template does not implement the Qwen3-ASR prompt contract")
    if language is not None:
        text += f"language {language}<asr_text>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    start, end = get_audio_pad_range(ids)
    if end - start != 1:
        raise ValueError("Chat template must contain exactly one audio placeholder")
    ids[start:end] = [AUDIO_PAD_TOKEN_ID] * int(audio_token_count)
    get_audio_pad_range(ids, expected_count=int(audio_token_count))
    return ids


def get_audio_pad_range(prompt_ids: list[int], expected_count: int | None = None) -> tuple[int, int]:
    pads = [i for i, token in enumerate(prompt_ids) if token == AUDIO_PAD_TOKEN_ID]
    if not pads:
        raise ValueError("No <|audio_pad|> tokens found in prompt")
    start, end = pads[0], pads[-1] + 1
    if len(pads) != end - start:
        raise ValueError("Audio placeholders must form one contiguous range")
    if (
        start == 0
        or end == len(prompt_ids)
        or prompt_ids[start - 1] != AUDIO_START_TOKEN_ID
        or prompt_ids[end] != AUDIO_END_TOKEN_ID
    ):
        raise ValueError("Audio placeholders must be enclosed by audio delimiters")
    if prompt_ids.count(AUDIO_START_TOKEN_ID) != 1 or prompt_ids.count(AUDIO_END_TOKEN_ID) != 1:
        raise ValueError("Prompt must contain exactly one audio segment")
    if expected_count is not None and len(pads) != expected_count:
        raise ValueError(f"Audio placeholder count {len(pads)} != expected {expected_count}")
    return start, end
