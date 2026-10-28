"""Template from Qwen/Qwen3-ASR-0.6B revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from src import prompt

OFFICIAL_CHAT_TEMPLATE = "{%- set ns = namespace(system_text=\"\") -%}\n{%- for m in messages -%}\n  {%- if m.role == 'system' -%}\n    {%- if m.content is string -%}\n      {%- set ns.system_text = ns.system_text + m.content -%}\n    {%- else -%}\n      {%- for c in m.content -%}\n        {%- if c.type == 'text' and (c.text is defined) -%}\n          {%- set ns.system_text = ns.system_text + c.text -%}\n        {%- endif -%}\n      {%- endfor -%}\n    {%- endif -%}\n  {%- endif -%}\n{%- endfor -%}\n\n{%- set ns2 = namespace(audio_tokens=\"\") -%}\n{%- for m in messages -%}\n  {%- if m.content is not string -%}\n    {%- for c in m.content -%}\n      {%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}\n        {%- set ns2.audio_tokens = ns2.audio_tokens + \"<|audio_start|><|audio_pad|><|audio_end|>\" -%}\n      {%- endif -%}\n    {%- endfor -%}\n  {%- endif -%}\n{%- endfor -%}\n\n{{- '<|im_start|>system\\n' + (ns.system_text if ns.system_text is string else '') + '<|im_end|>\\n' -}}\n{{- '<|im_start|>user\\n' + ns2.audio_tokens + '<|im_end|>\\n' -}}\n{%- if add_generation_prompt -%}\n{{- '<|im_start|>assistant\\n' -}}\n{%- endif -%}"


def save_tokenizer_assets(root, template_format="jinja", include_template=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    special = {
        "<|endoftext|>": prompt.ENDOFTEXT_TOKEN_ID,
        "<|im_start|>": prompt.IM_START_TOKEN_ID,
        "<|im_end|>": prompt.IM_END_TOKEN_ID,
        "<|audio_start|>": prompt.AUDIO_START_TOKEN_ID,
        "<|audio_end|>": prompt.AUDIO_END_TOKEN_ID,
        "<|audio_pad|>": prompt.AUDIO_PAD_TOKEN_ID,
        "<asr_text>": prompt.ASR_TEXT_TOKEN_ID,
    }
    vocab = {
        "[UNK]": 0,
        "system": 31,
        "user": 42,
        "assistant": 53,
        "language": 64,
        "English": 75,
        "Kepler": 86,
    } | special
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": index,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token, index in special.items()
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }
    (root / "tokenizer.json").write_text(json.dumps(tokenizer))
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "[UNK]",
        "additional_special_tokens": list(special),
        "origin_marker": "tiny-local-fixture",
    }
    for name in ("chat_template.jinja", "chat_template.json"):
        (root / name).unlink(missing_ok=True)
    if include_template:
        if template_format == "jinja":
            (root / "chat_template.jinja").write_text(OFFICIAL_CHAT_TEMPLATE)
        elif template_format == "json":
            (root / "chat_template.json").write_text(json.dumps({"chat_template": OFFICIAL_CHAT_TEMPLATE}))
        elif template_format == "embedded":
            config["chat_template"] = OFFICIAL_CHAT_TEMPLATE
        else:
            raise ValueError(f"Unknown template format: {template_format}")
    (root / "tokenizer_config.json").write_text(json.dumps(config))
    (root / "special_tokens_map.json").write_text(
        json.dumps({"unk_token": "[UNK]", "additional_special_tokens": list(special)})
    )
    if include_template:
        return prompt.load_prompt_tokenizer(str(root), local_files_only=True)
    return AutoTokenizer.from_pretrained(root, local_files_only=True)
