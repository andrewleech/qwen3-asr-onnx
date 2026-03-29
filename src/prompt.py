"""
Prompt template construction and token ID constants for Qwen3-ASR.

The prompt structure for ASR transcription:
    <|im_start|>system\n<|im_end|>\n
    <|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_pad|><|audio_end|><|im_end|>\n
    <|im_start|>assistant\n

Where <|audio_pad|> is repeated N times (N = encoder output sequence length).
"""

# Special token IDs — shared across all Qwen3-ASR model sizes; validated by
# verify_special_tokens() in export.py at export time against the actual tokenizer.
ENDOFTEXT_TOKEN_ID = 151643  # <|endoftext|> - pad token, also EOS
IM_START_TOKEN_ID = 151644  # <|im_start|>
IM_END_TOKEN_ID = 151645  # <|im_end|> - also EOS
AUDIO_START_TOKEN_ID = 151669  # <|audio_start|>
AUDIO_END_TOKEN_ID = 151670  # <|audio_end|>
AUDIO_PAD_TOKEN_ID = 151676  # <|audio_pad|> - replaced by encoder output
ASR_TEXT_TOKEN_ID = 151704  # <asr_text>

# Both are EOS tokens — generation stops on either
EOS_TOKEN_IDS = [ENDOFTEXT_TOKEN_ID, IM_END_TOKEN_ID]

# Newline token (byte-level BPE for '\n')
NEWLINE_TOKEN_ID = 198


def get_feat_extract_output_lengths(input_lengths: int) -> int:
    """Compute the number of audio tokens the encoder produces from a given mel frame count.

    Delegates to encoder_wrapper._get_feat_extract_output_lengths which uses
    ONNX-safe integer division.
    """
    from src.encoder_wrapper import _get_feat_extract_output_lengths

    return _get_feat_extract_output_lengths(input_lengths)  # type: ignore[no-any-return]


def build_prompt_ids(audio_token_count: int, language: str | None = None) -> list[int]:
    """
    Build the full prompt token ID sequence for ASR transcription.

    Args:
        audio_token_count: Number of encoder output tokens. Use
            get_feat_extract_output_lengths(mel_frames) to compute this
            from mel frame count.
            This determines how many <|audio_pad|> tokens are inserted.
        language: Optional language name to force (e.g. "English"). If provided,
            the assistant prefix becomes "language {name}<asr_text>".

    Returns:
        List of token IDs forming the complete prompt.
    """
    # System turn (empty)
    ids = [
        IM_START_TOKEN_ID,
        # "system" = tokens for the word, but in practice the tokenizer
        # encodes "system\n" as specific token IDs. We use the known encoding:
        # system = [9125], \n = [198]
        9125,
        NEWLINE_TOKEN_ID,
        IM_END_TOKEN_ID,
        NEWLINE_TOKEN_ID,
    ]

    # User turn with audio
    ids.extend(
        [
            IM_START_TOKEN_ID,
            # "user" = [882], \n = [198]
            882,
            NEWLINE_TOKEN_ID,
            AUDIO_START_TOKEN_ID,
        ]
    )

    # Audio pad tokens — these get replaced by encoder output
    ids.extend([AUDIO_PAD_TOKEN_ID] * audio_token_count)

    ids.extend(
        [
            AUDIO_END_TOKEN_ID,
            IM_END_TOKEN_ID,
            NEWLINE_TOKEN_ID,
        ]
    )

    # Assistant turn (generation prefix)
    ids.extend(
        [
            IM_START_TOKEN_ID,
            # "assistant" = [77091], \n = [198]
            77091,
            NEWLINE_TOKEN_ID,
        ]
    )

    # Optional language forcing
    if language is not None:
        # "language " prefix tokens — this would need proper tokenization
        # For now, we skip language forcing in the export tool
        raise NotImplementedError(
            "Language forcing requires tokenizer access. Use the processor's chat template for language-forced prompts."
        )

    return ids


def get_audio_pad_range(prompt_ids: list[int]) -> tuple[int, int]:
    """
    Find the start and end indices of <|audio_pad|> tokens in the prompt.

    Returns:
        (start_idx, end_idx) — the range [start_idx, end_idx) where audio
        features should replace token embeddings.
    """
    start = None
    end = None
    for i, tid in enumerate(prompt_ids):
        if tid == AUDIO_PAD_TOKEN_ID:
            if start is None:
                start = i
            end = i + 1
    if start is None or end is None:
        raise ValueError("No <|audio_pad|> tokens found in prompt")
    return start, end
