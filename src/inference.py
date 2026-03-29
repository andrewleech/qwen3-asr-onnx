"""Shared ONNX greedy-decode loop for Qwen3-ASR."""

import numpy as np

from src.prompt import EOS_TOKEN_IDS, get_audio_pad_range


def greedy_decode_onnx(
    sessions: dict,
    embed_tokens: np.ndarray,
    audio_features: np.ndarray,
    prompt_ids: list[int],
    max_tokens: int = 256,
) -> list[int]:
    """
    Greedy decode using ONNX Runtime sessions.

    Supports both v1 (input_embeds for both init and step) and v3 (input_ids
    for init, input_embeds for step) decoder formats. The format is detected
    by checking decoder_init's input names.

    Args:
        sessions: dict with keys "decoder_init" and "decoder_step" (ORT sessions)
        embed_tokens: [vocab, hidden] embedding matrix
        audio_features: [1, audio_len, hidden] encoder output
        prompt_ids: list of token IDs including <|audio_pad|> placeholders
        max_tokens: maximum number of tokens to generate

    Returns:
        List of generated token IDs (including EOS if hit).
    """
    position_ids = np.arange(len(prompt_ids), dtype=np.int64)[np.newaxis, :]

    # Detect decoder format from input names
    init_input_names = {inp.name for inp in sessions["decoder_init"].get_inputs()}

    if "input_ids" in init_input_names:
        # v3 format: decoder_init accepts input_ids + audio_features + audio_offset
        audio_start, _ = get_audio_pad_range(prompt_ids)
        input_ids = np.array(prompt_ids, dtype=np.int64)[np.newaxis, :]
        audio_offset = np.array([audio_start], dtype=np.int64)

        logits, present_keys, present_values = sessions["decoder_init"].run(
            ["logits", "present_keys", "present_values"],
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "audio_features": audio_features,
                "audio_offset": audio_offset,
            },
        )
    else:
        # v1 format: decoder_init accepts input_embeds
        input_embeds = embed_tokens[prompt_ids].copy()

        audio_start, audio_end = get_audio_pad_range(prompt_ids)
        audio_len = audio_end - audio_start
        if audio_features.shape[1] != audio_len:
            raise ValueError(f"Audio feature length {audio_features.shape[1]} != audio_pad count {audio_len}")
        input_embeds[audio_start:audio_end] = audio_features[0]
        input_embeds = input_embeds[np.newaxis, :, :]

        logits, present_keys, present_values = sessions["decoder_init"].run(
            ["logits", "present_keys", "present_values"],
            {"input_embeds": input_embeds, "position_ids": position_ids},
        )

    next_token = int(np.argmax(logits[0, -1, :]))
    output_tokens = [next_token]

    if next_token in EOS_TOKEN_IDS:
        return output_tokens

    # Autoregressive loop — decoder_step always accepts input_embeds
    pos = len(prompt_ids)
    for _ in range(max_tokens - 1):
        token_embed = embed_tokens[next_token][np.newaxis, np.newaxis, :]
        step_pos = np.array([[pos]], dtype=np.int64)

        logits, present_keys, present_values = sessions["decoder_step"].run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": token_embed,
                "position_ids": step_pos,
                "past_keys": present_keys,
                "past_values": present_values,
            },
        )

        next_token = int(np.argmax(logits[0, -1, :]))
        output_tokens.append(next_token)
        pos += 1

        if next_token in EOS_TOKEN_IDS:
            break

    return output_tokens
