import torch


def run_native_encoder(tower, mel: torch.Tensor) -> torch.Tensor:
    """Run the official tower with its native attention-window mask, returning [1, tokens, dim]."""
    if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[2] < 1:
        raise ValueError("Expected mel with shape [1, mel_bins, positive_time]")
    if tower.config._attn_implementation == "flash_attention_2":
        raise ValueError("Native encoder reference requires eager or sdpa attention")

    def apply_window_mask(attention, args, kwargs):
        kwargs["attention_mask"] = tower._prepare_attention_mask(kwargs["hidden_states"], kwargs["cu_seqlens"])
        return args, kwargs

    handles = []
    try:
        for layer in tower.layers:
            handles.append(layer.self_attn.register_forward_pre_hook(apply_window_mask, with_kwargs=True))
        with torch.no_grad():
            lengths = torch.tensor([mel.shape[2]], dtype=torch.long, device=mel.device)
            features = tower(mel[0], feature_lens=lengths).last_hidden_state
            if not isinstance(features, torch.Tensor):
                raise TypeError("Native audio tower must return a tensor")
            return features.unsqueeze(0)
    finally:
        for handle in handles:
            handle.remove()
