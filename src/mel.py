"""
Whisper-compatible mel spectrogram computation for validation.

Not exported to ONNX — this runs on the host side (Python/Rust).
Parameters are identical to OpenAI Whisper.
"""

import numpy as np
import torch
import torch.nn.functional as F


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Slaney-normalized mel filterbank, matching librosa.filters.mel(norm='slaney')."""
    import librosa

    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm="slaney")


def log_mel_spectrogram(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute log-mel spectrogram from raw audio waveform.

    Args:
        audio: 1-D float32 array, 16kHz mono
        sample_rate: Expected sample rate (for mel filterbank)
        n_fft: FFT window size
        hop_length: Hop between STFT frames
        n_mels: Number of mel bins
        fmin: Minimum frequency for mel scale
        fmax: Maximum frequency for mel scale
        device: torch device

    Returns:
        Tensor of shape [1, n_mels, time] (log-mel, float32)
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Compute mel filterbank
    mel_filters = _mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel_filters = torch.from_numpy(mel_filters).float().to(device)

    # STFT
    window = torch.hann_window(n_fft).to(device)
    audio_tensor = torch.from_numpy(audio).float().to(device)

    stft = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = stft.abs() ** 2

    # Apply mel filterbank
    mel_spec = mel_filters @ magnitudes  # [n_mels, time]

    # Log scale (clamp for numerical stability)
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.unsqueeze(0)  # [1, n_mels, time]
