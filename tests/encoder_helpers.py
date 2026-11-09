from pathlib import Path

import torch

from src.onnx_export import export_onnx

FRAME_LENGTHS = (1, 2, 7, 8, 9, 49, 50, 51, 74, 75, 76, 99, 100, 101, 799, 800, 801, 1600)


def export_encoder_at_100(wrapper: torch.nn.Module, output_path: Path) -> None:
    generator = torch.Generator().manual_seed(100)
    with torch.no_grad():
        export_onnx(
            wrapper,
            (torch.randn(1, 128, 100, generator=generator),),
            str(output_path),
            input_names=["mel"],
            output_names=["audio_features"],
            dynamic_axes={"mel": {2: "time"}, "audio_features": {1: "enc_time"}},
            opset_version=17,
            do_constant_folding=True,
        )
