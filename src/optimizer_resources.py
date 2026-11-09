import gc
import inspect

import onnxruntime
from onnxruntime.transformers import optimizer

MODEL_TYPE = "qwen3_asr_resource_gpt2"


def _supported_base():
    if onnxruntime.__version__ != "1.31.0":
        raise RuntimeError("Optimizer resource management requires ONNX Runtime 1.31.0")
    base = getattr(optimizer, "Gpt2OnnxModel", None)
    model_types = getattr(optimizer, "MODEL_TYPES", None)
    if base is None or not isinstance(model_types, dict) or model_types.get("gpt2") != (base, "pytorch", 1):
        raise RuntimeError("Unsupported ONNX Runtime GPT-2 optimizer factory")
    try:
        parameters = list(inspect.signature(base.infer_runtime_shape).parameters.values())
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Unsupported ONNX Runtime shape inference API") from error
    if (
        [parameter.name for parameter in parameters] != ["self", "dynamic_axis_mapping", "update"]
        or any(parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters)
        or not isinstance(parameters[1].default, dict)
        or parameters[2].default is not False
    ):
        raise RuntimeError("Unsupported ONNX Runtime shape inference API")
    return base


_BASE = _supported_base()
_DEFAULT_MAPPING = inspect.signature(_BASE.infer_runtime_shape).parameters["dynamic_axis_mapping"].default


class ResourceGpt2OnnxModel(optimizer.Gpt2OnnxModel):
    def infer_runtime_shape(self, dynamic_axis_mapping=_DEFAULT_MAPPING, update=False):
        if self.enable_shape_infer and update and self.shape_infer_helper is not None:
            self.shape_infer_helper = None
            gc.collect()
        return super().infer_runtime_shape(dynamic_axis_mapping, update)


def resource_model_type():
    if _supported_base() is not _BASE:
        raise RuntimeError("ONNX Runtime GPT-2 optimizer changed after initialization")
    expected = (ResourceGpt2OnnxModel, "pytorch", 1)
    if optimizer.MODEL_TYPES.setdefault(MODEL_TYPE, expected) != expected:
        raise RuntimeError("Optimizer resource factory key is already owned by another implementation")
    return MODEL_TYPE
