"""Post-export ONNX graph fixups for runtime compatibility."""

import onnx


def fix_reshape_allowzero(onnx_path: str) -> int:
    """Remove allowzero=1 from Reshape nodes for DirectML compatibility.

    The torch ONNX exporter sets allowzero=1 on every Reshape it emits,
    but DirectML rejects this attribute. Since no shape tensor in our
    graphs contains a literal 0 dimension, the attribute is safe to strip.

    Loads only the graph protobuf (not external weight data), modifies
    Reshape attributes in-place, and overwrites the .onnx file. The
    .onnx.data weight file is untouched.

    Args:
        onnx_path: Path to the .onnx file to fix.

    Returns:
        Number of Reshape nodes that had allowzero=1 removed.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    count = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        for attr in list(node.attribute):
            if attr.name == "allowzero" and attr.i == 1:
                node.attribute.remove(attr)
                count += 1
    if count > 0:
        onnx.save(model, onnx_path)
    return count
