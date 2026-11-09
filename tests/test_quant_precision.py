import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.quantization_fixtures import source

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("algo", ["rtn", "gptq"])
def test_eight_bits_rejected_before_output(tmp_path, algo):
    inputs, calib = source(tmp_path)
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--bits",
            "8",
            "--block-size",
            "32",
            "--accuracy-level",
            "4",
            "--algo",
            algo,
            "--calib-data",
            str(calib),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    assert not output.exists(), result.stdout


@pytest.mark.parametrize("algo", ["rtn", "gptq"])
def test_four_bit_cli_actual_nodes_and_metadata(tmp_path, algo):
    inputs, calib = source(tmp_path)
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--bits",
            "4",
            "--block-size",
            "32",
            "--accuracy-level",
            "4",
            "--algo",
            algo,
            "--calib-data",
            str(calib),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    for name in ["decoder_init", "decoder_step"]:
        model = onnx.load(output / f"{name}.int4.onnx")
        nodes = [n for n in model.graph.node if n.op_type == "MatMulNBits"]
        assert len(nodes) == 2
        assert all({a.name: helper.get_attribute_value(a) for a in n.attribute}["bits"] == 4 for n in nodes)
        assert all({a.name: helper.get_attribute_value(a) for a in n.attribute}["block_size"] == 32 for n in nodes)
        session = ort.InferenceSession(str(output / f"{name}.int4.onnx"), providers=["CPUExecutionProvider"])
        got = session.run(None, {"x": np.ones((2, 64), dtype=np.float32)})[0]
        assert got.shape == (2, 32)
        assert np.isfinite(got).all()
    cfg = json.loads((output / "config.json").read_text())
    assert set(cfg["quantization"].values()) == {f"int4_{algo}_block32_al4"}


def cli(inputs, output, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--bits",
            "4",
            "--block-size",
            "32",
            *extra,
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("algo", ["rtn", "gptq"])
def test_unsupported_request_preserves_existing_output(tmp_path, algo):
    inputs, _ = source(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "config.json"
    sentinel.write_text("previous usable revision")
    result = cli(inputs, output, "--bits", "8", "--algo", algo)
    assert result.returncode != 0
    assert "only 4-bit" in result.stderr
    assert sentinel.read_text() == "previous usable revision"
    assert list(output.iterdir()) == [sentinel]


def test_rtn_excludes_one_node_without_lying_about_others(tmp_path):
    inputs, _ = source(tmp_path)
    output = tmp_path / "out"
    result = cli(inputs, output, "--nodes-to-exclude", "projection1")
    assert result.returncode == 0, result.stderr
    model = onnx.load(output / "decoder_init.int4.onnx")
    assert any(n.name == "projection1" and n.op_type == "MatMul" for n in model.graph.node)
    assert len([n for n in model.graph.node if n.op_type == "MatMulNBits"]) == 1


def test_gptq_exclusions_are_rejected_explicitly(tmp_path):
    inputs, calib = source(tmp_path)
    output = tmp_path / "out"
    result = cli(inputs, output, "--algo", "gptq", "--calib-data", str(calib), "--nodes-to-exclude", "projection1")
    assert result.returncode != 0
    assert "GPTQ node exclusions are unsupported" in result.stderr
    assert not output.exists()


def test_excluding_every_node_does_not_label_fp32_as_int4(tmp_path):
    inputs, _ = source(tmp_path)
    output = tmp_path / "out"
    result = cli(inputs, output, "--nodes-to-exclude", "projection1", "projection2")
    assert result.returncode != 0
    assert "No MatMulNBits" in result.stderr
    assert not list(output.glob("*.int4.onnx"))
    assert not (output / "config.json").exists()


@pytest.mark.parametrize("block_size", ["0", "8", "24"])
def test_bad_block_size_fails_before_output(tmp_path, block_size):
    inputs, _ = source(tmp_path)
    output = tmp_path / "out"
    result = cli(inputs, output, "--block-size", block_size)
    assert result.returncode != 0
    assert not output.exists()


def test_missing_decoder_fails_before_output(tmp_path):
    inputs, _ = source(tmp_path)
    (inputs / "decoder_step.onnx").unlink()
    output = tmp_path / "out"
    result = cli(inputs, output)
    assert result.returncode != 0
    assert not output.exists()


@pytest.fixture(params=["rtn", "gptq"])
def split_algorithm(request):
    return request.param


def _split_output(tmp_path, stem="decoder_init", algo="rtn"):
    inputs, calib = source(tmp_path)
    if stem != "decoder_init":
        shutil.copy2(inputs / "decoder_init.onnx", inputs / f"{stem}.onnx")
    baseline = cli(
        inputs, inputs, "--decoders", *sorted(path.stem for path in inputs.glob("*.onnx")), "--block-size", "16"
    )
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    output = tmp_path / "out"
    first = cli(inputs, output, "--decoders", stem, "--algo", algo, "--calib-data", str(calib))
    assert first.returncode == 0, first.stdout + first.stderr
    assert (output / f"{stem}.int4.onnx").read_bytes() != (inputs / f"{stem}.int4.onnx").read_bytes()
    return inputs, output


@pytest.mark.parametrize("stem", ["decoder_init", "encoder"])
def test_split_quantization_keeps_previously_generated_sibling(tmp_path, stem, split_algorithm):
    inputs, output = _split_output(tmp_path, stem, split_algorithm)
    generated = (output / f"{stem}.int4.onnx").read_bytes()
    second = cli(inputs, output, "--decoders", "decoder_step")
    assert second.returncode == 0, second.stdout + second.stderr
    assert (output / f"{stem}.int4.onnx").read_bytes() == generated
    metadata = json.loads((output / "config.json").read_text())["quantization"]
    assert metadata[stem] == f"int4_{split_algorithm}_block32"
    assert metadata["decoder_step"] == "int4_rtn_block32"
    model = onnx.load(output / f"{stem}.int4.onnx")
    nodes = [node for node in model.graph.node if node.op_type == "MatMulNBits"]
    assert len(nodes) == 2
    assert all(
        helper.get_attribute_value(next(attribute for attribute in node.attribute if attribute.name == "block_size"))
        == 32
        for node in nodes
    )


def _output_bytes(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _relocate_split_weights(output, name):
    from share_weights import external_data_files, model_tensors

    graph = output / "decoder_init.int4.onnx"
    dependencies = external_data_files(graph)
    assert len(dependencies) == 1
    destination = output / name
    dependencies.pop().rename(destination)
    model = onnx.load(graph, load_external_data=False)
    for tensor in model_tensors(model):
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = name
    graph.write_bytes(model.SerializeToString())
    return graph, destination


def test_split_source_closure_cannot_replace_retained_graph(tmp_path, split_algorithm):
    from tests.artifact_fixtures import _write_external_identity

    inputs, output = _split_output(tmp_path, algo=split_algorithm)
    _write_external_identity(inputs, "encoder.onnx", "decoder_init.int4.onnx")
    before = _output_bytes(output)
    result = cli(inputs, output, "--decoders", "decoder_step")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "retained graph" in result.stderr
    assert _output_bytes(output) == before


def test_split_source_copy_cannot_change_retained_weights(tmp_path, split_algorithm):
    inputs, output = _split_output(tmp_path, algo=split_algorithm)
    graph, data = _relocate_split_weights(output, "retained-weights.data")
    (inputs / data.name).write_bytes(bytes(data.stat().st_size))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    expected = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"]).run(
        None, {"x": np.ones((2, 64), dtype=np.float32)}
    )[0]
    before = _output_bytes(output)
    result = cli(inputs, output, "--decoders", "decoder_step")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "retained graph" in result.stderr
    assert _output_bytes(output) == before
    actual = ort.InferenceSession(str(graph), options, providers=["CPUExecutionProvider"]).run(
        None, {"x": np.ones((2, 64), dtype=np.float32)}
    )[0]
    np.testing.assert_array_equal(actual, expected)


def test_split_source_identical_retained_weights_are_usable(tmp_path, split_algorithm):
    inputs, output = _split_output(tmp_path, algo=split_algorithm)
    graph, data = _relocate_split_weights(output, "retained-weights.data")
    shutil.copy2(data, inputs / data.name)
    before = graph.read_bytes(), data.read_bytes()
    result = cli(inputs, output, "--decoders", "decoder_step")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (graph.read_bytes(), data.read_bytes()) == before


def test_split_generated_graph_cannot_replace_retained_tensor_storage(tmp_path, split_algorithm):
    inputs, output = _split_output(tmp_path, algo=split_algorithm)
    _relocate_split_weights(output, "decoder_step.int4.onnx")
    before = _output_bytes(output)
    result = cli(inputs, output, "--decoders", "decoder_step")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "retained graph" in result.stderr
    assert _output_bytes(output) == before


def test_split_tokenizer_removal_cannot_delete_retained_tensor_storage(tmp_path, split_algorithm):
    from tests.tokenizer_fixtures import save_tokenizer_assets

    inputs, output = _split_output(tmp_path, algo=split_algorithm)
    _relocate_split_weights(output, "chat_template.jinja")
    save_tokenizer_assets(inputs, template_format="embedded")
    before = _output_bytes(output)
    result = cli(inputs, output, "--decoders", "decoder_step")
    assert result.returncode != 0, result.stdout + result.stderr
    assert "retained graph" in result.stderr
    assert _output_bytes(output) == before


def test_verifies_every_effective_operator_before_saving(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import quantize_nbits

    inputs, _ = source(tmp_path)
    model = onnx.load(inputs / "decoder_init.onnx")
    del model.graph.node[:]
    model.graph.node.extend(
        [
            helper.make_node(
                "MatMulNBits", ["x", "w1"], ["h"], domain="com.microsoft", name="first", bits=4, block_size=32
            ),
            helper.make_node(
                "MatMulNBits", ["h", "w2"], ["y"], domain="com.microsoft", name="second", bits=8, block_size=32
            ),
        ]
    )
    saved = []

    class Quantizer:
        def __init__(self, **kwargs):
            self.model = SimpleNamespace(model=model, save_model_to_file=lambda *a, **k: saved.append(a))

        def process(self):
            pass

    monkeypatch.setattr(quantize_nbits, "MatMulNBitsQuantizer", Quantizer)
    with pytest.raises(ValueError, match="second"):
        quantize_nbits.quantize_decoder(
            str(inputs / "decoder_init.onnx"), str(tmp_path / "out.int4.onnx"), 4, 32, 4, "rtn", None
        )
    assert not saved
    assert not (tmp_path / "out.int4.onnx").exists()


def test_gptq_external_input_isolated_without_losing_unquantized_data(tmp_path):
    import quantize_nbits

    inputs, calib = source(tmp_path)
    path = inputs / "decoder_init.onnx"
    model = onnx.load(path)
    model.graph.node[-1].output[0] = "linear"
    model.graph.initializer.append(numpy_helper.from_array(np.full((32,), 0.25, dtype=np.float32), "bias"))
    model.graph.node.append(helper.make_node("Add", ["linear", "bias"], ["y"], name="bias_add"))
    (inputs / "weights").mkdir()
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights/decoder.data",
        size_threshold=0,
    )
    original = {p.relative_to(inputs).as_posix(): p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    x = np.ones((2, 64), dtype=np.float32)
    reference = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    output = tmp_path / "quantized" / "decoder.int4.onnx"
    quantize_nbits.quantize_decoder(str(path), str(output), 4, 32, 4, "gptq", str(calib))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"x": x})[0]
    assert actual.shape == reference.shape
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 0.3
    assert {p.relative_to(inputs).as_posix(): p.read_bytes() for p in inputs.rglob("*") if p.is_file()} == original
    assert not any(
        "qwen-gptq" in e.value
        for t in onnx.load(output, load_external_data=False).graph.initializer
        for e in t.external_data
    )


def test_gptq_failure_preserves_input_config_and_external_files(tmp_path, monkeypatch):
    import quantize_nbits

    inputs, calib = source(tmp_path)
    path = inputs / "decoder_init.onnx"
    onnx.save_model(
        onnx.load(path),
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.name + ".data",
        size_threshold=0,
    )
    original = {p.name: p.read_bytes() for p in inputs.iterdir() if p.is_file()}
    staged = []

    class Quantizer:
        def __init__(self, **kwargs):
            model_path = Path(kwargs["model"])
            assert model_path != path
            assert not (model_path.parent / "config.json").exists()
            assert model_path.exists()
            staged.append(model_path.parent)

        def process(self):
            raise RuntimeError("quantizer fault")

    monkeypatch.setattr(quantize_nbits, "MatMulNBitsQuantizer", Quantizer)
    with pytest.raises(RuntimeError, match="quantizer fault"):
        quantize_nbits.quantize_decoder(str(path), str(tmp_path / "result.int4.onnx"), 4, 32, 4, "gptq", str(calib))
    assert staged and all(not p.exists() for p in staged)
    assert {p.name: p.read_bytes() for p in inputs.iterdir() if p.is_file()} == original
    assert not (tmp_path / "result.int4.onnx").exists()


def test_quantization_replaces_stale_output_template(tmp_path):
    from src.prompt import build_prompt_ids, load_prompt_tokenizer
    from tests.tokenizer_fixtures import save_tokenizer_assets

    inputs, _ = source(tmp_path)
    save_tokenizer_assets(inputs, template_format="embedded")
    output = tmp_path / "out"
    output.mkdir()
    (output / "chat_template.jinja").write_text("stale template")
    result = cli(inputs, output)
    assert result.returncode == 0, result.stderr + result.stdout
    assert not (output / "chat_template.jinja").exists()
    assert build_prompt_ids(2, load_prompt_tokenizer(str(output), local_files_only=True)) == build_prompt_ids(
        2, load_prompt_tokenizer(str(inputs), local_files_only=True)
    )


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("name", ["encoder", "custom_projection"])
def test_quantize_safe_model_leaf_real_command(tmp_path, name, inplace):
    from tests.artifact_fixtures import _write_model

    source = tmp_path / "input"
    source.mkdir()
    _write_model(source, name + ".onnx", weight=np.arange(2048, dtype=np.float32).reshape(128, 16) / 2048)
    (source / "config.json").write_text("{}")
    _write_model(source, name + ".int4.onnx")
    if inplace:
        (source / (name + ".int4.onnx")).rename(source / (name + ".int4.onnx.bak"))
    output = source if inplace else tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--bits",
            "4",
            "--block-size",
            "64",
            "--accuracy-level",
            "4",
            "--decoders",
            name,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    graph = onnx.load(output / (name + ".int4.onnx"))
    nodes = [n for n in graph.graph.node if n.op_type == "MatMulNBits"]
    assert nodes
    assert json.loads((output / "config.json").read_text())["quantization"][name] == "int4_rtn_block64_al4"


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../outside/encoder", "/tmp/encoder", "sub/encoder", "sub\\encoder", "C:encoder"]
)
def test_unsafe_model_name_is_rejected_before_output(tmp_path, name):
    inputs, _ = source(tmp_path)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "quantize_nbits.py"),
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--bits",
            "4",
            "--block-size",
            "64",
            "--accuracy-level",
            "4",
            "--decoders",
            name,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Unsafe model name" in result.stderr
    assert not output.exists()


def test_failed_asset_publication_does_not_destroy_retained_quantization_provenance(tmp_path, monkeypatch):
    import quantize_nbits
    from src.tokenizer_assets import PreparedTokenizerAssets
    from tests.tokenizer_fixtures import save_tokenizer_assets

    inputs, output = _split_output(tmp_path)
    save_tokenizer_assets(inputs, "embedded")
    prior = (output / "config.json").read_bytes()
    graph = (output / "decoder_init.int4.onnx").read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize_nbits.py",
            "--input",
            str(inputs),
            "--output",
            str(output),
            "--bits",
            "4",
            "--block-size",
            "32",
            "--decoders",
            "decoder_step",
        ],
    )

    def fail_after_ordinary_copies(*args, **kwargs):
        raise OSError("asset publication failed")

    with monkeypatch.context() as publication:
        publication.setattr(PreparedTokenizerAssets, "publish", fail_after_ordinary_copies)
        with pytest.raises(OSError, match="asset publication failed"):
            quantize_nbits.main()
    assert (output / "config.json").read_bytes() == prior
    assert (output / "decoder_init.int4.onnx").read_bytes() == graph
    quantize_nbits.main()
    tags = json.loads((output / "config.json").read_text())["quantization"]
    assert tags["decoder_init"] == "int4_rtn_block32"
    assert tags["decoder_step"] == "int4_rtn_block32"
    assert (output / "decoder_init.int4.onnx").read_bytes() == graph


@pytest.mark.parametrize("function_bits", [4, 8])
def test_precision_guard_checks_local_function_operators(function_bits):
    from quantize_nbits import verify_quantized_precision

    def operator(bits):
        return helper.make_node(
            "MatMulNBits", ["x", "w", "scales"], ["y"], domain="com.microsoft", bits=bits, block_size=32
        )

    function = helper.make_function(
        "local",
        "Quantized",
        ["x", "w", "scales"],
        ["y"],
        [operator(function_bits)],
        [helper.make_opsetid("com.microsoft", 1)],
    )
    graph = helper.make_graph(
        [operator(4), helper.make_node("Quantized", ["x", "w", "scales"], ["z"], domain="local")], "root", [], []
    )
    model = helper.make_model(graph, functions=[function])
    if function_bits == 4:
        verify_quantized_precision(model, 4, 32)
    else:
        with pytest.raises(ValueError, match="bits=8"):
            verify_quantized_precision(model, 4, 32)


def test_unused_function_cannot_authorize_quantized_output(tmp_path):
    from quantize_nbits import quantize_decoder

    function = helper.make_function(
        "local",
        "Unused",
        ["x", "w", "scales"],
        ["y"],
        [helper.make_node("MatMulNBits", ["x", "w", "scales"], ["y"], domain="com.microsoft", bits=4, block_size=32)],
        [helper.make_opsetid("com.microsoft", 1)],
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "dynamic",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [32, 32]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
    )
    path, output = tmp_path / "dynamic.onnx", tmp_path / "dynamic.int4.onnx"
    onnx.save(
        helper.make_model(graph, functions=[function], opset_imports=[helper.make_opsetid("", 18)], ir_version=10), path
    )
    output.write_bytes(b"prior output")
    with pytest.raises(ValueError, match="No MatMulNBits"):
        quantize_decoder(str(path), str(output), 4, 32, None, "rtn", None)
    assert output.read_bytes() == b"prior output"


@pytest.mark.parametrize("placement", ["transitive", "nested"])
@pytest.mark.parametrize("bits", [4, 8])
def test_precision_follows_called_functions_and_nested_graphs(placement, bits):
    from quantize_nbits import verify_quantized_precision

    operator = helper.make_node(
        "MatMulNBits", ["x", "w", "scales"], ["y"], domain="com.microsoft", bits=bits, block_size=32
    )
    leaf = helper.make_function(
        "local", "Leaf", ["x", "w", "scales"], ["y"], [operator], [helper.make_opsetid("com.microsoft", 1)]
    )
    call = helper.make_node("Leaf", ["x", "w", "scales"], ["y"], domain="local")
    if placement == "transitive":
        parent = helper.make_function(
            "local", "Parent", ["x", "w", "scales"], ["y"], [call], [helper.make_opsetid("local", 1)]
        )
        nodes = [helper.make_node("Parent", ["x", "w", "scales"], ["y"], domain="local")]
        functions = [parent, leaf]
    else:
        branch = helper.make_graph([call], "branch", [], [])
        nodes = [helper.make_node("If", ["condition"], ["y"], then_branch=branch, else_branch=branch)]
        functions = [leaf]
    model = helper.make_model(helper.make_graph(nodes, "root", [], []), functions=functions)
    if bits == 4:
        verify_quantized_precision(model, 4, 32)
    else:
        with pytest.raises(ValueError, match="bits=8"):
            verify_quantized_precision(model, 4, 32)


def test_precision_rejects_called_function_attribute_indirection():
    from quantize_nbits import verify_quantized_precision

    operator = helper.make_node("MatMulNBits", ["x", "w", "scales"], ["y"], domain="com.microsoft", block_size=32)
    operator.attribute.append(
        helper.make_attribute_ref("bits", onnx.AttributeProto.INT, ref_attr_name="requested_bits")
    )
    function = helper.make_function(
        "local",
        "Indirect",
        ["x", "w", "scales"],
        ["y"],
        [operator],
        [helper.make_opsetid("com.microsoft", 1)],
        attributes=["requested_bits"],
    )
    call = helper.make_node("Indirect", ["x", "w", "scales"], ["y"], domain="local", requested_bits=4)
    model = helper.make_model(helper.make_graph([call], "root", [], []), functions=[function])
    with pytest.raises(ValueError, match="reference attribute"):
        verify_quantized_precision(model, 4, 32)


def _function_graph_argument_model(active_bits=None, argument_bits=4):
    def operator(bits):
        return helper.make_node(
            "MatMulNBits", ["x", "w", "s"], ["y"], domain="com.microsoft", bits=bits, block_size=32, K=32, N=32
        )

    body = [operator(active_bits)] if active_bits is not None else [helper.make_node("Identity", ["x"], ["y"])]
    function = helper.make_function(
        "local",
        "Projection",
        ["x", "w", "s"],
        ["y"],
        body,
        [helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        attributes=["unused"],
    )
    argument = helper.make_graph(
        [operator(argument_bits)], "argument", [], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])]
    )
    graph = helper.make_graph(
        [helper.make_node("Projection", ["x", "w", "s"], ["y"], domain="local", name="projection", unused=argument)],
        "root",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])],
        [
            numpy_helper.from_array(np.zeros((32, 1, 16), np.uint8), "w"),
            numpy_helper.from_array(np.ones((32, 1), np.float32), "s"),
        ],
    )
    return helper.make_model(
        graph,
        functions=[function],
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("local", 1),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )


def test_unused_local_function_graph_argument_cannot_authorize_output(tmp_path):
    from quantize_nbits import quantize_decoder

    path, output = tmp_path / "plain.onnx", tmp_path / "plain.int4.onnx"
    model = _function_graph_argument_model()
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = np.arange(32, dtype=np.float32).reshape(1, 32)
    np.testing.assert_array_equal(session.run(None, {"x": inputs})[0], inputs)
    output.write_bytes(b"prior output")
    with pytest.raises(ValueError, match="No MatMulNBits"):
        quantize_decoder(str(path), str(output), 4, 32, None, "rtn", None)
    assert output.read_bytes() == b"prior output"


def test_unused_graph_argument_does_not_override_called_precision(tmp_path):
    from quantize_nbits import quantize_decoder

    path, output = tmp_path / "called.onnx", tmp_path / "called.int4.onnx"
    onnx.save(_function_graph_argument_model(active_bits=4, argument_bits=8), path)
    onnx.checker.check_model(str(path))
    quantize_decoder(str(path), str(output), 4, 32, None, "rtn", None)
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert np.isfinite(session.run(None, {"x": np.ones((1, 32), np.float32)})[0]).all()


def test_quantization_preserves_portable_rms_scalar_attributes(tmp_path):
    from optimize_graphs import RMS_DOMAIN, _rms_function
    from quantize_nbits import quantize_decoder

    model = _function_graph_argument_model(active_bits=4)
    model.graph.node[0].output[0] = "projected"
    model.graph.node.append(
        helper.make_node("RMSNormalization", ["projected", "gain"], ["y"], domain=RMS_DOMAIN, name="rms", epsilon=1e-6)
    )
    model.graph.initializer.append(numpy_helper.from_array(np.ones(32, np.float32), "gain"))
    model.functions.append(_rms_function(18))
    model.opset_import.append(helper.make_opsetid(RMS_DOMAIN, 1))
    path, output = tmp_path / "rms.onnx", tmp_path / "rms.int4.onnx"
    onnx.save(model, path)
    onnx.checker.check_model(str(path))
    original = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {"x": np.ones((1, 32), np.float32)}
    expected = original.run(None, inputs)[0]
    quantize_decoder(str(path), str(output), 4, 32, None, "rtn", None)
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, inputs)[0], expected)


def test_precision_rejects_unresolved_executable_graph_attribute():
    from quantize_nbits import verify_quantized_precision

    branch = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "branch", [], [])
    conditional = helper.make_node("If", ["condition"], ["y"], else_branch=branch)
    conditional.attribute.append(
        helper.make_attribute_ref("then_branch", onnx.AttributeProto.GRAPH, ref_attr_name="body")
    )
    function = helper.make_function(
        "local", "Branch", ["x", "condition"], ["y"], [conditional], [helper.make_opsetid("", 18)], attributes=["body"]
    )
    call = helper.make_node("Branch", ["x", "condition"], ["y"], domain="local", body=branch)
    model = helper.make_model(helper.make_graph([call], "root", [], []), functions=[function])
    with pytest.raises(ValueError, match="indirection is unsupported"):
        verify_quantized_precision(model, 4, 32)


def test_precision_uses_called_function_overload():
    from quantize_nbits import verify_quantized_precision

    functions = []
    for bits in (4, 8):
        operator = helper.make_node(
            "MatMulNBits", ["x", "w", "s"], ["y"], domain="com.microsoft", bits=bits, block_size=32
        )
        function = helper.make_function(
            "local", "Projection", ["x", "w", "s"], ["y"], [operator], [helper.make_opsetid("com.microsoft", 1)]
        )
        function.overload = str(bits)
        functions.append(function)
    call = helper.make_node("Projection", ["x", "w", "s"], ["y"], domain="local")
    call.overload = "4"
    model = helper.make_model(helper.make_graph([call], "root", [], []), functions=functions)
    verify_quantized_precision(model, 4, 32)
    model.graph.node[0].overload = "8"
    with pytest.raises(ValueError, match="bits=8"):
        verify_quantized_precision(model, 4, 32)


def test_gptq_completed_output_survives_input_staging_cleanup_error(tmp_path, monkeypatch, capsys):
    import quantize_nbits
    from src import artifact_publication

    inputs, calib = source(tmp_path)
    before = {path.name: path.read_bytes() for path in inputs.iterdir()}
    output = tmp_path / "result.int4.onnx"
    cleanup = artifact_publication.shutil.rmtree
    leftovers = []

    def fail_cleanup(path, *args, **kwargs):
        path = Path(path)
        if path.name.startswith("qwen-gptq-"):
            leftovers.append(path)
            raise PermissionError("GPTQ staging cleanup failed")
        return cleanup(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(artifact_publication.shutil, "rmtree", fail_cleanup)
        quantize_nbits.quantize_decoder(str(inputs / "decoder_init.onnx"), str(output), 4, 32, 4, "gptq", str(calib))
    model = onnx.load(output)
    quantize_nbits.verify_quantized_precision(model, 4, 32)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert np.isfinite(session.run(None, {"x": np.ones((2, 64), dtype=np.float32)})[0]).all()
    assert {path.name: path.read_bytes() for path in inputs.iterdir()} == before
    log = capsys.readouterr().out
    assert leftovers and all(str(path) in log for path in leftovers)
    assert "Publication completed; staging cleanup" in log
    for path in leftovers:
        cleanup(path)
