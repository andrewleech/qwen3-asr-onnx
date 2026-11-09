import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import quantize as dynamic
import quantize_static as static
from src import onnx_artifacts
from src.tokenizer_assets import PreparedTokenizerAssets
from tests.artifact_fixtures import _ort_result, _write_external_identity, _write_model
from tests.converter_fixtures import SCRIPTS, bytes_under, command, input_for


@pytest.mark.parametrize("script", SCRIPTS)
def test_internal_directory_alias_cannot_redirect_dependency_copy(tmp_path, script):
    source = input_for(tmp_path, script)
    (source / "alias").mkdir()
    np.array([99], dtype=np.float32).tofile(source / "alias/weights.data")
    _write_external_identity(source, "encoder.onnx", "alias/weights.data")
    target = tmp_path / "target"
    (target / "nested").mkdir(parents=True)
    np.array([42], dtype=np.float32).tofile(target / "nested/weights.data")
    (target / "alias").symlink_to("nested", target_is_directory=True)
    _write_external_identity(target, "aux.onnx", "nested/weights.data")
    before = bytes_under(target)
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "linked parent" in result.stdout + result.stderr
    assert bytes_under(target) == before
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])


@pytest.mark.parametrize("operation", ["copy", "json", "generated"])
def test_direct_writers_refuse_internal_linked_parents(tmp_path, operation):
    source, target = tmp_path / "source", tmp_path / "target"
    (source / "alias").mkdir(parents=True)
    (source / "alias/value").write_bytes(b"incoming")
    (target / "nested").mkdir(parents=True)
    (target / "nested/value").write_bytes(b"retained")
    (target / "alias").symlink_to("nested", target_is_directory=True)
    with pytest.raises(ValueError, match="linked parent"):
        if operation == "copy":
            onnx_artifacts.copy_artifact_entry(source, target, "alias/value")
        elif operation == "json":
            onnx_artifacts.write_artifact_json(target, "alias/value", {})
        else:
            onnx_artifacts.validate_generated_destinations(target, ["alias/value"])
    assert (target / "nested/value").read_bytes() == b"retained"


@pytest.mark.parametrize("script", SCRIPTS)
def test_nested_output_cannot_overwrite_selected_source_storage(tmp_path, script):
    source = input_for(tmp_path, script)
    target = source / "out"
    target.mkdir()
    stem = "decoder_init"
    generated = f"{stem}.int8.onnx.data" if script == "quantize.py" else f"{stem}.onnx"
    if script == "quantize_nbits.py":
        generated = f"{stem}.int4.onnx"
    weight = np.ones((64, 64) if script in {"quantize_nbits.py", "quantize.py"} else (4, 3), dtype=np.float32)
    _write_model(source, f"{stem}.onnx", external=True, data_name=f"out/{generated}", weight=weight)
    before = bytes_under(source)
    expected = _ort_result(source / f"{stem}.onnx")
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode != 0
    assert "source graph destination conflict" in result.stdout + result.stderr
    assert bytes_under(source) == before
    np.testing.assert_array_equal(_ort_result(source / f"{stem}.onnx"), expected)


@pytest.mark.parametrize("stem", ["encoder", "decoder_step"])
def test_nested_inline_output_preserves_selected_source_storage(tmp_path, stem):
    script = "quantize.py"
    source = input_for(tmp_path, script)
    target = source / "out"
    target.mkdir()
    storage = f"{stem}.int8.onnx.data"
    _write_model(
        source,
        f"{stem}.onnx",
        external=True,
        data_name=f"out/{storage}",
        weight=np.ones((4, 3), dtype=np.float32),
    )
    _write_external_identity(target, "aux.onnx", storage)
    before = bytes_under(source)
    expected = _ort_result(source / f"{stem}.onnx")
    retained = _ort_result(target / "aux.onnx")
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all((source / name).read_bytes() == value for name, value in before.items())
    assert not onnx_artifacts.external_data_files(target / f"{stem}.int8.onnx")
    np.testing.assert_array_equal(_ort_result(source / f"{stem}.onnx"), expected)
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), retained)
    np.testing.assert_allclose(_ort_result(target / f"{stem}.int8.onnx"), expected, atol=0.03, rtol=0.03)


@pytest.mark.parametrize("reader", ["complete-tokenizer", "embedded-tokenizer", "config", "graph"])
def test_inplace_conversion_preserves_original_indirect_read_routes(tmp_path, reader):
    from src.tokenizer_assets import TOKENIZER_FILES, validate_tokenizer_assets

    source = input_for(tmp_path, "quantize.py")
    name = "tokenizer.json"
    if reader == "complete-tokenizer":
        config = json.loads((source / "tokenizer_config.json").read_text())
        (source / "chat_template.json").write_text(json.dumps({"chat_template": config["chat_template"]}))
    elif reader == "embedded-tokenizer":
        for template in ("chat_template.json", "chat_template.jinja"):
            (source / template).unlink(missing_ok=True)
    elif reader == "config":
        for asset in TOKENIZER_FILES:
            (source / asset).unlink(missing_ok=True)
        name = "config.json"
    elif reader == "graph":
        name = "decoder_init.onnx"
    backing = source / "encoder.int8.onnx"
    backing.write_bytes((source / name).read_bytes())
    (source / name).unlink()
    (source / name).symlink_to(backing.name)
    before = bytes_under(source)
    expected = _ort_result(source / "decoder_init.onnx")
    for _ in range(2):
        result = subprocess.run(
            command(tmp_path, "quantize.py", source, source), capture_output=True, text=True, timeout=90
        )
        assert result.returncode != 0
        assert bytes_under(source) == before
        np.testing.assert_array_equal(_ort_result(source / "decoder_init.onnx"), expected)
        if "tokenizer" in reader:
            validate_tokenizer_assets(source)
        elif reader == "config":
            assert isinstance(json.loads((source / name).read_text()), dict)


@pytest.mark.parametrize("script", ["convert_fp16.py", "quantize_static.py", "quantize_nbits.py"])
def test_inline_converter_preserves_retained_potential_sidecar(tmp_path, script):
    import onnx

    source = input_for(tmp_path, script)
    if script == "quantize_nbits.py":
        for name in ("decoder_init.onnx", "decoder_step.onnx"):
            _write_model(source, name, weight=np.arange(256, dtype=np.float32).reshape(64, 4) / 256)
            model = onnx.load(source / name)
            model.graph.node[0].name = "projection"
            onnx.save(model, source / name)
    target = tmp_path / "target"
    target.mkdir()
    graph = "decoder_init.int4.onnx" if script == "quantize_nbits.py" else "decoder_init.onnx"
    data = target / f"{graph}.data"
    data.write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", data.name)
    before_source = bytes_under(source)
    before_data, before_aux = data.read_bytes(), (target / "aux.onnx").read_bytes()
    result = subprocess.run(command(tmp_path, script, source, target), capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert bytes_under(source) == before_source
    assert data.read_bytes() == before_data
    assert (target / "aux.onnx").read_bytes() == before_aux
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])
    assert not onnx_artifacts.external_data_files(target / graph)
    reference = source / "decoder_init.onnx"
    if script == "quantize_nbits.py":
        import quantize_nbits

        reference = tmp_path / "reference.onnx"
        quantize_nbits.quantize_decoder(str(source / "decoder_init.onnx"), str(reference), 4, 32, None, "rtn", None)
    np.testing.assert_allclose(_ort_result(target / graph), _ort_result(reference), atol=0.3, rtol=0.05)


@pytest.mark.parametrize("script", SCRIPTS)
def test_noncolliding_nested_output_can_be_rebuilt(tmp_path, script):
    source = input_for(tmp_path, script)
    target = source / "out"
    before = bytes_under(source)
    args = command(tmp_path, script, source, target)
    for _ in range(2):
        result = subprocess.run(args, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert all((source / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize("kind", ["dynamic", "static"])
@pytest.mark.parametrize("storage_alias", [False, True])
def test_ort_inference_never_writes_in_source_namespace(tmp_path, kind, storage_alias):
    source = input_for(tmp_path, "quantize.py")
    stem = "encoder" if kind == "dynamic" else "decoder_init"
    inferred = f"{stem}-inferred.onnx"
    _write_model(
        source,
        f"{stem}.onnx",
        external=storage_alias,
        data_name=inferred if storage_alias else None,
        weight=np.ones((4, 3), dtype=np.float32),
    )
    if not storage_alias:
        _write_model(source, inferred, weight=np.ones((4, 3), dtype=np.float32))
    before = bytes_under(source)
    expected = _ort_result(source / f"{stem}.onnx")
    output = tmp_path / "quantized.onnx"
    if kind == "dynamic":
        dynamic.quantize_onnx_file(str(source / f"{stem}.onnx"), str(output))
    else:
        static.quantize_decoder(str(source / f"{stem}.onnx"), str(output), [{"x": np.ones((1, 4), dtype=np.float32)}])
    assert bytes_under(source) == before
    np.testing.assert_array_equal(_ort_result(source / f"{stem}.onnx"), expected)
    np.testing.assert_allclose(_ort_result(output), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kind", ["dynamic", "static"])
def test_ort_quantizers_use_graph_alias_directory_for_weights(tmp_path, kind):
    physical, source = tmp_path / "physical", tmp_path / "source"
    physical.mkdir()
    source.mkdir()
    _write_model(
        physical, "model.onnx", external=True, data_name="weights.data", weight=np.full((4, 3), 99, np.float32)
    )
    (source / "model.onnx").symlink_to(physical / "model.onnx")
    (source / "weights.data").write_bytes(np.full((4, 3), 42, np.float32).tobytes())
    expected = _ort_result(source / "model.onnx")
    before = bytes_under(source), bytes_under(physical)
    output = tmp_path / "quantized.onnx"
    if kind == "dynamic":
        dynamic.quantize_onnx_file(str(source / "model.onnx"), str(output))
    else:
        static.quantize_decoder(str(source / "model.onnx"), str(output), [{"x": np.ones((1, 4), dtype=np.float32)}])
    np.testing.assert_allclose(_ort_result(output), expected, atol=1e-4, rtol=1e-4)
    assert (bytes_under(source), bytes_under(physical)) == before


def test_dynamic_retry_keeps_both_attempts_in_owned_input_stage(tmp_path, monkeypatch):
    source = input_for(tmp_path, "quantize.py")
    _write_model(
        source,
        "encoder.onnx",
        external=True,
        data_name="encoder-inferred.onnx",
        weight=np.ones((4, 3), dtype=np.float32),
    )
    before = bytes_under(source)
    expected = _ort_result(source / "encoder.onnx")
    quantize = dynamic.quantize_dynamic
    seen = []

    def fail_once(input_path, *args, **kwargs):
        seen.append(Path(input_path))
        if len(seen) == 1:
            raise ValueError("injected direct quantization failure")
        return quantize(input_path, *args, **kwargs)

    monkeypatch.setattr(dynamic, "quantize_dynamic", fail_once)
    output = tmp_path / "quantized.onnx"
    dynamic.quantize_onnx_file(str(source / "encoder.onnx"), str(output))
    assert len(seen) == 2
    assert all(path.parent.name.startswith("qwen-onnx-") for path in seen)
    assert all(not path.parent.exists() for path in seen)
    assert bytes_under(source) == before
    np.testing.assert_allclose(_ort_result(output), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("retained", [False, True])
def test_dynamic_metadata_only_labels_generated_graphs(tmp_path, retained):
    source = input_for(tmp_path, "quantize.py")
    for name in ("encoder.onnx", "decoder_step.onnx"):
        (source / name).unlink()
    target = tmp_path / "target"
    target.mkdir()
    previous = {}
    if retained:
        for stem in ("encoder", "decoder_step"):
            _write_model(target, f"{stem}.int8.onnx", external=True, weight=np.ones((4, 3), dtype=np.float32))
        _write_external_identity(target, "aux.onnx", "encoder.int8.onnx.data")
        config = json.loads((source / "config.json").read_text())
        config["quantization"] = {"encoder": "int8_static", "decoder_step": "int8_static"}
        (target / "config.json").write_text(json.dumps(config))
        previous = {name: value for name, value in bytes_under(target).items() if name != Path("config.json")}
    result = subprocess.run(
        command(tmp_path, "quantize.py", source, target), capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert all((target / name).read_bytes() == value for name, value in previous.items())
    expected = {"decoder_init": "int8_dynamic"}
    if retained:
        expected.update(encoder="int8_static", decoder_step="int8_static")
        np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [1])
    assert json.loads((target / "config.json").read_text())["quantization"] == expected


def test_dynamic_late_asset_failure_keeps_retained_provenance(tmp_path, monkeypatch):
    source = input_for(tmp_path, "quantize.py")
    for name in ("encoder.onnx", "decoder_step.onnx"):
        (source / name).unlink()
    target = tmp_path / "target"
    target.mkdir()
    _write_model(target, "decoder_step.int8.onnx", weight=np.ones((4, 3), dtype=np.float32))
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = {"decoder_step": "int8_static"}
    (target / "config.json").write_text(json.dumps(config))
    previous = (target / "config.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["quantize.py", "--input", str(source), "--output", str(target)])
    with monkeypatch.context() as patch:

        def fail_assets(*_args, **_kwargs):
            raise ValueError("injected late asset failure")

        patch.setattr(PreparedTokenizerAssets, "publish", fail_assets)
        with pytest.raises(ValueError, match="late asset failure"):
            dynamic.main()
    assert (target / "config.json").read_bytes() == previous
    dynamic.main()
    assert json.loads((target / "config.json").read_text())["quantization"] == {
        "decoder_init": "int8_dynamic",
        "decoder_step": "int8_static",
    }
