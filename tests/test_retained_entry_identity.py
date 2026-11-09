import subprocess

import numpy as np
import pytest

import share_weights
from src.onnx_artifacts import copy_artifact_entry, preflight_artifact_mutations
from tests.artifact_fixtures import _ort_result, _write_external_identity, _write_model
from tests.converter_fixtures import bytes_under, command, input_for


def _case_alias(path):
    alias = path.with_name(path.name.upper())
    if not alias.exists() or not alias.samefile(path):
        pytest.skip("Requires filesystem case aliases")
    return alias.name


def test_actual_nbits_sidecar_case_alias_preserves_retained_tensor(tmp_path):
    source = input_for(tmp_path, "quantize_nbits.py")
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        weight = np.arange(128 * 128, dtype=np.float32).reshape(128, 128) / (128 * 128)
        _write_model(source, name, weight=weight)
    target = tmp_path / "target"
    target.mkdir()
    data = target / "decoder_init.int4.onnx.data"
    data.write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", _case_alias(data))
    before_source, before_target = bytes_under(source), bytes_under(target)
    for _ in range(2):
        result = subprocess.run(
            command(tmp_path, "quantize_nbits.py", source, target), capture_output=True, text=True, timeout=90
        )
        np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])
        assert bytes_under(target) == before_target
        assert bytes_under(source) == before_source
        assert result.returncode != 0, result.stdout + result.stderr
        assert "retained graph" in result.stdout + result.stderr


@pytest.mark.parametrize("entry", ["case-alias", "hardlink"])
def test_sharing_preserves_tensor_using_selected_graph_bytes(tmp_path, entry):
    weight = np.full((64, 64), 42, np.float32)
    for name in ("decoder_init.onnx", "decoder_step.onnx"):
        _write_model(tmp_path, name, weight=weight)
    graph = tmp_path / "decoder_init.onnx"
    offset = graph.read_bytes().index(weight.tobytes()) + weight.nbytes - 4
    if entry == "case-alias":
        location = _case_alias(graph)
    else:
        retained = tmp_path / "retained.data"
        retained.hardlink_to(graph)
        location = retained.name
    _write_external_identity(tmp_path, "aux.onnx", location, offset)
    before = bytes_under(tmp_path)
    for _ in range(2):
        if entry == "case-alias":
            with pytest.raises(ValueError, match="destination conflict"):
                share_weights.share_external_models(tmp_path)
            assert bytes_under(tmp_path) == before
        else:
            assert share_weights.share_external_models(tmp_path)
            assert (tmp_path / location).read_bytes() == before[next(p for p in before if p.name == location)]
        np.testing.assert_array_equal(_ort_result(tmp_path / "aux.onnx"), [42])


@pytest.mark.parametrize("mutation", ["generated", "rewritten", "removed"])
def test_retained_tensor_case_alias_rejects_each_mutation(tmp_path, mutation):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    payload = target / "payload.data"
    payload.write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", _case_alias(payload))
    before = bytes_under(target)
    with pytest.raises(ValueError, match="retained graph"):
        preflight_artifact_mutations(source, target, (), **{mutation: [payload.name]})
    assert bytes_under(target) == before
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("value", [42, 99])
def test_retained_tensor_case_alias_checks_actual_copy_bytes(tmp_path, prepared, value):
    source, target, assets = (tmp_path / name for name in ("source", "target", "assets"))
    for root in (source, target, assets):
        root.mkdir()
    payload = target / "payload.data"
    payload.write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", _case_alias(payload))
    owner = assets if prepared else source
    (owner / payload.name).write_bytes(np.array([value], np.float32).tobytes())
    kwargs = {"copied_assets": [payload.name], "asset_source": assets} if prepared else {}
    copies = () if prepared else [payload.name]
    if value == 99:
        with pytest.raises(ValueError, match="retained graph"):
            preflight_artifact_mutations(source, target, copies, **kwargs)
    else:
        assert preflight_artifact_mutations(source, target, copies, **kwargs) == set()
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])


@pytest.mark.parametrize("entry", ["hardlink", "symlink"])
def test_distinct_copy_entry_preserves_retained_tensor(tmp_path, entry):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    retained = target / "retained.data"
    retained.write_bytes(np.array([42], np.float32).tobytes())
    destination = target / "payload.data"
    if entry == "hardlink":
        destination.hardlink_to(retained)
    else:
        destination.symlink_to(retained.name)
    (source / destination.name).write_bytes(np.array([99], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", retained.name)
    copies = preflight_artifact_mutations(source, target, [destination.name])
    assert copies == {destination.name}
    copy_artifact_entry(source, target, destination.name)
    assert destination.read_bytes() == np.array([99], np.float32).tobytes()
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])


def test_selected_graph_case_alias_is_not_retained(tmp_path):
    from src.onnx_artifacts import inventory_graphs, retained_quantization

    source = tmp_path / "source"
    source.mkdir()
    graph = tmp_path / "decoder_init.int4.onnx"
    _write_model(tmp_path, graph.name)
    selected = _case_alias(graph)
    assert inventory_graphs(tmp_path, [selected]) == {}
    assert retained_quantization(tmp_path, ".int4", [selected], existing={graph.name: set()}) == (set(), {})
    assert (
        preflight_artifact_mutations(source, tmp_path, (), generated=[selected], existing={graph.name: set()}) == set()
    )


def test_unrelated_copy_does_not_hash_retained_tensor_data(tmp_path, monkeypatch):
    from src import onnx_artifacts

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "unrelated.data").write_bytes(b"ordinary copy")
    (target / "retained.data").write_bytes(np.array([42], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", "retained.data")

    def unexpected_digest(path):
        pytest.fail(f"Unrelated copy caused digest read: {path}")

    monkeypatch.setattr(onnx_artifacts, "file_digest", unexpected_digest)
    assert preflight_artifact_mutations(source, target, ["unrelated.data"]) == {"unrelated.data"}


def test_differently_cased_distinct_copy_preserves_retained_tensor(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    retained = target / "PAYLOAD.DATA"
    retained.write_bytes(np.array([42], np.float32).tobytes())
    destination = target / "payload.data"
    if destination.exists():
        pytest.skip("Requires distinct filesystem entries for differing case")
    (source / destination.name).write_bytes(np.array([99], np.float32).tobytes())
    _write_external_identity(target, "aux.onnx", retained.name)
    assert preflight_artifact_mutations(source, target, [destination.name]) == {destination.name}
    copy_artifact_entry(source, target, destination.name)
    np.testing.assert_array_equal(_ort_result(target / "aux.onnx"), [42])
