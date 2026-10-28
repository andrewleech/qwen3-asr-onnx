import pytest
from transformers import PreTrainedTokenizerFast

import export
from src.prompt import load_prompt_tokenizer
from tests.tokenizer_fixtures import save_tokenizer_assets


def _snapshot(root):
    return {
        path.relative_to(root): ("symlink", str(path.readlink())) if path.is_symlink() else ("file", path.read_bytes())
        for path in root.rglob("*")
        if path.is_symlink() or path.is_file()
    }


@pytest.mark.parametrize("chained", [False, True])
def test_export_refuses_legacy_output_backing_original_tokenizer(tmp_path, chained, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    expected = save_tokenizer_assets(source, "embedded").get_vocab()
    target.mkdir()
    retained = target / "chat_template.json"
    retained.write_bytes((source / "tokenizer.json").read_bytes())
    (source / "tokenizer.json").unlink()
    if chained:
        bridge = tmp_path / "bridge"
        bridge.symlink_to(retained)
        (source / "tokenizer.json").symlink_to(bridge)
    else:
        (source / "tokenizer.json").symlink_to(retained)
    assert load_prompt_tokenizer(str(source), local_files_only=True).get_vocab() == expected
    before = _snapshot(source), _snapshot(target)
    saved = []
    save_pretrained = PreTrainedTokenizerFast.save_pretrained

    def observe(tokenizer, *args, **kwargs):
        saved.append(args)
        return save_pretrained(tokenizer, *args, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerFast, "save_pretrained", observe)
    with pytest.raises(ValueError, match="clean output directory"):
        export.copy_tokenizer(str(source), str(target))
    assert not saved
    assert (_snapshot(source), _snapshot(target)) == before
    assert load_prompt_tokenizer(str(source), local_files_only=True).get_vocab() == expected


def test_export_accepts_cache_links_with_clean_legacy_output(tmp_path):
    cache, source, target = tmp_path / "cache", tmp_path / "source", tmp_path / "target"
    expected = save_tokenizer_assets(cache, "embedded").get_vocab()
    source.mkdir()
    for path in cache.iterdir():
        (source / path.name).symlink_to(path)
    before = _snapshot(source)
    export.copy_tokenizer(str(source), str(target))
    assert _snapshot(source) == before
    assert all(path.is_symlink() for path in source.iterdir())
    assert load_prompt_tokenizer(str(target), local_files_only=True).get_vocab() == expected
