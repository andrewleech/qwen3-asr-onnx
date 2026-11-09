import os
from pathlib import Path

import pytest

import src.artifact_publication as publication
from src.artifact_publication import publish_artifacts


def _snapshot(root):
    state = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            value = ("link", os.readlink(path))
        elif path.is_file():
            value = ("file", path.read_bytes())
        else:
            value = ("directory",)
        state[path.relative_to(root)] = value
    return state


def _publication(tmp_path):
    published, stage = tmp_path / "published", tmp_path / "stage"
    published.mkdir()
    stage.mkdir()
    old_release, new_release = published / "release", stage / "release"
    old_release.mkdir()
    new_release.mkdir()
    (old_release / "model").write_bytes(b"previous model")
    (new_release / "model").write_bytes(b"new model")
    old_archive, new_archive = published / "release.tar.gz", stage / "release.tar.gz"
    old_archive.write_bytes(b"previous archive")
    new_archive.write_bytes(b"new archive")
    removed = published / "chat_template.jinja"
    removed.symlink_to(tmp_path / "missing-original.jinja")
    fresh = stage / "new.json"
    fresh.write_bytes(b"new asset")
    return published, [(new_release, old_release), (new_archive, old_archive), (fresh, published / fresh.name)], removed


@pytest.mark.parametrize(
    "boundary,when",
    [
        ("backup", "before"),
        ("backup", "after"),
        ("install", "before"),
        ("install", "after"),
        ("removed", "after"),
        ("fresh", "after"),
    ],
)
def test_interruption_restores_every_prior_directory_entry(tmp_path, monkeypatch, boundary, when):
    published, replacements, removed = _publication(tmp_path)
    before = _snapshot(published)
    trigger = {
        "backup": replacements[0][1],
        "install": replacements[0][0],
        "removed": removed,
        "fresh": replacements[2][0],
    }[boundary]
    replace = os.replace
    interrupted = False

    def interrupt(source, target):
        nonlocal interrupted
        selected = not interrupted and Path(source) == trigger
        if selected and when == "before":
            interrupted = True
            raise KeyboardInterrupt("injected publication interruption")
        result = replace(source, target)
        if selected and when == "after":
            interrupted = True
            raise KeyboardInterrupt("injected publication interruption")
        return result

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt, match="injected publication"):
        publish_artifacts(replacements, [removed])
    assert interrupted
    assert _snapshot(published) == before


def test_failed_rollback_keeps_previous_artifacts_and_reports_backup(tmp_path, monkeypatch):
    published, replacements, removed = _publication(tmp_path)
    replace = os.replace

    def fail(source, target):
        if Path(source) == replacements[0][0]:
            raise OSError("injected installation failure")
        if Path(source).parent.name.startswith(".release-backup-") and Path(target) == replacements[0][1]:
            raise OSError("injected rollback failure")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(RuntimeError, match="backup directory") as caught:
        publish_artifacts(replacements, [removed])
    backups = list(published.glob(".release-backup-*"))
    assert len(backups) == 1
    assert str(backups[0]) in str(caught.value)
    assert (backups[0] / "0" / "model").read_bytes() == b"previous model"


@pytest.mark.parametrize(
    "boundary,error_type", [("restore", OSError), ("restore", KeyboardInterrupt), ("cleanup", KeyboardInterrupt)]
)
def test_interrupted_recovery_reports_all_preserved_originals(tmp_path, monkeypatch, boundary, error_type):
    published, replacements, removed = _publication(tmp_path)
    original_link = os.readlink(removed)
    replace, rmtree = os.replace, publication.shutil.rmtree
    recovery_failed = False

    def fail_replace(source, target):
        nonlocal recovery_failed
        if Path(source) == replacements[0][0]:
            replace(source, target)
            raise OSError("injected initial publication failure")
        if boundary == "restore" and Path(source).parent.name.startswith(".release-backup-"):
            recovery_failed = True
            raise error_type("injected recovery failure")
        return replace(source, target)

    def fail_cleanup(path, *args, **kwargs):
        nonlocal recovery_failed
        if boundary == "cleanup" and Path(path) == replacements[0][1]:
            recovery_failed = True
            raise error_type("injected recovery failure")
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_replace)
    monkeypatch.setattr(publication.shutil, "rmtree", fail_cleanup)
    with pytest.raises(BaseException) as caught:
        publish_artifacts(replacements, [removed])
    assert recovery_failed
    assert isinstance(caught.value, RuntimeError)
    assert isinstance(caught.value.__cause__, error_type)
    backups = list(published.glob(".release-backup-*"))
    assert len(backups) == 1
    assert str(backups[0]) in str(caught.value)
    assert (backups[0] / "0" / "model").read_bytes() == b"previous model"
    assert (backups[0] / "1").read_bytes() == b"previous archive"
    assert os.readlink(backups[0] / "3") == original_link


def test_interruption_after_final_restore_reports_destinations_and_backup(tmp_path, monkeypatch):
    published, replacements, removed = _publication(tmp_path)
    original_link = os.readlink(removed)
    replace = os.replace
    interrupted = False

    def interrupt(source, target):
        nonlocal interrupted
        if Path(source) == replacements[0][0]:
            raise OSError("injected installation failure")
        result = replace(source, target)
        if Path(source).parent.name.startswith(".release-backup-") and Path(target) == replacements[0][1]:
            interrupted = True
            raise KeyboardInterrupt("interrupted after final restoration")
        return result

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(RuntimeError) as caught:
        publish_artifacts(replacements, [removed])
    assert interrupted
    assert (published / "release" / "model").read_bytes() == b"previous model"
    assert (published / "release.tar.gz").read_bytes() == b"previous archive"
    assert os.readlink(removed) == original_link
    backups = list(published.glob(".release-backup-*"))
    assert len(backups) == 1
    assert list(backups[0].iterdir()) == []
    assert str(backups[0]) in str(caught.value)
    assert "inspect the destination files and backup directory" in str(caught.value)
    assert "previous artifacts remain" not in str(caught.value)


def test_backup_cleanup_error_preserves_restored_failure_identity(tmp_path, monkeypatch, capsys):
    import src.artifact_publication as publication

    original, staged = tmp_path / "original", tmp_path / "staged"
    original.write_text("old")
    staged.write_text("new")
    replace = os.replace
    cleanup = publication.shutil.rmtree

    def fail_install(source, target):
        if Path(source) == staged:
            raise OSError("install failed")
        return replace(source, target)

    def fail_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".release-backup-"):
            raise PermissionError("cleanup failed")
        return cleanup(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_install)
    monkeypatch.setattr(publication.shutil, "rmtree", fail_cleanup)
    with pytest.raises(OSError, match="install failed"):
        publication.publish_artifacts([(staged, original)])
    assert original.read_text() == "old"
    output = capsys.readouterr().out
    assert "Original artifacts restored" in output
    backups = list(tmp_path.glob(".release-backup-*"))
    assert backups and str(backups[0]) in output


@pytest.mark.parametrize("state", ["completed", "primary-error", "incomplete"])
def test_staging_cleanup_preserves_publication_outcome(tmp_path, monkeypatch, capsys, state):
    from src.artifact_publication import ArtifactStaging

    staging = ArtifactStaging(prefix="stage-", dir=tmp_path)
    cleanup = publication.shutil.rmtree
    primary = ValueError("primary operation failure")

    def fail_cleanup(path, *args, **kwargs):
        if Path(path) == staging.path:
            raise PermissionError("staging cleanup failed")
        return cleanup(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(publication.shutil, "rmtree", fail_cleanup)
        if state == "completed":
            with staging:
                staging.mark_complete()
        elif state == "primary-error":
            with pytest.raises(ValueError) as failure, staging:
                raise primary
            assert failure.value is primary
        else:
            with pytest.raises(PermissionError, match="staging cleanup failed"), staging:
                pass
    if state != "incomplete":
        assert str(staging.path) in capsys.readouterr().out
    assert staging.path.is_dir()
    cleanup(staging.path)
