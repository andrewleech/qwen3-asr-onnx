import os
import shutil
import tempfile
from pathlib import Path


class ArtifactStaging:
    def __init__(self, *, prefix, dir):
        self._temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=dir)
        self.path = Path(self._temporary.name)
        self._complete = False

    def mark_complete(self):
        self._complete = True

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            self._temporary.cleanup()
        except OSError as error:
            if not self._complete and exception_type is None:
                raise
            state = "Publication completed" if self._complete else "Operation failed"
            print(f"{state}; staging cleanup reported an error for {self.path}: {error}")


def publish_artifacts(replacements, removals=()):
    backup = Path(tempfile.mkdtemp(prefix=".release-backup-", dir=replacements[0][1].parent))
    saved = []
    installed = []
    clean = False
    committed = False
    try:
        targets = [target for _, target in replacements] + list(removals)
        for index, target in enumerate(targets):
            if target.exists() or target.is_symlink():
                old = backup / str(index)
                saved.append((old, target))
                os.replace(target, old)
        for source, target in replacements:
            installed.append(target)
            os.replace(source, target)
        clean = True
        committed = True
    except BaseException:
        try:
            for target in reversed(installed):
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
            for old, target in reversed(saved):
                if old.exists() or old.is_symlink():
                    os.replace(old, target)
            clean = True
        except BaseException as recovery_error:
            raise RuntimeError(
                f"Publication recovery was interrupted; inspect the destination files and backup directory: {backup}"
            ) from recovery_error
        raise
    finally:
        if clean:
            try:
                shutil.rmtree(backup)
            except BaseException as error:
                state = "Publication completed" if committed else "Original artifacts restored"
                print(f"{state}; backup cleanup reported an error for {backup}: {error}")
                if not isinstance(error, OSError):
                    raise
