import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.artifact_sources import paths_overlap, validate_publication_sources


class SourceAccessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_distinct_regular_file_replacement_is_allowed(self):
        source, target = self.root / "source", self.root / "target"
        source.write_bytes(b"source")
        target.write_bytes(b"previous")
        validate_publication_sources([source], files=[target])
        self.assertEqual(source.read_bytes(), b"source")

    def test_direct_rewrite_requires_explicit_entry_authority(self):
        source = self.root / "source"
        source.write_bytes(b"source")
        with self.assertRaisesRegex(ValueError, "source access path"):
            validate_publication_sources([source], files=[source])
        validate_publication_sources([source], files=[source], direct_rewrites=[source])

    @unittest.skipIf(os.name == "nt", "Uses POSIX symlinks; junction tests cover Windows")
    def test_direct_rewrite_does_not_authorize_another_readers_backing_entry(self):
        source, reader = self.root / "source", self.root / "reader"
        source.write_bytes(b"source")
        reader.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "source access path"):
            validate_publication_sources([source, reader], files=[source], direct_rewrites=[source])
        with self.assertRaisesRegex(ValueError, "source access path"):
            validate_publication_sources([reader], files=[source, reader], direct_rewrites=[reader])
        validate_publication_sources([reader], files=[reader], direct_rewrites=[reader])

    def test_replacing_distinct_hardlink_entry_is_allowed(self):
        source, target = self.root / "source", self.root / "target"
        source.write_bytes(b"source")
        target.hardlink_to(source)
        validate_publication_sources([source], files=[target])
        replacement = self.root / "replacement"
        replacement.write_bytes(b"new")
        os.replace(replacement, target)
        self.assertEqual(source.read_bytes(), b"source")

    def test_existing_case_alias_follows_actual_filesystem_identity(self):
        source, target = self.root / "source", self.root / "SOURCE"
        source.write_bytes(b"source")
        if target.exists():
            with self.assertRaisesRegex(ValueError, "source access path"):
                validate_publication_sources([source], files=[target])
        else:
            target.write_bytes(b"different file")
            validate_publication_sources([source], files=[target])

    def test_directory_alias_follows_actual_filesystem_identity(self):
        source, alias = self.root / "source", self.root / "SOURCE"
        source.mkdir()
        self.assertEqual(paths_overlap(source, alias), alias.exists())

    def test_uncreated_parent_dotdot_overlap_is_retained(self):
        source = self.root / "source"
        source.mkdir()
        self.assertTrue(paths_overlap(source, self.root / "uncreated" / ".." / "source"))

    @unittest.skipIf(os.name == "nt", "Uses POSIX symlinks; junction tests cover Windows")
    def test_intermediate_symlink_dependency_is_protected(self):
        safe, output = self.root / "safe", self.root / "output"
        safe.mkdir()
        output.mkdir()
        (safe / "data").write_bytes(b"source")
        (output / "bridge").symlink_to(safe, target_is_directory=True)
        source = self.root / "source"
        source.symlink_to(output / "bridge", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "source access path"):
            validate_publication_sources([source / "data"], directories=[output])
        self.assertEqual((source / "data").read_bytes(), b"source")

    @unittest.skipIf(os.name == "nt", "Uses POSIX symlinks; junction tests cover Windows")
    def test_safe_symlink_read_and_destination_entry_replacement(self):
        source, linked, target = self.root / "source", self.root / "linked", self.root / "target"
        source.write_bytes(b"source")
        linked.symlink_to(source)
        target.symlink_to(source)
        validate_publication_sources([linked], files=[target])
        replacement = self.root / "replacement"
        replacement.write_bytes(b"new")
        os.replace(replacement, target)
        self.assertEqual(source.read_bytes(), b"source")

    def junction(self, link, target):
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(link.is_junction())
        self.assertFalse(link.is_symlink())

    @unittest.skipUnless(os.name == "nt", "Requires real Windows junctions")
    def test_windows_direct_junction_read_is_rejected_before_publication(self):
        safe, output = self.root / "safe", self.root / "output"
        safe.mkdir()
        output.mkdir()
        (safe / "data").write_bytes(b"source")
        source = self.root / "source"
        self.junction(source, safe)
        with self.assertRaisesRegex(ValueError, "Junction in source access path"):
            validate_publication_sources([source / "data"], directories=[output])
        self.assertEqual((source / "data").read_bytes(), b"source")
        self.assertEqual(list(output.iterdir()), [])

    @unittest.skipUnless(os.name == "nt", "Requires real Windows junctions")
    def test_windows_chained_junction_read_is_rejected_before_publication(self):
        safe, output = self.root / "safe", self.root / "output"
        safe.mkdir()
        output.mkdir()
        (safe / "data").write_bytes(b"source")
        bridge, source = output / "bridge", self.root / "source"
        self.junction(bridge, safe)
        self.junction(source, bridge)
        with self.assertRaisesRegex(ValueError, "Junction in source access path"):
            validate_publication_sources([source / "data"], directories=[output])
        self.assertEqual((source / "data").read_bytes(), b"source")
        self.assertTrue(bridge.is_junction())


if __name__ == "__main__":
    unittest.main()
