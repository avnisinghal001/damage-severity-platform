"""Security and atomicity checks using tiny in-memory-created ZIP archives."""

import stat
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from damage_analytics.archives import checked_members, extract_archives
from damage_analytics.config import ArchiveLimits, Config


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive_path = self.root / "test.zip"

    def test_rejects_unsafe_member_paths(self):
        for name in ("../escape.jpg", "/absolute.jpg", "C:/drive.jpg", "x\\escape.jpg"):
            with self.subTest(name=name):
                with ZipFile(self.archive_path, "w") as archive:
                    archive.writestr(name, b"data")
                with ZipFile(self.archive_path) as archive, self.assertRaises(ValueError):
                    checked_members(archive, ArchiveLimits())

    def test_rejects_symlinks(self):
        entry = ZipInfo("link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with ZipFile(self.archive_path, "w") as archive:
            archive.writestr(entry, "../outside")
        with ZipFile(self.archive_path) as archive, self.assertRaises(ValueError):
            checked_members(archive, ArchiveLimits())

    def test_rejects_collisions_and_resource_limits(self):
        for names in (("a.jpg", "A.jpg"), ("folder", "folder/a.jpg")):
            with self.subTest(names=names):
                with ZipFile(self.archive_path, "w") as archive:
                    for name in names:
                        archive.writestr(name, b"data")
                with ZipFile(self.archive_path) as archive, self.assertRaises(ValueError):
                    checked_members(archive, ArchiveLimits())
        with ZipFile(self.archive_path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("large.txt", b"a" * 10000)
        for limits in (ArchiveLimits(max_member_bytes=10), ArchiveLimits(max_total_bytes=10),
                       ArchiveLimits(max_compression_ratio=2)):
            with self.subTest(limits=limits):
                with ZipFile(self.archive_path) as archive, self.assertRaises(ValueError):
                    checked_members(archive, limits)

    def test_failed_preflight_does_not_publish_or_modify_raw(self):
        raw = self.root / "raw"
        raw.mkdir()
        for name, member in (("imagedata.zip", "a.jpg"), ("labeldata.zip", "../escape.json")):
            with ZipFile(raw / name, "w") as archive:
                archive.writestr(member, b"source")
        before = {path.name: path.read_bytes() for path in raw.iterdir()}
        config = Config(raw, self.root / "extracted", self.root / "outputs", ArchiveLimits())
        with self.assertRaises(ValueError):
            extract_archives(config)
        self.assertFalse(config.extracted_dir.exists())
        self.assertEqual(before, {path.name: path.read_bytes() for path in raw.iterdir()})

    def test_success_preserves_paths_and_refuses_overwrite(self):
        raw = self.root / "raw"
        raw.mkdir()
        for name in ("imagedata.zip", "labeldata.zip"):
            with ZipFile(raw / name, "w") as archive:
                archive.writestr("wrapper/sample.txt", b"original")
        config = Config(raw, self.root / "extracted", self.root / "outputs", ArchiveLimits())
        extract_archives(config)
        self.assertEqual((config.extracted_dir / "images/wrapper/sample.txt").read_bytes(),
                         b"original")
        with self.assertRaises(FileExistsError):
            extract_archives(config)
