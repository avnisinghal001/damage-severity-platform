"""Pairing never relies on listing order, case folding, or suffix guesses."""

import tempfile
import unittest
from pathlib import Path

from damage_analytics.discovery import discover_pairs


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "images").mkdir()
        (self.root / "annotations").mkdir()

    def touch(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def test_pairs_across_wrapper_folders_and_reports_orphans(self):
        for path in ("images/wrapper/a.jpg", "annotations/labels/a.json", "images/orphan.jpg"):
            self.touch(path)
        pairs, issues = discover_pairs(self.root)
        self.assertEqual([pair.sample_id for pair in pairs], ["a"])
        self.assertEqual(issues[0]["code"], "missing_pair")

    def test_repeated_stems_are_not_silently_overwritten(self):
        for path in ("images/first/a.jpg", "images/second/a.png", "annotations/a.json"):
            self.touch(path)
        pairs, issues = discover_pairs(self.root)
        self.assertFalse(pairs)
        self.assertEqual(issues[0]["code"], "ambiguous_stem")

    def test_source_raster_masks_are_explicitly_unsupported(self):
        self.touch("annotations/a.png")
        _, issues = discover_pairs(self.root)
        self.assertEqual(issues[0]["code"], "unsupported_file")

    def test_rejects_symlinks(self):
        self.touch("images/a.jpg")
        (self.root / "images/link.jpg").symlink_to(self.root / "images/a.jpg")
        with self.assertRaises(ValueError):
            discover_pairs(self.root)
