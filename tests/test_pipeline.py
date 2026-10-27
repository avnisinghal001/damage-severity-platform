"""Synthetic pipeline checks: artifacts, duplicate detection, failures, and provenance."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from damage_analytics.config import ArchiveLimits, Config
from damage_analytics.io import write_json

HAS_PLOTTING = all(importlib.util.find_spec(name) for name in ("PIL", "numpy", "matplotlib"))


@unittest.skipUnless(HAS_PLOTTING, "Pillow, NumPy and Matplotlib are not installed")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.config = Config(root / "raw", root / "interim", root / "outputs",
                             ArchiveLimits(), expected_pairs=2, gallery_size=2)
        for role in ("images", "annotations"):
            (self.config.extracted_dir / role).mkdir(parents=True)
        for name in ("first", "second"):
            image_path = self.config.extracted_dir / f"images/{name}.png"
            Image.new("RGB", (8, 6), "white").save(image_path)
            write_json(self.config.extracted_dir / f"annotations/{name}.json", {
                "imagePath": f"{name}.png", "imageWidth": 8, "imageHeight": 6,
                "shapes": [{"label": "Microcrack", "shape_type": "polygon",
                            "points": [[1, 1], [4, 1], [4, 4], [1, 4]]}],
            })

    def test_reports_masks_statistics_and_gallery(self):
        from damage_analytics.analysis import analyze_dataset
        from damage_analytics.validation import validate_dataset
        from damage_analytics.visualization import create_gallery

        records, validation = validate_dataset(self.config)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["duplicate_groups"], 2)  # byte and decoded pixels
        summary = analyze_dataset(self.config, records, validation)
        self.assertEqual(summary["total_pixels"], 96)
        self.assertEqual(sum(row["pixels"] for row in summary["classes"]), 96)
        self.assertEqual(summary["classes"][1]["pixels"], 32)
        self.assertEqual(summary["classes"][1]["images_with_annotation"], 2)
        self.assertEqual(create_gallery(self.config), create_gallery(self.config))
        for name in ("manifest.csv", "issues.csv", "validation.json", "summary.json",
                     "sample_gallery.png", "plots/class_pixels.png", "per_image.csv"):
            self.assertTrue((self.config.output_dir / name).is_file(), name)

    def test_invalid_annotation_is_excluded_and_reports_failure(self):
        from damage_analytics.validation import validate_dataset

        path = self.config.extracted_dir / "annotations/first.json"
        document = json.loads(path.read_text())
        document["shapes"][0]["label"] = "invented severity"
        write_json(path, document)
        records, validation = validate_dataset(self.config)
        self.assertEqual(validation["status"], "failed")
        self.assertEqual(len(records), 1)
        self.assertEqual(validation["errors"], 1)

    def test_corrupted_image_does_not_abort_other_pairs(self):
        from damage_analytics.validation import validate_dataset

        (self.config.extracted_dir / "images/first.png").write_bytes(b"not an image")
        records, validation = validate_dataset(self.config)
        self.assertEqual(len(records), 1)
        self.assertEqual(validation["errors"], 1)

    def test_gallery_refuses_stale_source_and_unexpected_mask_ids(self):
        from PIL import Image

        from damage_analytics.validation import validate_dataset
        from damage_analytics.visualization import colorize_mask, create_gallery

        validate_dataset(self.config)
        (self.config.extracted_dir / "annotations/first.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            create_gallery(self.config)
        with self.assertRaises(ValueError):
            colorize_mask(Image.new("L", (2, 2), 255))
