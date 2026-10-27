"""Strict annotation checks independent of the full dataset."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from damage_analytics.annotations import Polygon, rasterize, read_annotation

HAS_IMAGES = all(importlib.util.find_spec(name) for name in ("numpy", "PIL"))


def annotation(label="Microcrack"):
    """Return a tiny source-format document, not a severity annotation."""
    return {"imagePath": "sample.png", "imageWidth": 8, "imageHeight": 6,
            "shapes": [{"label": label, "shape_type": "polygon",
                        "points": [[1, 1], [4, 1], [4, 4], [1, 4]]}]}


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "sample.json"

    def read(self, document):
        self.path.write_text(json.dumps(document), encoding="utf-8")
        return read_annotation(self.path, "sample.png", (8, 6))

    def test_accepts_source_label_verbatim(self):
        polygons = self.read(annotation())
        self.assertEqual(polygons[0].label, "Microcrack")
        self.assertEqual(len(polygons[0].points), 4)

    def test_rejects_unknown_labels_and_non_polygons(self):
        for field, value in (("label", "severe"), ("label", "Non-structural crack"),
                             ("shape_type", "rectangle")):
            document = annotation()
            document["shapes"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.read(document)

    def test_rejects_mismatched_image_reference_and_dimensions(self):
        for field, value in (("imageWidth", 9), ("imageHeight", True), ("imagePath", "other.png")):
            document = annotation()
            document[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read(document)

    def test_rejects_invalid_geometry(self):
        bad_points = (
            [[1, 1], [2, 2]], [[1, 1], [2, 2], [3, 3]],
            [[-1, 0], [2, 0], [2, 2]], [[1, 0], [9, 0], [2, 2]],
            [[float("nan"), 1], [2, 0], [2, 2]], [[True, 0], [2, 0], [2, 2]],
        )
        for points in bad_points:
            document = annotation()
            document["shapes"][0]["points"] = points
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.read(document)

    def test_embedded_paths_are_not_followed(self):
        document = annotation()
        document["imagePath"] = "../../untrusted/sample.png"
        document["imageData"] = "not valid base64 and deliberately never decoded"
        self.assertEqual(len(self.read(document)), 1)

    @unittest.skipUnless(HAS_IMAGES, "Pillow and NumPy are not installed")
    def test_rasterization_counts_overlap_and_last_shape_wins(self):
        document = annotation()
        second = copy.deepcopy(document["shapes"][0])
        second["label"] = "Structural crack"
        document["shapes"].append(second)
        mask, overlap = rasterize(self.read(document), (8, 6))
        self.assertEqual(overlap, 16)
        self.assertEqual(mask.getpixel((2, 2)), 2)
        self.assertEqual(mask.getpixel((0, 0)), 0)

    @unittest.skipUnless(HAS_IMAGES, "Pillow and NumPy are not installed")
    def test_boundary_edges_clip_without_resizing(self):
        polygon = Polygon("Microcrack", ((0, 0), (8, 0), (8, 6), (0, 6)))
        mask, overlap = rasterize([polygon], (8, 6))
        self.assertEqual(mask.size, (8, 6))
        self.assertEqual(mask.histogram()[1], 48)
        self.assertEqual(overlap, 0)
