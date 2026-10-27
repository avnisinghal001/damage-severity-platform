"""Configuration paths are portable and keep raw data separate from outputs."""

import tempfile
import unittest
from pathlib import Path

from damage_analytics.config import load_config


class ConfigTests(unittest.TestCase):
    def test_paths_resolve_relative_to_configuration_not_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "config"
            folder.mkdir()
            path = folder / "task.toml"
            path.write_text('project_root = ".."\nraw_dir = "data/raw"\n'
                            'extracted_dir = "data/interim"\noutput_dir = "outputs"\n')
            config = load_config(path)
            self.assertEqual(config.raw_dir, (root / "data/raw").resolve())

    def test_rejects_output_overlapping_raw_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "task.toml"
            path.write_text('project_root = "."\nraw_dir = "data"\n'
                            'extracted_dir = "interim"\noutput_dir = "data/reports"\n')
            with self.assertRaises(ValueError):
                load_config(path)
