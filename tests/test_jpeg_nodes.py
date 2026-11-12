"""CPU encoding tests; only ComfyUI's directory service is stubbed."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "jpeg_nodes.py"


class JPEGNodesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output"
        self.temp = self.root / "temp"
        self.output.mkdir()
        self.temp.mkdir()
        paths = types.ModuleType("folder_paths")
        paths.get_output_directory = lambda: str(self.output)
        paths.get_temp_directory = lambda: str(self.temp)
        paths.get_save_image_path = self.save_path
        self.paths = paths
        spec = importlib.util.spec_from_file_location("warper_jpeg_test", MODULE_PATH)
        self.nodes = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"folder_paths": paths}):
            spec.loader.exec_module(self.nodes)

    def save_path(self, prefix, directory, width, height):
        target = Path(directory) / prefix
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = list(target.parent.glob("*.jpg"))
        subfolder = str(target.parent.relative_to(directory))
        return str(target.parent), target.name, len(existing) + 1, subfolder, prefix

    def image_path(self, item):
        base = self.temp if item["type"] == "temp" else self.output
        return base / item["subfolder"] / item["filename"]

    def test_save_batch_produces_real_jpegs_with_comfy_ui_descriptors(self):
        images = torch.rand((2, 32, 48, 3))
        result = self.nodes.SaveJPEG().save_images(images, "nested/frame_%batch_num%")
        items = result["ui"]["images"]
        self.assertEqual(len(items), 2)
        for index, item in enumerate(items):
            self.assertEqual(item["type"], "output")
            self.assertEqual(item["subfolder"], "nested")
            self.assertIn(f"frame_{index}_", item["filename"])
            with Image.open(self.image_path(item)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (48, 32))
                self.assertEqual(image.mode, "RGB")
        self.assertFalse(list(self.root.rglob("*.png")))

    def test_preview_uses_only_temp_and_does_not_overwrite_previous_preview(self):
        images = torch.zeros((1, 8, 8, 3))
        node = self.nodes.PreviewJPEG()
        first = node.save_images(images)["ui"]["images"][0]
        second = node.save_images(images)["ui"]["images"][0]
        other_node = self.nodes.PreviewJPEG().save_images(images)["ui"]["images"][0]
        names = {item["filename"] for item in (first, second, other_node)}
        self.assertEqual(len(names), 3)
        for item in (first, second, other_node):
            self.assertEqual(item["type"], "temp")
            self.assertTrue(self.image_path(item).is_file())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_quality_changes_encoding(self):
        images = torch.from_numpy(np.random.default_rng(42).random((1, 64, 64, 3)).astype(np.float32))
        node = self.nodes.SaveJPEG()
        low = node.save_images(images, "low", quality=20)["ui"]["images"][0]
        high = node.save_images(images, "high", quality=95)["ui"]["images"][0]
        self.assertLess(self.image_path(low).stat().st_size, self.image_path(high).stat().st_size)
        with Image.open(self.image_path(high)) as image:
            # All components use the same sampling factors: 4:4:4.
            self.assertTrue(all(layer[1:3] == (1, 1) for layer in image.layer))

    def test_rgba_and_grayscale_inputs(self):
        for channels in (1, 4):
            with self.subTest(channels=channels):
                item = self.nodes.SaveJPEG().save_images(
                    torch.ones((1, 8, 8, channels)), f"channels_{channels}"
                )["ui"]["images"][0]
                with Image.open(self.image_path(item)) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_pixels_are_clipped_and_input_is_unchanged(self):
        images = torch.tensor([-0.5, 0.5, 1.5]).expand(1, 8, 8, 3).clone().requires_grad_()
        before = images.detach().clone()
        item = self.nodes.SaveJPEG().save_images(images, quality=100)["ui"]["images"][0]
        self.assertTrue(torch.equal(images, before))
        with Image.open(self.image_path(item)) as image:
            self.assertTrue(np.allclose(image.getpixel((0, 0)), (0, 127, 255), atol=2))

    def test_both_nodes_are_output_nodes_and_preview_has_no_filename_widget(self):
        for node_class in (self.nodes.SaveJPEG, self.nodes.PreviewJPEG):
            self.assertTrue(node_class.OUTPUT_NODE)
            self.assertTrue(callable(getattr(node_class(), node_class.FUNCTION)))
            self.assertEqual(node_class.INPUT_TYPES()["required"]["quality"][1]["default"], 95)
        self.assertNotIn("filename_prefix", self.nodes.PreviewJPEG.INPUT_TYPES()["required"])

    def test_package_import_registers_only_jpeg_nodes(self):
        package_root = MODULE_PATH.parents[1]
        spec = importlib.util.spec_from_file_location(
            "warper_package_test", package_root / "__init__.py",
            submodule_search_locations=[str(package_root)],
        )
        package = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "folder_paths": self.paths, "warper_package_test": package,
        }):
            spec.loader.exec_module(package)
            expected = {"SaveJPEG_Warper", "PreviewJPEG_Warper"}
            self.assertEqual(set(package.NODE_CLASS_MAPPINGS), expected)
            self.assertEqual(set(package.NODE_DISPLAY_NAME_MAPPINGS), expected)
            # Every registered node can be constructed without legacy dependencies.
            for node_class in package.NODE_CLASS_MAPPINGS.values():
                node = node_class()
                result = node.save_images(torch.zeros((1, 8, 8, 3)))
                self.assertTrue(self.image_path(result["ui"]["images"][0]).is_file())


if __name__ == "__main__":
    unittest.main()
