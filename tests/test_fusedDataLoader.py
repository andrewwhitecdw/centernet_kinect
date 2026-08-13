import os
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from dataLoader import fusedDataLoader


class TestFusedDataLoaderImreadFlags(unittest.TestCase):
    """Verify depth and IR images are loaded with cv2.IMREAD_UNCHANGED."""

    def test_depth_and_ir_use_imread_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ir_dir = os.path.join(tmpdir, "ir")
            depth_dir = os.path.join(tmpdir, "depth")
            ann_dir = os.path.join(tmpdir, "annotations")
            for d in (ir_dir, depth_dir, ann_dir):
                os.makedirs(d)

            img_name = "000000.png"
            fake_img = np.ones((100, 100), dtype=np.uint16) * 0xABCD
            cv2.imwrite(os.path.join(ir_dir, img_name), fake_img)
            cv2.imwrite(os.path.join(depth_dir, img_name), fake_img)

            ann_path = os.path.join(ann_dir, "train.json")
            with open(ann_path, "w") as f:
                json.dump([{
                    "img_path": f"foo/{img_name}",
                    "boxes": [[10, 10, 20, 20]],
                    "labels": [1]
                }], f)

            loader = fusedDataLoader.FusedProjDataLoader(
                json_annotation_path=ann_dir,
                train=True,
                ir_img_dir_path=ir_dir,
                depth_img_dir_path=depth_dir,
            )

            fake_box = np.array([[10, 10, 20, 20]], dtype=np.float32)
            fake_label = np.array([1])
            fake_heatmap = np.zeros((1, 80, 80), dtype=np.float32)

            with patch("dataLoader.fusedDataLoader.Transform") as mock_transform, \
                 patch("dataLoader.fusedDataLoader.CreateHeatMap") as mock_heatmap, \
                 patch("dataLoader.fusedDataLoader.cv2.imread") as mock_imread:
                mock_transform.return_value = (fake_img, fake_img, fake_box, fake_label)
                mock_heatmap.return_value = (fake_img, fake_img, fake_heatmap, fake_box)
                mock_imread.return_value = fake_img

                loader[0]

            self.assertEqual(len(mock_imread.call_args_list), 2)
            _, flags = zip(*[call.args for call in mock_imread.call_args_list])
            self.assertEqual(flags[0], cv2.IMREAD_UNCHANGED)
            self.assertEqual(flags[1], cv2.IMREAD_UNCHANGED)


if __name__ == "__main__":
