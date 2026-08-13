import json
import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from dataLoader.dataLoader import ProjDataLoader


class TestDataLoaderImreadFlag(unittest.TestCase):
    def test_uses_imread_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_name = "0000.png"
            img_path = os.path.join(tmpdir, img_name)
            # Write a single-channel 16-bit grayscale image
            cv2.imwrite(img_path, np.full((64, 64), 100, dtype=np.uint16))

            annotation = [
                {
                    "img_path": f"ir/{img_name}",
                    "boxes": [[0, 0, 10, 10]],
                    "labels": [1],
                }
            ]
            json_dir = os.path.join(tmpdir, "annotations")
            os.makedirs(json_dir, exist_ok=True)
            with open(os.path.join(json_dir, "val.json"), "w") as f:
                json.dump(annotation, f)

            def mock_transform(image, bboxes, labels, train):
                return image[np.newaxis, ...], bboxes, labels

            def mock_create_heatmap(image, bboxes, labels):
                return image, np.zeros((2, 80, 80), dtype=np.float32), bboxes

            with patch("cv2.imread") as mock_imread:
                mock_imread.return_value = np.full((64, 64), 100, dtype=np.uint16)
                with patch("dataLoader.dataLoader.Transform", side_effect=mock_transform):
                    with patch(
                        "dataLoader.dataLoader.CreateHeatMap", side_effect=mock_create_heatmap
                    ):
                        loader = ProjDataLoader(
                            json_annotation_path=json_dir,
                            train=False,
                            ir_img_dir_path=tmpdir,
                            depth_img_dir_path=tmpdir,
                        )
                        _ = loader[0]

            self.assertEqual(mock_imread.call_count, 1)
            self.assertEqual(mock_imread.call_args[0][1], cv2.IMREAD_UNCHANGED)


if __name__ == "__main__":
    unittest.main()
