import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.color_retrieval.extraction import extract_grid, extract_metadata
from src.color_retrieval.palette import COLOR_NAMES
from src.color_retrieval.search import ColorIndex


class ColorRetrievalTests(unittest.TestCase):
    def test_solid_red_frame_extracts_red_in_every_cell(self):
        grid = extract_grid(Image.new("RGB", (50, 40), (220, 35, 35)))
        self.assertEqual(grid.shape, (25,))
        self.assertTrue(np.all(grid == COLOR_NAMES.index("red")))

    def test_extract_and_spatial_search_keep_metadata_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (50, 50), (220, 35, 35)).save(root / "red.jpg")
            split = Image.new("RGB", (50, 50), (35, 90, 200))
            split.paste((30, 150, 60), (0, 0, 10, 10))
            split.save(root / "split.png")
            metadata = pd.DataFrame([
                {"video_id": "red-video", "keyframe_id": 1, "keyframe_path": "red.jpg"},
                {"video_id": "split-video", "keyframe_id": 2, "keyframe_path": "split.png"},
            ])
            metadata_path = root / "metadata.csv"
            output_path = root / "colors.npz"
            metadata.to_csv(metadata_path, index=False)
            result = extract_metadata(metadata_path, output_path)
            self.assertEqual(result["failures"], [])
            records = metadata.to_dict(orient="records")
            index = ColorIndex(output_path, records)
            frame = index.search(records, [{"row": 0, "col": 0, "color": "green"}], 10)
            self.assertEqual(frame.iloc[0]["video_id"], "split-video")
            filtered = index.search(records, [{"row": 0, "col": 0, "color": "red"}], 10, ["split-video"])
            self.assertTrue(filtered.empty)

    def test_index_rejects_reordered_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "colors.npz"
            records = [{"video_id": "a", "keyframe_id": 1}, {"video_id": "b", "keyframe_id": 2}]
            np.savez(path, colors=np.zeros((2, 25), np.uint8), identities=np.asarray(["a\0" + "1", "b\0" + "2"]))
            with self.assertRaisesRegex(ValueError, "identities"):
                ColorIndex(path, records[::-1])


if __name__ == "__main__":
    unittest.main()
