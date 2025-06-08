import os
from pathlib import Path
import unittest
import numpy as np
import cv2

from modules.data_loader import load_images_from_folder

class TestDataLoaderAdditional(unittest.TestCase):
    def test_returns_empty_and_prints_warning_when_no_images(self):
        with unittest.mock.patch("builtins.print") as mock_print:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmpdir:
                Path(os.path.join(tmpdir, "a.txt")).write_text("not image")
                images = load_images_from_folder(tmpdir)
                self.assertEqual(images, [])
                mock_print.assert_any_call("경고: 유효한 이미지가 없습니다.")

    def test_skip_unreadable_image(self):
        with unittest.mock.patch("cv2.imread", return_value=None) as mock_read, \
             unittest.mock.patch("builtins.print") as mock_print:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmpdir:
                Path(os.path.join(tmpdir, "img.png")).write_bytes(b"fake")
                images = load_images_from_folder(tmpdir)
                self.assertEqual(images, [])
                self.assertTrue(mock_read.called)
                mock_print.assert_any_call(
                    f"경고: 이미지를 불러오지 못했습니다: {Path(tmpdir) / 'img.png'}"
                )


if __name__ == "__main__":
    unittest.main()
