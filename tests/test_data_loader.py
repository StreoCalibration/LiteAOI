import sys
import os
import unittest
import numpy as np
import cv2
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

# 프로젝트 루트 경로를 PYTHONPATH에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_loader import load_images_from_folder


class TestDataLoaderAdditional(unittest.TestCase):
    def test_returns_empty_and_prints_warning_when_no_images(self):
        with mock.patch("builtins.print") as mock_print:
            with TemporaryDirectory() as tmpdir:
                # 텍스트 파일 생성 (이미지가 아님)
                Path(os.path.join(tmpdir, "a.txt")).write_text("not image")
                images = load_images_from_folder(tmpdir)
                self.assertEqual(images, [])  # 빈 리스트 확인
                mock_print.assert_any_call("경고: 유효한 이미지가 없습니다.")

    def test_skip_unreadable_image(self):
        with mock.patch("cv2.imread", return_value=None) as mock_read, \
             mock.patch("builtins.print") as mock_print:
            with TemporaryDirectory() as tmpdir:
                fake_img_path = Path(os.path.join(tmpdir, "img.png"))
                fake_img_path.write_bytes(b"fake")  # 유효하지 않은 이미지 바이트 작성

                images = load_images_from_folder(tmpdir)
                self.assertEqual(images, [])  # 로드 실패 → 빈 리스트 반환
                self.assertTrue(mock_read.called)
                mock_print.assert_any_call(
                    f"경고: 이미지를 불러오지 못했습니다: {str(fake_img_path)}"
                )

    def test_load_images_sorted_and_ignore_invalid(self):
        with self.subTest("valid and invalid files"):
            with unittest.mock.patch('builtins.print'):
                from tempfile import TemporaryDirectory
                with TemporaryDirectory() as tmpdir:
                    # create images with different shapes to identify order
                    img_a = np.zeros((10, 10, 3), dtype=np.uint8)
                    img_b = np.ones((20, 20, 3), dtype=np.uint8) * 255
                    cv2.imwrite(os.path.join(tmpdir, 'a.png'), img_a)
                    cv2.imwrite(os.path.join(tmpdir, 'b.jpg'), img_b)
                    Path(os.path.join(tmpdir, 'c.txt')).write_text('not image')

                    images = load_images_from_folder(tmpdir)
                    self.assertEqual(len(images), 2)
                    self.assertEqual(images[0].shape, img_a.shape)
                    self.assertEqual(images[1].shape, img_b.shape)

    def test_nonexistent_folder(self):
        with self.assertRaises(FileNotFoundError):
            load_images_from_folder('no_such_folder')


if __name__ == "__main__":
    unittest.main()
