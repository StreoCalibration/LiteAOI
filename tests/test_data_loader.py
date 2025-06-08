import os
from pathlib import Path
import unittest
import numpy as np
import cv2

from modules.data_loader import load_images_from_folder

class TestDataLoader(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()
