import unittest

from build_dataset import get_reduced_size


class TestBuildDataset(unittest.TestCase):
    def test_resizer(self):
        self.assertEqual((1024, 1024), get_reduced_size({'w': 1024, 'h': 1024}))
        self.assertEqual((1024, 1024), get_reduced_size({'w': 2048, 'h': 2048}))
        self.assertEqual((1024, 2048), get_reduced_size({'w': 1024, 'h': 2048}))

        self.assertEqual((512, 2048), get_reduced_size({'w': 1024, 'h': 4096}))
        self.assertEqual((2048, 512), get_reduced_size({'w': 4096, 'h': 1024}))

        self.assertEqual((100, 100), get_reduced_size({'w': 100, 'h': 100}))
