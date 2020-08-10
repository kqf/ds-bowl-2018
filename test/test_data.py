import unittest
from model.data import DataManager


class TestData(unittest.TestCase):

    def test_reads_data(self):
        manager = DataManager("input", "stage1")
        self.assertIsNotNone(manager.images())
        self.assertIsNotNone(manager.masks())
