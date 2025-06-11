import unittest
import numpy as np
from src.constants import LABEL_OVERLAP_VAL
from src.dataset import get_data_grid, DatasetGridded

TEST_FILE = "/nr/bamjo/jodata5/pro/crimac/rechunked_data/2011/S2011206/ACOUSTIC/GRIDDED/S2011206_sv.zarr"

class TestGrid(unittest.TestCase):
    def test_grid(self):
        # Test the get_data_grid function
        start_ping = 0
        end_ping = 1000
        start_range = 0
        end_range = 500
        window_size = np.array([128, 256])  # Patch height, patch width
        patch_overlap = 20

        grid = get_data_grid(start_ping, end_ping, start_range, end_range, window_size, patch_overlap)

        assert len(grid) > 0, "Grid should not be empty"
        
        # Check som coordinates
        assert grid[0][0] == window_size[0]//2 - patch_overlap - 1, "First grid point should be at the center of the first patch"
        assert grid[0][1] == window_size[1]//2 - patch_overlap - 1, "First grid point should be at the center of the first patch"
        assert grid[-1][0] > end_range - window_size[0]//2 - (patch_overlap + 1), "Last grid point should cover the last patch"
        assert grid[-1][1] > end_ping - window_size[1]//2 - (patch_overlap + 1), "Last grid point should cover the last patch"

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Initialize the dataset with a test file and parameters
        self.dataset = DatasetGridded(TEST_FILE, frequencies=[38000, 120000], window_size=(128, 256), patch_overlap=20)
        self.dataset.define_data_grid(start_ping=0, end_ping=1000, start_range=0, end_range=500)


    def test_a_few(self):
        for _ in range(20):
            out = self.dataset[0]
            assert out['data'].shape == (2, 128, 256), "Data shape should match the window size"
            assert out['labels'].shape == (128, 256), "Labels shape should match the window size"
            assert np.all(out['data'][:, 0:20, 0:20] == 0), "Data should be zero in the out of bounds area"
            assert np.all(out['data'][:, -20:, -20:] != 0), "Data should be zero if not out of bounds, even if overlapping"

    def test_dataset_out(self):

        assert len(self.dataset.data_grid) > 0, "Data grid should not be empty"

        out = self.dataset[0]

        assert out['data'].shape == (2, 128, 256), "Data shape should match the window size"
        assert out['labels'].shape == (128, 256), "Labels shape should match the window size"
        assert np.all(out['data'][:, 0:20, 0:20] == 0), "Data should be zero in the out of bounds area"
        assert np.all(out['data'][:, -20:, -20:] != 0), "Data should be zero if not out of bounds, even if overlapping"

        assert np.all(out['data'][0][out['labels'] == 0] != 0), "Data should exist everywhere where labels are 0"
        assert np.all(out['data'][1][out['labels'] == 0] != 0), "Data should exist everywhere where labels are 0"

    def test_labels_out(self):
        out = self.dataset[0]

        # Asser that labels correctly mask out of bonds data
        assert np.all(out['labels'][0:20, 0:20] == LABEL_OVERLAP_VAL), "Labels should mask overlap area"
        assert np.all(out['labels'][-20:, -20:] == LABEL_OVERLAP_VAL), "Labels should be 0 in the valid data area"
        assert np.all(out['labels'][20:-20, 20:-20:] == 0), "Labels should be 0 in the valid data area"

if __name__ == '__main__':
    unittest.main()