import unittest
import numpy as np
from src.constants import LABEL_OVERLAP_VAL, LABEL_BOUNDARY_VAL, DATA_BOUNDARY_VAL
from src.dataset import get_data_grid, DatasetGridded

#TEST_FILE = "/nr/bamjo/jodata5/pro/crimac/rechunked_data/2011/S2011206/ACOUSTIC/GRIDDED/S2011206_sv.zarr"
TEST_FILE = "/nr/bamjo/jodata5/pro/crimac/data/2007/S2007205/ACOUSTIC/GRIDDED/sv/2007205-D20070505-T101116.nc"

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
        self.window_size = (128, 256)  # Patch height, patch width
        self.patch_overlap = 20
        self.end_range = 500
        self.end_ping = 1000
        self.dataset = DatasetGridded(TEST_FILE, frequencies=[38000, 120000], window_size=self.window_size, patch_overlap=self.patch_overlap)
        self.dataset.define_data_grid(start_ping=0, end_ping=self.end_ping, start_range=0, end_range=self.end_range)


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

    def test_labels_boundaries(self):
        out = self.dataset[-1]

        labels = out['labels']
        center_location = out['center_coordinates']
        window_size = np.array(self.window_size)
        patch_corners = np.array([[0, 0], [window_size[0], window_size[1]]])  # (y0, x0), (y1, x1)
        data_corners = patch_corners + center_location - window_size//2 + 1

        data_corners[1][0] = min(data_corners[1][0], self.end_range)
        data_corners[1][1] = min(data_corners[1][1], self.end_ping)
        data_corners[:, 0] -= data_corners[0, 0]
        data_corners[:, 1] -= data_corners[0, 1]

        assert np.all(labels[self.patch_overlap:data_corners[1, 0], self.patch_overlap:data_corners[1, 1]]) == 0, "Labels should be 0 in the valid data area"
        assert np.all(labels[data_corners[1, 0]:-self.patch_overlap, self.patch_overlap:-self.patch_overlap] == LABEL_BOUNDARY_VAL), "Labels should be LABEL_BOUNDARY_VAL in the area outside the grid, even if valid data exists"
        assert np.all(labels[self.patch_overlap:-self.patch_overlap, data_corners[1, 1]:-self.patch_overlap] == LABEL_BOUNDARY_VAL), "Labels should be LABEL_BOUNDARY_VAL in the area outside the grid, even if valid data exists"

        #assert np.all(out['data'] != DATA_BOUNDARY_VAL), "Data should not contain DATA_BOUNDARY_VAL, only LABEL_BOUNDARY_VAL"
        
if __name__ == '__main__':
    unittest.main()