import xarray as xr
import numpy as np
import os
from constants import LABEL_BOUNDARY_VAL, LABEL_OVERLAP_VAL


class DatasetGridded:
    """
    Grid a data reader, return regular gridded data patches
    """
    def __init__(self, data_path,
                 window_size,
                 frequencies,
                 meta_channels=[],
                 patch_overlap=20,
                 augmentation_function=None,
                 data_transform_function=None):
        """_summary_

        Args:
            zarr_file (_type_): _description_
            window_size (_type_): output window size of the patches (height, width)
            frequencies (_type_): _description_
            meta_channels (list, optional): _description_. Defaults to [].
            patch_overlap (int, optional): _description_. Defaults to 20.
            augmentation_function (_type_, optional): _description_. Defaults to None.
            data_transform_function (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """

        # assert that data_path is either .zarr or .nc file
        assert data_path.endswith('.zarr') or data_path.endswith('.nc'), \
            f"Data path must be a .zarr or .nc file, got {data_path}"
        
        
        if data_path.endswith('.zarr'):
            assert os.path.isdir(data_path), \
                f"Zarr file {data_path} does not exist. Please provide a valid zarr file."
            
            self.ds = xr.open_zarr(data_path)
        else:
            assert os.path.isfile(data_path), \
                f"NetCDF file {data_path} does not exist. Please provide a valid netCDF file."
            
            self.ds = xr.open_dataset(data_path)

        # assert that the frequencies in 'frequencies' are in the dataset
        assert all([f in self.ds.frequency for f in frequencies]), \
            f"Some frequencies are not present in the dataset: {frequencies} not in {self.ds.frequency}"
        
        self.num_pings = len(self.ds.ping_time)
        self.num_ranges = len(self.ds.range)
        self.start_ping = None
        self.end_ping = None
        self.start_range = None
        self.end_range = None

        self.window_size = np.array(window_size)
        self.frequencies = frequencies
        self.meta_channels = meta_channels
        self.augmentation_function = augmentation_function
        self.data_transform_function = data_transform_function
        self.patch_overlap = patch_overlap

        if len(self.meta_channels) > 0:
            raise NotImplementedError("Meta channels are not implemented yet")

        # Initialize sampler
        self.data_grid = self.define_data_grid()

    def __len__(self):
        # Return the number of patches
        return len(self.data_grid)

    def define_data_grid(self, start_ping=0, end_ping=None, start_range=0, end_range=None):
        """
        Define the data grid for the dataset
        :param start_ping: Start ping index
        :param end_ping: End ping index
        :param start_range: Start range index
        :param end_range: End range index
        """
        if end_ping is None:
            end_ping = self.num_pings
        if end_range is None:
            end_range = self.num_ranges

        self.start_ping = start_ping
        self.end_ping = end_ping
        self.start_range = start_range
        self.end_range = end_range

        # Get grid with center point of all patches
        self.data_grid = get_data_grid(start_ping, end_ping, start_range, end_range,
                                       self.window_size, self.patch_overlap)


    def get_crop(self, center_location):
        # Initialize crop
        boundary_val_data = 0
        out_data = np.ones(shape=(len(self.frequencies),
                                  self.window_size[0],
                                  self.window_size[1])) * boundary_val_data  # num_frequencies, height, width
        
        # Get upper left and lower right corners of patch in data
        patch_corners = np.array([[0, 0], [self.window_size[0], self.window_size[1]]])  # (y0, x0), (y1, x1)
        data_corners = patch_corners + center_location - self.window_size//2 + 1

        y0, x0 = data_corners[0]  
        y1, x1 = data_corners[1]

        # retrieve the data (making sure not to go out of bounds)
        zarr_crop_x = (max(x0, 0), min(x1, self.num_pings))
        zarr_crop_y = (max(y0, 0), min(y1, self.num_ranges))

        # get the data, shape is (num_frequencies, num_pings, num_ranges)
        data = self.ds.sv.sel(frequency=self.frequencies)[:, zarr_crop_x[0]:zarr_crop_x[1],
                                                          zarr_crop_y[0]:zarr_crop_y[1]].values

        # add to crop  
        crop = [zarr_crop_y[0] - y0, self.window_size[0] - (y1 - zarr_crop_y[1]),
                zarr_crop_x[0] - x0, self.window_size[1] - (x1 - zarr_crop_x[1])]
    
        # Swap axes to match the expected shape (num_frequencies, height, width)
        data = np.nan_to_num(data.swapaxes(1, 2), nan=boundary_val_data)

        # Add to out data patch
        out_data[:, crop[0]:crop[1], crop[2]:crop[3]] = data
        
        # Label the parts that are outside the data range
        label_crop_x = (max(x0, self.start_ping), min(x1, self.end_ping))
        label_crop_y = (max(y0, self.start_range), min(y1, self.end_range))
        label_crop = [label_crop_y[0] - y0, self.window_size[0] - (y1 - label_crop_y[1]),
                label_crop_x[0] - x0, self.window_size[1] - (x1 - label_crop_x[1])]
        
        out_labels = np.ones_like(out_data[0]) * LABEL_BOUNDARY_VAL
        out_labels[label_crop[0]:label_crop[1], label_crop[2]:label_crop[3]] = 0
        
        return out_data, out_labels


    def __getitem__(self, index):
        # Retrieve center location of the patch 
        center_location = self.data_grid[index]  # (y, x) coordinates of the center of the patch
        
        # Load data
        data, labels = self.get_crop(center_location)

        if self.augmentation_function is not None:
            # Apply augmentation
            data, labels, _ = self.augmentation_function(data, labels, self.frequencies)
        
        if self.data_transform_function is not None:
            # Apply data-transform-function
            data, labels, _ = self.data_transform_function(data, labels, self.frequencies)

        # Label the parts of the data that are overlapping and should be ignored
        out_labels = np.ones_like(data[0]) * LABEL_OVERLAP_VAL
        out_labels[self.patch_overlap:-self.patch_overlap, self.patch_overlap:-self.patch_overlap] = \
            labels[self.patch_overlap:-self.patch_overlap, self.patch_overlap:-self.patch_overlap]

        return {'data': data, 'labels': out_labels,
                'center_coordinates': np.array(center_location)}


def get_data_grid(start_ping, end_ping, start_range, end_range, patch_size, patch_overlap):
    """
    Get the center coordinates for a grid of data patches
    :param reader: Data reader
    :param patch_size: Size of the data patches
    :param patch_overlap: Nr of pixels of overlap between neighboring patches
    :param mode: Optionally ignore patches far away from fish schools
    :param max_depth: Maximum depth to consider (as the height of zarr files are much larger than the distance from
    surface to seabed)
    :return: list of center coordinates for the grid
    """
    (patch_height, patch_width) = patch_size

    # Get grid with center point of all patches
    # TODO check why original code uses patch_overlap + 1 
    ys_upper_left = np.arange(start_range - (patch_overlap + 1), end_range - (patch_overlap + 1),
                              step=patch_height - 2 * patch_overlap)
    xs_upper_left = np.arange(start_ping - (patch_overlap + 1), end_ping - (patch_overlap + 1),
                              step=patch_width - 2 * patch_overlap)

    # Get center coordinates of all grid points
    ys_center = ys_upper_left + patch_height // 2
    xs_center = xs_upper_left + patch_width // 2

    # Return center coordinates of all patches in the grid
    # Get all combinations of the coordinates
    mesh = np.array(np.meshgrid(ys_center, xs_center)).T.reshape(-1, 2)

    # Sort array by ping index
    mesh = mesh[np.argsort(mesh[:, 1])]
    return mesh

