import xarray as xr
import numpy as np

class DatasetGriddedReader:
    """
    Grid a data reader, return regular gridded data patches
    """
    def __init__(self, zarr_file,
                 window_size,
                 frequencies,
                 meta_channels=[],
                 patch_overlap=20,
                 augmentation_function=None,
                 data_transform_function=None):
        
        self.ds = xr.open_dataset(zarr_file, decode_times=False)

        # assert that the frequencies in 'frequencies' are in the dataset
        assert all([f in self.ds.frequencies for f in frequencies]), \
            f"Some frequencies are not present in the dataset: {frequencies} not in {self.ds.data_vars}"
        
        self.num_pings = len(self.ds.ping_time)
        self.num_ranges = len(self.ds.range)

        self.window_size = window_size
        self.frequencies = frequencies
        self.meta_channels = meta_channels
        self.augmentation_function = augmentation_function
        self.data_transform_function = data_transform_function
        self.patch_overlap = patch_overlap

        if len(self.meta_channels) > 0:
            raise NotImplementedError("Meta channels are not implemented yet")
            # # Check valid meta_channels input
            # assert all([isinstance(cond, bool) for cond in self.meta_channels.values()])
            # assert set(self.meta_channels.keys()) == \
            #        {'portion_year', 'portion_day', 'depth_rel', 'depth_abs_surface', 'depth_abs_seabed', 'time_diff'}


        # Initialize sampler
        self.data_grid = get_data_grid(self.num_pings, 
                                       self.num_ranges,
                                       window_size,
                                       patch_overlap)

    def __len__(self):
        # Return the number of patches
        return len(self.data_grid)
    

    def get_crop(self, center_location):
        # Initialize crop
        boundary_val_data = 0
        out_data = np.ones(shape=(len(self.frequencies),
                                  self.window_size[0],
                                  self.window_size[1])) * boundary_val_data
        
        # Get upper left and lower right corners of patch in data
        patch_corners = np.array([[0, 0], [self.window_size[0], self.window_size[1]]])
        data_corners = patch_corners + center_location - self.window_size//2 + 1

        y0, x0 = data_corners[0]
        y1, x1 = data_corners[1]

        # retrieve the data (making sure not to go out of bounds)
        zarr_crop_x = (max(x0, 0), min(x1, self.num_pings))
        zarr_crop_y = (max(y0, 0), min(y1, self.num_ranges))

        # get the data
        data = self.ds.sv()

        # add to crop
        crop = [zarr_crop_x[0] - x0, self.window_size[0] - (x1 - zarr_crop_x[1]),
                zarr_crop_y[0] - y0, self.window_size[1] - (y1 - zarr_crop_y[1])]


        # outputshape freqs, y, x
        out_data[:, crop[2]:crop[3], crop[0]:crop[1]] = np.nan_to_num(data.swapaxes(1, 2), nan=boundary_val_data)
        return out_data


    def __getitem__(self, index):
        # Retrieve center location of the patch
        center_location = self.data_grid[index]
        
        # Load data
        data = self.get_crop(center_location)

        if self.augmentation_function is not None:
            # Apply augmentation
            data = self.augmentation_function(data)
        
        if self.data_transform_function is not None:
            # Apply data-transform-function
            data = self.data_transform_function(data)

        return data

def get_data_grid(num_pings, num_ranges, patch_size, patch_overlap):
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
    # Get coordinates for data area
    start_ping, end_ping = (0, num_pings)
    start_range, end_range = (0, num_ranges)

    (patch_width, patch_height) = patch_size

    # Get grid with center point of all patches
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
    return mesh

