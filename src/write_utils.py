import os
import xarray as xr
import numpy as np
import shutil
from numcodecs import Blosc
import datetime



def initialize_dataset(target_dname, resume):
    if not resume:
        # Delete existing zarr dir of predictions
        if os.path.isdir(target_dname):
            print(f"Overwrite {target_dname}")
            shutil.rmtree(target_dname)
        write_first_loop = True
        start_ping = 0
    else:
        assert os.path.isdir(target_dname), \
            f"Cannot resume saving predictions as no existing prediction directory was fount at {target_dname}"
        print("Attempting to resume predictions")
        
        # Check if the directory is zarr or netCDF
        if os.path.exists(os.path.join(target_dname, 'zattrs')):
            print(f"Found zarr directory at {target_dname}")
            write_first_loop = False
            start_ping = xr.open_zarr(target_dname).sizes['ping_time']
        elif os.path.split(target_dname)[-1].endswith('.nc'):
            print(f"Found netCDF file at {target_dname}")
            write_first_loop = False
            start_ping = xr.open_dataset(target_dname).ping_time.size
        else:
            raise ValueError(f"Unknown format in {target_dname}. Expected zarr or netCDF.")
    return start_ping, write_first_loop


def append_to_dataset(ds, target_dname, write_first_loop):
    # Check if target is zarr or netcdf
    if os.path.exists(os.path.join(target_dname, 'zattrs')):
        # Re-chunk so that we have a full range in a chunk
        ds = ds.chunk({"range": ds.range.shape[0], "ping_time": "auto"})

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        encoding = {var: {"compressor": compressor} for var in ds.data_vars}
        if write_first_loop:
            ds.to_zarr(target_dname, mode="w", encoding=encoding)
        else:
            ds.to_zarr(target_dname, append_dim="ping_time")
    elif target_dname.endswith('.nc'):
        # For netCDF, we need to ensure the data is in float32 format
        ds["annotation"] = ds["annotation"].astype('float32')
        if write_first_loop:
            ds.to_netcdf(target_dname, mode="w")
        else:
            # Append to existing netCDF file
            ds.to_netcdf(target_dname, mode="a")


def create_xarray_ds_predictions(predictions, time_vector, range_vector, description):
    ds = xr.Dataset({"annotation": xr.DataArray(data=np.swapaxes(predictions, 1, 2),  # swap axes to match zarr
                                                dims=["category", "ping_time", "range"],
                                                coords={"category": [27, 1],
                                                        "ping_time": time_vector,
                                                        "range": range_vector})},
                    attrs={"description": description, 
                           "time": "{date:%Y-%m-%d %H:%M:%S}".format(date=datetime.datetime.now())}).astype(np.float16)
    return ds

