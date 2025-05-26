import os
import xarray as xr
import numpy as np
import shutil
from numcodecs import Blosc
import datetime


def initialize_zarr_directory(target_dname, resume):
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
        start_ping = xr.open_zarr(target_dname).sizes['ping_time']
        write_first_loop = False
    return start_ping, write_first_loop


def append_to_zarr(ds, target_dname, write_first_loop):
    # Re-chunk so that we have a full range in a chunk
    ds = ds.chunk({"range": ds.range.shape[0], "ping_time": "auto"})

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {var: {"compressor": compressor} for var in ds.data_vars}

    if write_first_loop:
        ds.to_zarr(target_dname, mode="w", encoding=encoding)
    else:
        ds.to_zarr(target_dname, append_dim="ping_time")


def create_xarray_ds_predictions(reader, predictions, start_ping, end_ping, model_name):
    ds = xr.Dataset({"annotation": xr.DataArray(data=np.swapaxes(predictions, 1, 2),  # swap axes to match zarr
                                                dims=["category", "ping_time", "range"],
                                                coords={"category": [27, 1],
                                                        "ping_time": reader.time_vector[start_ping:end_ping],
                                                        "range": reader.range_vector})},
                    attrs={"description": f"{model_name} predictions", 
                           "time": "{date:%Y-%m-%d %H:%M:%S}".format(date=datetime.datetime.now())}).astype(np.float16)
    return ds