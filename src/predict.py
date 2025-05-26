"""
Code to run U-Net inference on a single file
"""
from utils import read_config
from unet import load_pretrained
from data_transforms import get_data_transform_function
from dataset import DatasetGriddedReader
from zarr_utils import initialize_zarr_directory, append_to_zarr, create_xarray_ds_predictions

def run_unet_inference(config, checkpoint_path, device, input_file, output_file):
    """
    Run U-Net inference on a single file
    """
    # Read config yaml file
    config = read_config(config)

    ## Load model
    model = load_pretrained(checkpoint_path, config, late_meta_inject=False)
    model.to(device)

    ## Set up dataset
    data_transform_function = get_data_transform_function(config['data_transform'])
    dataset = DatasetGriddedReader(zarr_file=input_file,
                                   window_size=config['window_size'],
                                   frequencies=config['frequencies'],
                                   meta_channels=config['meta_channels'],
                                   patch_overlap=20,
                                   augmentation_function=None,
                                   data_transform_function=None)

    ## Initialize output zarr file
    start_ping, write_first_loop = initialize_zarr_directory(output_file, resume=False)


    ## Loop over all crops in the dataset

    