"""
Code to run U-Net inference on a single file
"""
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import load_pretrained
from data_transforms import get_data_transform_function
from dataset import DatasetGridded
from zarr_utils import initialize_zarr_directory, append_to_zarr, create_xarray_ds_predictions
from constants import LABEL_OVERLAP_VAL, LABEL_SEABED_MASK_VAL, LABEL_BOUNDARY_VAL, SANDEEL, OTHER
from utils import read_config


def get_data_split(valid_pings_ranges, max_n_pings=1000):
    """ Split data into smaller portions which can be preloaded """
    splits = []
    for start, end in valid_pings_ranges:
        n_splits = np.ceil((end - start) / max_n_pings)
        split_range = np.linspace(start, end, int(n_splits + 1)).astype(int)

        splits.extend([[split_range[i], split_range[i + 1]] for i in range(len(split_range) - 1)])
    return np.array(splits)


def patch_coord_to_data_coord(patch_coords, center_coord, patch_size):
    data_coord = patch_coords + center_coord - patch_size//2 + 1
    return data_coord.astype(int)


def fill_out_array(out_array, preds, labels, center_coordinates, ping_start):
    # TODO gather in post-processing step
    selected_label_idxs = np.argwhere((labels != LABEL_OVERLAP_VAL)  # Ignore overlap areas
                                      & (labels != LABEL_SEABED_MASK_VAL)  # Ignore areas under seabed
                                      & (labels != LABEL_BOUNDARY_VAL))  # Ignore areas outside data boundary

    if len(selected_label_idxs) == 0:
        return out_array

    # TODO better variable names
    y_label, x_label = np.transpose(selected_label_idxs)


    # Get corresponding coordinates in data
    data_coords = patch_coord_to_data_coord(np.array(selected_label_idxs),
                                            np.array(center_coordinates),
                                            np.array(labels.shape))
    y_array, x_array = np.transpose(data_coords)

    # adjust according to ping start time
    x_array -= ping_start
    out_array[:, y_array, x_array] = preds[[SANDEEL, OTHER]][:, y_label, x_label]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_unet_inference(config, checkpoint_path, device, input_file, output_file, 
                       batch_size=4, num_workers=4):
    """
    Run U-Net inference on a single file
    """
    # Read config yaml file
    config = read_config(config)
    description = config['description']
    categories = [cat for cat in config['model']['categories'] if cat != 0] # Remove background category

    ## Load model
    model = load_pretrained(checkpoint_path, config, late_meta_inject=False)
    model.to(device)
    model.eval()

    # Set up dataset
    data_transform_function = get_data_transform_function(config['data_transforms'])
    dataset = DatasetGridded(zarr_file=input_file,
        window_size=config['model']['patch_size'],
        frequencies=config['model']['frequencies'],
        meta_channels=config['model']['meta_channels'],
        patch_overlap=20,
        augmentation_function=None,
        data_transform_function=data_transform_function)

    # Initialize output zarr file
    start_ping, write_first_loop = initialize_zarr_directory(output_file, resume=False)

    # Fill inn parts of the output array at a time
    n_pings = dataset.num_pings

    # Get dataset chunk size
    chunk_size = np.max(dataset.ds.chunksizes['ping_time'])

    if chunk_size > 10000:
        split_size = 10000
    else:
        split_size = (10000 // chunk_size) * chunk_size

    splits = get_data_split([[start_ping, n_pings]], split_size)  # Split into smaller portions to avoid memory issues

    for (start_ping, end_ping) in tqdm(splits, total=len(splits), desc="Predicting patches"):
        # Select area in dataset
        dataset.define_data_grid(start_ping=start_ping, end_ping=end_ping)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            worker_init_fn=seed_worker)

        # Initialize output array
        out_array = np.zeros([len(categories), dataset.num_ranges, end_ping - start_ping])

        for batch in dataloader:
            # Get predictions for batch
            with torch.no_grad():
                predictions = model(batch['data'].float().to(device))

                # Get softmax
                boundary_labels = batch['labels'].cpu().numpy()  # Indicates which parts of the data should be ignored
                predictions = torch.softmax(predictions, dim=1).cpu().numpy()
                center_coordinates = batch['center_coordinates'].cpu().numpy()

                # Fill output array with predictions in the batch
                for patch in range(len(batch['center_coordinates'])):
                    pred_patch = predictions[patch]
                    center_coordinates_patch = center_coordinates[patch]
                    boundary_labels_patch = boundary_labels[patch]

                    fill_out_array(out_array, pred_patch, boundary_labels_patch, center_coordinates_patch, start_ping)

        # After array has been filled, create xarray dataset
        time_vector = dataset.ds.ping_time[start_ping:end_ping]
        range_vector = dataset.ds.range
        ds = create_xarray_ds_predictions(predictions=out_array, 
                                          time_vector=time_vector, 
                                          range_vector=range_vector, 
                                          description=description)

        # Write to zarr
        append_to_zarr(ds, output_file, write_first_loop)

        write_first_loop = False


if __name__ == "__main__":
    checkpoint_path = "/nr/project/bild/CRIMAC/Models/Olav_Unet_model.pt"
    input_file = "/nr/bamjo/jodata5/pro/crimac/rechunked_data/2011/S2011206/ACOUSTIC/GRIDDED/S2011206_sv.zarr"
    output_file = "S2011206_predictions.zarr"
    config = "/nr/bamjo/user/utseth/crimac/code/CRIMAC-unet-inference/src/configs/config_brautaset.yaml"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_unet_inference(config, checkpoint_path, device, input_file, output_file)