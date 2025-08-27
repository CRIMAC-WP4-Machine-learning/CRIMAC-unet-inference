import argparse
import os
import shutil
import subprocess
from pathlib import Path
from src.predict import run_unet_inference
import torch

# Print environment variables
print('CRIMAC-classifiers-unet')
commit_sha = os.getenv('COMMIT_SHA')
if commit_sha:
    print(f'commit_sha: {commit_sha}')
version_number = os.getenv('VERSION_NUMBER')
if version_number:
    print(f'version_number: {version_number}')

# Set up argument parser
parser = argparse.ArgumentParser(description='Process a single file with Korona')
parser.add_argument('--filename', required=True, type=str,
                    help='Name of the file to process')
args = parser.parse_args()
print(f"Processing file: {args.filename}")

# Create a copy of the input file in the scratch directory
filename = args.filename
shutil.copy(
    Path('/datain', filename), 
    Path('/scratchin', filename)
    )

# Run korona on the single file from the internal folder scratchin and write the
# nc file to the external folder scratchnc
cmdstr = ['/lsss-3.0.0/korona/KoronaCli.sh',
          'batch',
          '--cfs', '/app/CW.cfs',
          '--destination', '/scratchnc',
          '--source', '/scratchin']

subprocess.run(cmdstr, check=True)

print('Content of scratchnc : '+str(os.listdir('/scratchnc/sv')))

# Check for GPU device (should be changed to an ENV variable to enable any cuda device)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print('Avilable device      : '+device)
output_file = Path('/dataout', filename.replace('.raw', '_predictions.nc'))
print('Output file          : '+str(output_file))

# Run the inference code on the netcdf file in internal 'scratchnc' folder
run_unet_inference(config="/app/src/configs/config_brautaset.yaml", 
                   checkpoint_path="/modelweights/Olav_Unet_model.pt",
                   device=device,
                   input_file=Path(
                       '/scratchnc/sv', filename.split('.')[0]+'.nc'),
                   output_file=output_file
                   )
