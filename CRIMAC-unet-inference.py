import argparse
import os
import shutil
import subprocess
from pathlib import Path

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

# Run korona on the single file
cmdstr = ['/lsss-3.0.0/korona/KoronaCli.sh',
          'batch',
          '--cfs', '/app/CW.cfs',
          '--destination', '/dataout', # Replace dataout with scratchout
          '--source', '/scratchin']
print(cmdstr)
subprocess.run(cmdstr, check=True)

# Do the Unet think on the nc file in scratch out
