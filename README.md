# CRIMAC-datacompression
Unet predictions using Dockerized Korona


## Build image

Build the docker image: 

`docker build --build-arg=commit_sha=$(git rev-parse HEAD) --build-arg=version_number=$(git describe --tags) --no-cache --tag crimac-classifiers-unet .`

## Test container

Set the data root location to the env variable, e.g.  `export CRIMACSCRATCH='/home/nilsolav/crimacscratch/`, and download the test data from IMR via [S3](https://s3browser.hi.no/files/crimac/test_data/) and store at `$CRIMACSCRATCH/test_data'.

Run `python3 run_test.py`. Use the script as a starting point for further implementation.

## Parameters
Mandatory parameters are `-v $datain:/datain`, `-v $dataout:/dataout` and the file name to be 
processed coded as an env variable: `--env FILE_NAME=$filename`.




