FROM python:3.11
#FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# Install linux libraries
RUN apt-get update && apt-get install -y wget unzip libnetcdf-dev

# Set up versioning
ARG version_number
ARG commit_sha
ENV VERSION_NUMBER=$version_number
ENV COMMIT_SHA=$commit_sha
LABEL COMMIT_SHA=$commit_sha
LABEL VERSION_NUMBER=$version_number
LABEL LSSS_VERSION='lsss-3.0.0-20250204-0841-linux'

# Install Korona
RUN wget https://marec.no/downloads/lsss-3.0.0-20250204-0841/lsss-3.0.0-20250204-0841-linux.zip
RUN unzip lsss-3.0.0-20250204-0841-linux.zip
RUN rm lsss-3.0.0-20250204-0841-linux.zip
RUN unzip /lsss-3.0.0-20250204-0841/lsss-3.0.0-linux.zip -d /
RUN rm /lsss-3.0.0-20250204-0841/lsss-3.0.0-linux.zip
COPY KoronaCli.sh /lsss-3.0.0/korona/KoronaCli.sh

# Copy korona files
COPY CW.cds /app/
COPY CW.cfs /app/
COPY TransducerRanges.xml /app/

# Install python libraries & python code
#COPY requirements.txt /requirements.txt
#RUN pip install --no-cache-dir --upgrade pip && \
#    pip install --no-cache-dir -r requirements.txt

# Prepare for running		
COPY CRIMAC-unet-inference.py /app/CRIMAC-unet-inference.py
RUN mkdir /scratchin
RUN mkdir /scratchout

WORKDIR /app

# Run the script
ENTRYPOINT ["python", "-u", "/app/CRIMAC-unet-inference.py"]
