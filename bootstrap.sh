#!/bin/bash
#
# Bootstrap script for P9 Fruit-App EMR Cluster
# This script installs Python dependencies required by the preprocessing.py script.
#

# set -e: This command ensures that the script will exit immediately if any command fails.
# set -x: This command prints each command to the logs before it is executed
set -e -x

# Update pip to the latest version
sudo python3 -m pip install --upgrade pip

# Install the required Python libraries using pip
# Penser à mettre nom des librairies entre "" pour ne pas avoir de soucis si caractères spéciaux
# pyspark is not installed as it is part of the EMR environment.
# Basé sur le requirements.txt sans les librairies de type
# Dependencies des librairies principales, déja sur EMR, non nécéssaires pour un script qui tourne sur EMR
# (e.g., ipykernel, matplotlib, jupyter_client, prompt_toolkit, etc.)


sudo python3 -m pip install \
    "absl-py==2.3.0" \
    "astunparse==1.6.3" \
    "certifi==2025.6.15" \
    "charset-normalizer==3.4.2" \
    "flatbuffers==25.2.10" \
    "gast==0.6.0" \
    "google-pasta==0.2.0" \
    "grpcio==1.73.1" \
    "h5py==3.14.0" \
    "idna==3.10" \
    "keras==3.10.0" \
    "libclang==18.1.1" \
    "Markdown==3.8.2" \
    "MarkupSafe==3.0.2" \
    "ml_dtypes==0.5.1" \
    "numpy==2.1.3" \
    "opt_einsum==3.4.0" \
    "optree==0.16.0" \
    "packaging==25.0" \
    "pandas==2.3.0" \
    "Pillow==11.3.0" \
    "protobuf==5.29.5" \
    "pyarrow==20.0.0" \
    "python-dateutil==2.9.0.post0" \
    "requests==2.32.4" \
    "six==1.17.0" \
    "tensorboard==2.19.0" \
    "tensorboard-data-server==0.7.2" \
    "tensorflow==2.19.0" \
    "termcolor==3.1.0" \
    "typing_extensions==4.14.0" \
    "urllib3==2.5.0" \
    "Werkzeug==3.1.3" \
    "wrapt==1.17.2"

# Exit with a success code to indicate the bootstrap action was successful
exit 0