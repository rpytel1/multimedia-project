#!/usr/bin/env bash
cd multimedia-project
sudo apt-get update
sudo apt-get -y install python3-pip
pip install -r requirements.txt
sudo apt-get install libsm6 libxrender1 libfontconfig1
mkdir data/our_jsons
python3 create_datasets.py
python3 dataset_statistics.py

