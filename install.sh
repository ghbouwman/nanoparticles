#!/usr/bin/env bash

mkdir output
mkdir output/figures

# Set up the venv
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --verbose -r requirements.txt
