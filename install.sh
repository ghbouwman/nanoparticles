#!/usr/bin/env bash

python3 -m venv npvenv
source npvenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
