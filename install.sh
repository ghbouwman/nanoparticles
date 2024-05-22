#!/usr/bin/env bash

python3 -m venv npvenv
source npvenv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
