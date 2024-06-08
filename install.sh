#!/usr/bin/env bash

python3 -m venv nanoparticles_venv
source nanoparticles_venv/bin/activate
pip install --verbose -r requirements.txt
echo "NOTA BENE: please run the command 'source nanoparticles_venv/bin/activate' before using the program"
