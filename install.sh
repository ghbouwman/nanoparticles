#!/usr/bin/env bash

python3 -m venv venv
source venv/bin/activate
pip install --verbose -r requirements.txt
echo "NOTA BENE: please run the command 'source venv/bin/activate' before using the program"
