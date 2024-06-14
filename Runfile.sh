#!/usr/bin/env bash

source venv/bin/activate
source begin.sh
cd src/
./main.py &> ../output/${run_name}_err.txt
