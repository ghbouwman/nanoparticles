#!/usr/bin/env bash

source venv/bin/activate
cd src/
./main.py &> ../output/main_err.txt
