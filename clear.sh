#!/usr/bin/env bash

scancel -u $USER
rm slurm-*.out
cd output/
rm *.*
cd figures/
rm *.*
