#!/usr/bin/env bash

for index in {0..9}; do
    run_name="run$index"
    echo $run_name > output/index.txt
    sbatch -N1 --tasks-per-node=1 --hint=multithread -c3 --mem-per-cpu=8G --job-name=$run_name --time="2-2" --output="output/$run_name.out" Runfile.sh
    sleep 5
done;
