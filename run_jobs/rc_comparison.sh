#!/bin/bash

# File to tune hyper-parameters of greedy construction heuristic
output_file=rc_comparison_ga.csv

# Activate conda (Adjust with your path)
# source /Users/samuele/opt/anaconda3/envs/HO_env

# Use find to get a list of all .txt files in the inst_test directory
inst_tuning_dir='data/inst_test/'
file_list=($(find "$inst_tuning_dir" -type f -name "*.txt"))

# Extract all instances
paths=("${file_list[@]}")

alpha=15
beta_list=100
kn=0.2
kmn=0.2

# Number of measurements: repeat 5 times: take 5 measurements
num_measurements=40
num_runs=5

# Create CSV header
echo "file_path,run_ID,meas_ID,score" >  "$output_file"

for file in "${paths[@]}"
do
    
    nodes_number=$(awk 'NR==1 {print $2}' "$file")
    edges_number=$(awk 'NR==1 {print $3}' "$file")
    k_value=$(printf "%.0f" "$(echo "$kn * $nodes_number / 100 + $kmn * $edges_number/$nodes_number" | bc)")

    for run_ID in 1 2 3 4 5
    do
        #for num_measurements in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
        for ((measurement=1; measurement<=40; measurement++))
        do
            python_output=$(python main.py "$file" --alg rc --alpha 0.15 --beta 1 -k "$k_value")
            score=$(echo "$python_output" | grep "Score:" | awk '{print $2}')
            # Output in CSV format
            echo "$file,$run_ID,$measurement,$score" >> "$output_file"
        done
    done

done
