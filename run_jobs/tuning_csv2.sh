#!/bin/bash

# File to tune hyper-parameters of greedy construction heuristic
output_file=tuning_results/rc_tuning2.csv

# Activate conda (Adjust with your path)
source /Users/samuele/opt/anaconda3/envs/HO_env

# Use find to get a list of all .txt files in the inst_test directory
inst_tuning_dir='data/inst_tuning'
file_list=($(find "$inst_tuning_dir" -type f -name "*.txt"))

# Extract all instances
paths=("${file_list[@]}")

alpha_list=(60 80 100)
beta_list=(60 80 100)

# Number of measurements
num_measurements=1

# Create CSV header
echo "file_path,alpha,beta,k,measurement,runtime,score" > "$output_file"

for file in "${paths[@]}"
do
    nodes_number=$(awk 'NR==1 {print $2}' "$file")

    for alpha in "${alpha_list[@]}"
    do
        for beta in "${beta_list[@]}"
        do
            for k in 1 2 5 10 # percentage of initial nodes selected as k.
            do 
                alpha_prob=$(echo "scale=2; $alpha/100" | bc)
                beta_prob=$(echo "scale=2; $beta/100" | bc)
                k_value=$(printf "%.0f" "$(echo "$k * $nodes_number / 100" | bc)")

                # Run Python script multiple times and capture output
                for ((measurement=1; measurement<=num_measurements; measurement++))
                do
                    python_output=$(python main.py "$file" --alg rc --alpha "$alpha_prob" --beta "$beta_prob" -k "$k_value")

                    # Extract runtime and score from the Python script output (modify as needed)
                    runtime=$(echo "$python_output" | grep "Runtime:" | awk '{print $2}')
                    score=$(echo "$python_output" | grep "Score:" | awk '{print $2}')

                    # Output in CSV format
                    echo "$file,$alpha_prob,$beta_prob,$k_value,$measurement,$runtime,$score" >> "$output_file"
                done
            done
        done
    done
done
