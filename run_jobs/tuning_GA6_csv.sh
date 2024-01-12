#!/bin/bash
#SBATCH --partition=THIN 
#SBATCH --job-name=GA6
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=12
#SBATCH --mem=200gb 
#SBATCH --time=02:00:00
#SBATCH --output=GA6.out 

# File to tune hyper-parameters of greedy construction heuristic
output_file=tuning_results/tuning6_GA.txt

# Activate conda (Adjust with your path)
eval "$(conda shell.bash hook)"
conda activate HO_env

# Use find to get a list of all .txt files in the inst_test directory
inst_tuning='data/inst_tuning6'
file_list=($(find "$inst_tuning" -type f -name "*.txt"))

# Extract all instances
paths=("${file_list[@]}")



# Create file
echo "tuning6" > "$output_file"

for file in "${paths[@]}"
do
    echo "$file" >> "$output_file"
    python_output=$(python GA_tuning.py "$file" --runtime '30m' --n_jobs -1)
    echo "$python_output" >> "$output_file"
done
