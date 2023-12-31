#!/bin/bash 
#SBATCH --partition=EPYC 
#SBATCH --job-name=construct
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=200gb 
#SBATCH --time=02:00:00 

# File to tune hyper-parameters of greedy construction heuristic
output_file=summary_c

# Activate conda (Adjust with your path)
eval "$(conda shell.bash hook)"
conda activate HO_env

# Get a list of all the files in the test directory (it is a reduced subset)
inst_test_dir='data/inst_test'
file_list=($(find "$inst_test_dir" -type f -name "*.txt"))

# Extract all instances
paths=("${file_list[@]}")

alpha=1
beta=1
k=20 # 20 % of nodes
for file in "${paths[@]}"
do
    nodes_number=$(awk 'NR==1 {print $2}' "$file")
    alpha_prob=$(echo "scale=2; $alpha/100" | bc)
    beta_prob=$(echo "scale=2; $beta/100" | bc)
    k_value=$(printf "%.0f" "$(echo "$k * $nodes_number / 100" | bc)")

    python main.py --alg c --alpha "$alpha_prob" --beta "$beta_prob" -k "$k_value" "$file" >> $output_file
done

