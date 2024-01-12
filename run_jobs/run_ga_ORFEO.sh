#!/bin/bash
#SBATCH --partition=THIN 
#SBATCH --job-name=GA1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=GA1.out 

output_file=solutions/summary_test/ga1.txt

# Activate conda (Adjust with your path)
eval "$(conda shell.bash hook)"
conda activate HO_env

# Use find to get a list of all .txt files in the inst_test directory
inst_test='data/inst_test1'
file_list=($(find "$inst_test" -type f -name "*.txt"))

# Extract all instances
paths=("${file_list[@]}")

# Specify hyper-parameters
n_sol=15
sm='lr'
pr=70
alpha=15
beta=100
kn=20
knm=20
join_p=28

# Create file
echo "test1" > "$output_file"

for file in "${paths[@]}"
do
    # echo "$file" >> "$output_file"
    nodes_number=$(awk 'NR==1 {print $2}' "$file")
    edge_number=$(awk 'NR==1 {print $3}' "$file")
    k_value=$(printf "%.0f" "$(echo "($kn * $nodes_number)/100+($knm * $edge_number/$nodes_number)/100" | bc)")
    python_output=$(python GA.py $file -k $k_value --alpha 0.15 --beta 1 --iterations 20 --n_solutions $n_sol --selection_method $sm --perc_replace 0.7 --join_p_param 28)
    echo "$python_output" >> "$output_file"
done
