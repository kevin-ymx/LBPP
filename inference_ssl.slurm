#!/bin/bash
#SBATCH --job-name=inferssl
#SBATCH --account=choiseprojec
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --qos=high

# Kestrel: 4 nodes x 4 GPUs = 16 tasks (option B: srun, one task per GPU)
# Each task processes CSV files where index % 16 == SLURM_PROCID

# Load env / modules (adjust for Kestrel)
# module load conda
# conda activate your_env

export MPICH_GPU_SUPPORT_ENABLED=1

INPUT_DIR="./filtered_csv"
OUTPUT_DIR="./filtered_csv_embeddings"
WORLD_SIZE=16

srun --ntasks=${WORLD_SIZE} --cpu-bind=cores --gpus-per-task=1 bash -c '
  python inference_ssl.py \
    --filtered_csv "'"${INPUT_DIR}"'" \
    --output_dir "'"${OUTPUT_DIR}"'" \
    --rank ${SLURM_PROCID} \
    --world_size '"${WORLD_SIZE}"'
'

echo "Done."
